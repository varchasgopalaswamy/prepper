# -*- coding: utf-8 -*-
from __future__ import annotations

import copy
import datetime
import importlib.metadata
import os
import shutil
import tempfile
import traceback
import uuid
from abc import ABCMeta
from inspect import Parameter, signature
from typing import Any, Dict, List, Tuple

import h5py
import loguru
import numpy as np
import inspect

from prepper import H5StoreException
from prepper.caching import break_key, make_cache_name
from prepper.enums import H5StoreTypes
from prepper.utils import check_equality
__all__ = [
    "ExportableClassMixin",
]


def saveable_class(api_version: float, attributes: List[str] = None, functions:List[str] = None):
    if attributes is None:
        attributes = []
    if functions is None:
        functions = []

    def decorator(cls: ExportableClassMixin):

        if not issubclass(cls, ExportableClassMixin):
            raise ValueError('Only subclasses of ExportableClassMixin can be decorated with saveable_class')
        attribute_names = {}
        function_names = {}

        exportable_functions = []
        exportable_attributes = []


        for parent in reversed(inspect.getmro(cls)):
            if hasattr(parent, "_exportable_attributes"):
                for attr in parent._exportable_attributes:
                    attribute_names[attr] = parent
            if hasattr(parent, "_exportable_functions"):
                for fcn in parent._exportable_functions:
                    function_names[fcn] = parent

        for attr in attributes:
            attribute_names[attr] = cls

        for fcn in functions:
            function_names[fcn] = cls

        for name, clsname in attribute_names.items():
            if not hasattr(clsname, name):
                loguru.logger.warning(f"{clsname} does not have attribute {name}. This may mean that {name} is missing (which is an error), or a dynamically assigned attribute. Consider making {name} a property to avoid this warning.")

            exportable_attributes.append(f"{clsname.__name__}.{name}")
        for name, clsname in function_names.items():
            if not hasattr(clsname, name):
                raise ValueError(f"{clsname} does not have function {name}")
            exportable_functions.append(f"{clsname.__name__}.{name}")

        cls._exportable_functions = set(exportable_functions)
        cls._exportable_attributes = set(exportable_attributes)
        cls.api_version = api_version

        return cls

    return decorator

class ExportableClassMixin(object, metaclass=ABCMeta):
    """
    Allows the class to be saved and loaded from an HDF5 file

    """

    _constructor_args: Dict[str, Any] = {}
    api_version: float
    _exportable_attributes: List[str]
    _exportable_functions: List[str]

    def __copy__(self):
        return self.__class__(**self._constructor_args)

    def __deepcopy__(self, memo):
        args = {k: copy.deepcopy(v) for k, v in self._constructor_args.items()}
        return self.__class__(**args)

    def __new__(cls, *args, **kwargs):
        """Intercepts the __init__ call to store the initialization arguments within the instance so that they can be saved.
        The default arguments are not saved.
        """
        instance = object.__new__(cls)
        instance._constructor_args = {}
        instance.__init__(*args, **kwargs)
        sig = signature(instance.__init__)
        bound_args = sig.bind(*args, **kwargs)
        # bound_args.apply_defaults()
        for i, (key, value) in enumerate(bound_args.arguments.items()):
            if sig.parameters[key].kind == Parameter.POSITIONAL_ONLY:
                raise ValueError(
                    "Cannot save arguments that are positional only!"
                )
            if sig.parameters[key].kind == Parameter.VAR_KEYWORD:
                for kwkey, kwvalue in value.items():
                    instance._constructor_args[kwkey] = kwvalue
                continue
            instance._constructor_args[key] = value

        return instance

    def __eq__(self, other) -> bool:

        if type(self) != type(other):
            return False

        same = True
        # Check that attributes are the same
        for attr in self._exportable_attributes:
            mine = getattr(self, attr, None)
            theirs = getattr(other, attr, None)
            this_is_same = check_equality(mine, theirs)
            same &= this_is_same

        # Check that cached function calls are the same
        if hasattr(self, "_exportable_functions"):
            for symbol in self._exportable_functions:
                # This means it's a cached property, so save it

                mine = self.__dict__.get(symbol, None)
                theirs = other.__dict__.get(symbol, None)
                this_is_same = check_equality(mine, theirs)
                same &= this_is_same
        return bool(same)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @classmethod
    def from_hdf5(cls, path, group="/"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find file {path}")

        with h5py.File(path, mode="r", track_order=True) as hdf5_file:
            if group not in hdf5_file:
                msg = f"Failed to load {group} because it does not exist!"
                loguru.logger.error(msg)
                raise H5StoreException(msg)
            entry = hdf5_file[group]

            if cls.__name__ in entry:
                init_kw_type, init_kws = ExportableClassMixin._load_h5_entry(
                    path, f"{group}/{cls.__name__}"
                )
                if init_kw_type != H5StoreTypes.ClassConstructor:
                    raise H5StoreException()
            else:
                init_kws = {}
            obj = cls(**init_kws)
            obj._read_hdf5_contents(path, group=group)

        return obj

    def _read_hdf5_contents(self, file, group):
        from prepper.io_handlers import read_h5_attr

        with h5py.File(file, mode="r", track_order=True) as hdf5_file:
            if group not in hdf5_file:
                msg = f"Failed to load {group} because it does not exist!"
                loguru.logger.error(msg)
                raise H5StoreException(msg)
            base = hdf5_file[group]
            entry_type = ExportableClassMixin._get_group_type(base)
            if entry_type != H5StoreTypes.PythonClass:
                raise ValueError(
                    f"_read_hdf5_contents was called on a HDF5 group {group} that is not a python class spec!"
                )

            try:
                class_name = read_h5_attr(base, "class")
            except KeyError:
                msg = f"Failed to load {group} because the class name was not stored!"
                loguru.logger.error(msg)
                raise H5StoreException(msg)
            if class_name != self.__class__.__name__:
                msg = f"Failed to load {group} because the stored class name ({class_name}) is not the same as the loading class ({self.__class__.__name__})!"
                loguru.logger.error(msg)
                raise H5StoreException(msg)

            for entry_name in base:
                entry_type, entry_value = self._load_h5_entry(
                    file, f"{group}/{entry_name}"
                )
                if entry_type == H5StoreTypes.FunctionCache:
                    fname = make_cache_name(entry_name)
                    if fname not in self.__dict__:
                        self.__dict__[fname] = {}
                    for key, value in entry_value:
                        self.__dict__[fname][key] = value
                else:
                    self.__dict__[entry_name] = entry_value

    @staticmethod
    def _load_h5_entry(file: str, group: str) -> Tuple[H5StoreTypes, Any]:
        from prepper.io_handlers import load_custom_h5_type

        with h5py.File(file, mode="r", track_order=True) as hdf5_file:
            if group not in hdf5_file:
                raise FileNotFoundError(
                    f"Could not find {group} in the cached result!"
                )
            entry = hdf5_file[group]

            entry_type = ExportableClassMixin._get_group_type(entry)

            return entry_type, load_custom_h5_type(file, group, entry_type)


    def to_hdf5(self, path):
        """
        Save this object to an h5 file
        """

        if not os.path.splitext(path)[1] == ".hdf5":
            raise ValueError(
                f"HDF5 save objects must end in .hdf5! The provided path was {path}"
            )
        if os.path.exists(path):
            loguru.logger.warning(f"HDF5 file {path} exists... overwriting.")

        if not os.path.exists(os.path.dirname(path)):
            raise FileNotFoundError(
                f"The parent directory for {path} does not exist!"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, str(uuid.uuid1()))
            file = h5py.File(temp_file, "w")
            file.close()
            self._write_hdf5_contents(temp_file, group="/", existing_groups={})
            shutil.copyfile(src=temp_file, dst=path)

    def _write_hdf5_contents(self, file: str, group: str, existing_groups:Dict[str,Any], attributes=None):
        from prepper.io_handlers import (
            dump_class_constructor,
            dump_custom_h5_type,
            load_custom_h5_type,
            read_h5_attr,
            write_h5_attr,
        )


        existing_groups[group] = self

        if attributes is None:
            attributes = {}
        # Write this class' metadata
        with h5py.File(file, mode="a", track_order=True) as hdf5_file:
            my_group = hdf5_file.require_group(group)
            code_name = self.__class__.__module__.split(".")[0]
            attributes["module"] = self.__class__.__module__
            attributes["class"] = self.__class__.__name__
            attributes["timestamp"] = datetime.datetime.now().isoformat()
            if code_name == '__main__':
                attributes["version"] = ""
            else:
                attributes["version"] = importlib.metadata.version(code_name)
            attributes["code"] = code_name
            attributes["type"] = H5StoreTypes.PythonClass.name
            attributes["api_version"] = self.api_version

            # Write out the attributes
            for k, v in attributes.items():
                try:
                    write_h5_attr(my_group, k, v)
                except H5StoreException:
                    msg = f"Failed to write attribute {k} to group {group}! Exception was \n {traceback.format_exc()}"
                    loguru.logger.error(msg, exc_info=True)

        # Store the class constructor arguments, if applicable
        if len(self._constructor_args) > 0:
            existing_groups = dump_class_constructor(file, group, self, existing_groups)

        # Save the attributes, if populated
        for attribute in self._exportable_attributes:
            if hasattr(self, attribute):
                existing_groups = self._dump_h5_entry(
                    file, f"{group}/{attribute}", getattr(self, attribute), existing_groups
                )
        # Save the marked cached properties and function calls
        if hasattr(self, "_exportable_functions"):
            for symbol in self._exportable_functions:
                # This means it's a cached property, so save it
                if symbol in self.__dict__:
                    classname, value = self.__dict__[symbol].split(".")
                    if classname == self.__class__.__name__:
                        existing_groups = self._dump_h5_entry(file, f"{group}/{symbol}", value, existing_groups)
                # This means its a cached function call, so save each call
                elif make_cache_name(symbol) in self.__dict__:
                    function_cache = self.__dict__[make_cache_name(symbol)]
                    with h5py.File(
                        file, mode="a", track_order=True
                    ) as hdf5_file:
                        my_group = hdf5_file.require_group(group)
                        function_group = my_group.require_group(symbol)
                        write_h5_attr(
                            function_group,
                            "type",
                            H5StoreTypes.FunctionCache.name,
                        )

                    n_cache_items = max(len(function_cache), 1)
                    pad_digits = max(int(np.log10(n_cache_items)), 0) + 1
                    for idx, (key, value) in enumerate(
                        self.__dict__[make_cache_name(symbol)].items()
                    ):
                        pad_number = str(idx + 1).zfill(pad_digits)
                        key_args, key_kwargs = break_key(key)
                        if len(key_args) == 1 ^ len(key_kwargs) == 1:
                            if len(key_args) == 1:
                                group_name = f"{group}/{symbol}/{key_args[0]}"
                            else:
                                group_name = f"{group}/{symbol}/{key_kwargs.values()[0]}"
                        else:
                            group_name = (
                                f"{group}/{symbol}/{symbol}_call{pad_number}"
                            )
                        attrs = {"call_args": key_args}
                        for k, v in key_kwargs.items():
                            attrs[f"call_kw_{k}"] = v
                        existing_groups = self._dump_h5_entry(
                            file, group_name, value, existing_groups, attributes=attrs
                        )

        return existing_groups

    @staticmethod
    def _dump_h5_entry(
        file: str,
        entry_name: str,
        value: Any,
        existing_groups:Dict[str,Any],
        attributes: Dict[str, Any] = None,
    ):
        from prepper.io_handlers import dump_custom_h5_type, write_h5_attr

        entry_name = entry_name.replace("//", "/").strip()
        if attributes is None:
            attributes = {}

        try:
            existing_groups = dump_custom_h5_type(file, entry_name, value, existing_groups)
        except H5StoreException:
            msg = f"Group {entry_name} is an object of type {type(value)} that does not support being saved to an HDF5 file!"
            loguru.logger.error(msg, exc_info=True)
            raise ValueError(msg)

        # Add any attributes that were needed
        with h5py.File(file, mode="a", track_order=True) as hdf5_file:
            new_entry = hdf5_file[entry_name]
            for k, v in attributes.items():
                try:
                    write_h5_attr(new_entry, k, v)
                except H5StoreException:
                    msg = (
                        f"Failed to write attribute {k} to group {new_entry}!"
                    )
                    loguru.logger.error(msg)

        return existing_groups

    @staticmethod
    def _get_group_type(group):
        try:
            entry_type = group.attrs["type"]
        except KeyError:
            msg = f"Failed to load {group} because the group type was not stored!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        try:
            entry_type = H5StoreTypes[group.attrs["type"]]
        except (KeyError, TypeError):
            msg = (
                f"Failed to load {group} because the group type is not in the enumeration of valid types! Valid types are "
                + ", ".join([e.name for e in H5StoreTypes])
            )
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        return entry_type


if __name__ == '__main__':
    from prepper import saveable_class, ExportableClassMixin

    @saveable_class("0.0.1", save=['test_string','test_array','test_array2'])
    class SimpleSaveableClass(ExportableClassMixin):
        """
        A simple saveable class, used to test saving as an attribute of another
        saveable class
        """
        def __init__(self):
            self.test_string = 'test string SimpleSaveableClass'
            self.test_array = np.random.random(size=(1000,1000))
            self.test_array2 = self.test_array

    self = SimpleSaveableClass()
    self.to_hdf5('/b1/vgop/test.hdf5')