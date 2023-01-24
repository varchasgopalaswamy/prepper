# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import importlib
import inspect
import re
import traceback
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, List, Sequence, Type, TYPE_CHECKING

import h5py
import loguru
import numpy as np

from prepper import H5StoreException
from prepper.enums import H5StoreTypes
if TYPE_CHECKING:
    from prepper.exportable import ExportableClassMixin
__all__ = [
    'dump_custom_h5_type',
    'read_h5_attr',
    'write_h5_attr',
    'load_custom_h5_type',
    'dump_class_constructor',
    'saveable_class'
]

_NONE_TYPE_SENTINEL = "__python_None_sentinel__"
PYTHON_BASIC_TYPES = (int, float, str)
NUMPY_NUMERIC_TYPES = (np.int32, np.int64, np.float32, np.float64)
CUSTOM_H5_WRITERS = {}
CUSTOM_H5_LOADERS = {}
HDF5_COMPRESSION = {}


def write_h5_attr(base: h5py.Group, name: str, value: Any):

    if value is None:
        base.attrs[name] = _NONE_TYPE_SENTINEL
    else:
        try:
            base.attrs[name] = value
        except TypeError:
            if isinstance(value, Iterable) and all(
                [isinstance(v, PYTHON_BASIC_TYPES) for v in value]
            ):
                try:
                    base.attrs[name] = np.asarray(value)
                except TypeError:
                    raise H5StoreException(
                        f"Could not write attribute {name} with type {type(value)}!"
                    )
            raise H5StoreException(
                f"Could not write attribute {name} with type {type(value)}!"
            )


def read_h5_attr(base: h5py.Group, name: str):
    try:
        value = base.attrs[name]
    except KeyError:
        msg = f"Could not find attribute {name} in group {base.name}!"
        raise KeyError(msg)
    else:
        if value == _NONE_TYPE_SENTINEL:
            value = None
        return value


def saveable_class(api_version: float, save: List[str] = None):
    if save is None:
        save = []

    def decorator(cls:ExportableClassMixin):

        attribute_names = [s for s in save]
        function_names = []
        for parent in inspect.getmro(cls):
            if hasattr(parent, "_exportable_attributes"):
                attribute_names += parent._exportable_attributes
            for methodname in dir(parent):
                method = getattr(parent, methodname)
                if (
                    hasattr(method, "_export_values")
                    and methodname not in function_names
                    and method._export_values
                ):
                    function_names.append(methodname)

        cls._exportable_functions = set(function_names)
        cls._exportable_attributes = set(attribute_names)
        cls.api_version = api_version

        return cls

    return decorator


def _register(store, validator):
    def decorator(func):
        store[func.__name__] = (validator, func)
        return func

    return decorator


def key_to_group_name(key):
    if isinstance(key, Enum):
        name = key.name
        if "/" in name:
            return None
        else:
            return name
    else:
        try:
            name = str(key)
        except Exception:
            return None
        else:
            if "/" in name:
                return None
            else:
                return name


def dump_custom_h5_type(file: str, group: str, value: Any):
    for validator, writer in CUSTOM_H5_WRITERS.values():
        if validator(value):
            attrs = writer(file, group, value)

            with h5py.File(file, mode="a", track_order=True) as hdf5_file:
                try:
                    entry = hdf5_file[group]
                except KeyError:
                    entry = hdf5_file.require_group(group)
                for k, v in attrs.items():
                    try:
                        write_h5_attr(entry, k, v)
                    except H5StoreException:
                        msg = (
                            f"Failed to write attribute {k} to group {group}!"
                        )
                        loguru.logger.error(msg, exc_info=True)

            return
    raise H5StoreException(
        f"None of the custom HDF5 writer functions supported a value of type {type(value)}!"
    )


def load_custom_h5_type(file: str, group: str, entry_type: H5StoreTypes):
    for loader_type, loader in CUSTOM_H5_LOADERS.values():
        if loader_type == entry_type:
            return loader(file, group)
    msg = f"No loader found for custom HDF5 store type {entry_type}!"
    loguru.logger.error(msg, exc_info=True)
    raise H5StoreException(msg)


#### NONE ####
@_register(CUSTOM_H5_WRITERS, lambda x: x is None)
def dump_None(file: str, group: str, value: None) -> Dict[str, Any]:
    attributes = {}
    attributes["type"] = H5StoreTypes.Null.name
    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.Null)
def load_None(file: str, group: str):
    return None


#### HDF5 Group ######
@_register(CUSTOM_H5_WRITERS, lambda x: isinstance(x, h5py.Group))
def dump_hdf5_group(
    file: str, group: str, value: h5py.Group, version:str = '', code_name:str = ''
) -> Dict[str, Any]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        value.copy(source="/", dest=new_entry, name="value")
        attributes["type"] = H5StoreTypes.HDF5Group.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()
        attributes["version"] = version
        attributes["code"] = code_name

    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.HDF5Group)
def load_hdf5_group(file: str, group: str):
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        return hdf5_file[group]["value"]


#### python class ####
@_register(
    CUSTOM_H5_WRITERS, lambda x: issubclass(type(x), ExportableClassMixin)
)
def dump_exportable_class(
    file: str, group: str, value: Type[ExportableClassMixin]
) -> Dict[str, Any]:
    attributes = {}
    value._write_hdf5_contents(
        file,
        group=group,
    )
    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.PythonClass)
def load_exportable_class(file: str, group: str) -> Type[ExportableClassMixin]:
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        if not "module" in entry.attrs:
            msg = f"Failed to load {group} because it was a class entry, but didnt have the class module path in the attribute list!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        class_module = importlib.import_module(entry.attrs["module"])
        if not "class" in entry.attrs:
            msg = f"Failed to load {group} because it was a class entry, but didnt have the class name in the attribute list!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        class_definition = getattr(class_module, entry.attrs["class"])
        return class_definition.from_hdf5(file, group)



#### generic HDF5 dataset ####
@_register(
    CUSTOM_H5_WRITERS,
    lambda x: isinstance(x, PYTHON_BASIC_TYPES)
    or isinstance(x, NUMPY_NUMERIC_TYPES)
    or isinstance(x, np.ndarray)
    or (
        isinstance(x, Iterable)
        and (not isinstance(x, dict))
        and all(
            [
                isinstance(v, PYTHON_BASIC_TYPES + NUMPY_NUMERIC_TYPES)
                for v in x
            ]
        )
    ),
)
def dump_python_types_or_ndarray(
    file: str, group: str, value: Any
) -> Dict[str, Any]:

    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        try:
            new_entry = hdf5_file.create_dataset(name=group, data=value)
        except TypeError:
            msg = f"Failed to store {group} because it was of type {type(value)} which is not natively supported in HDF5!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        attributes["type"] = H5StoreTypes.HDF5Dataset.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()

    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.HDF5Dataset)
def load_python_types_or_ndarray(file: str, group: str):
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        if not isinstance(entry, h5py.Dataset):
            msg = f"Failed to load {group} because it was of type {type(entry)} and not an HDF5 Dataset!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        try:
            return entry.asstr("utf-8")[()]
        except TypeError:
            return entry[()]


#### class constructor ####
def dump_class_constructor(
    file: str, group: str, value: ExportableClassMixin
) -> Dict[str, Any]:
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        my_group = hdf5_file.require_group(group)
        constructor_group = my_group.require_group(value.__class__.__name__)
        constructor_attributes = {}
        constructor_attributes["module"] = value.__class__.__module__
        constructor_attributes["class"] = value.__class__.__name__
        constructor_attributes[
            "timestamp"
        ] = datetime.datetime.now().isoformat()
        constructor_attributes["type"] = H5StoreTypes.ClassConstructor.name
        # Write out the attributes
        for k, v in constructor_attributes.items():
            try:
                write_h5_attr(constructor_group, k, v)
            except H5StoreException:
                msg = f"Failed to write attribute {k} to group {constructor_group}! Exception was \n {traceback.format_exc()}"
                loguru.logger.error(msg, exc_info=True)

    for k, v in value._constructor_args.items():
        value._dump_h5_entry(
            file, f"{group}/{value.__class__.__name__}/{k}", v
        )


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.ClassConstructor)
def load_class_constructor(file: str, group: str):
    kwargs = {}
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        my_group = hdf5_file[group]
        for key in my_group:
            kwargs[key] = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{key}"
            )[1]

    return kwargs


#### python dict ####
@_register(CUSTOM_H5_WRITERS, lambda x: isinstance(x, dict))
def dump_dictionary(file: str, group: str, value: Any) -> Dict[str, Any]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        _ = hdf5_file.require_group(group)
        n_cache_items = len(value)
        if n_cache_items > 0:
            pad_digits = max(int(np.log10(n_cache_items)), 0) + 1
            for idx, (key, this_value) in enumerate(value.items()):
                pad_number = str(idx + 1).zfill(pad_digits)
                coerced_name = key_to_group_name(key)
                if coerced_name is not None:
                    group_name = f"{group}/{coerced_name}"

                if coerced_name is None or group_name in hdf5_file[group]:
                    group_name = f"{group}/{group}_{pad_number}"

                if issubclass(type(this_value), ExportableClassMixin):
                    this_value._write_hdf5_contents(
                        file, f"{group_name}/value"
                    )
                else:
                    ExportableClassMixin._dump_h5_entry(
                        file,
                        f"{group_name}/value",
                        this_value,
                    )

                if issubclass(type(key), ExportableClassMixin):
                    key._write_hdf5_contents(file, f"{group_name}/key")
                else:
                    ExportableClassMixin._dump_h5_entry(
                        file,
                        f"{group_name}/key",
                        key,
                    )

    attributes["type"] = H5StoreTypes.Dictionary.name
    attributes["timestamp"] = datetime.datetime.now().isoformat()
    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.Dictionary)
def load_dictionary(file: str, group: str):
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        basename = group.split("/")[-1]
        items = [k for k in entry]

        results = {}

        for item in items:
            item_group = entry[item]

            _, item_result = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{item}/value"
            )

            _, item_key = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{item}/key"
            )

            results[item_key] = item_result

    return results


#### python sequence ####
@_register(CUSTOM_H5_WRITERS, lambda x: isinstance(x, Iterable))
def dump_generic_sequence(
    file: str, group: str, value: Sequence[Any]
) -> Dict[str, Any]:
    from prepper.exportable import ExportableClassMixin

    basename = group.split("/")[-1]
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        _ = hdf5_file.require_group(group)
        n_cache_items = len(value)
        if n_cache_items > 0:
            pad_digits = max(int(np.log10(n_cache_items)), 0) + 1
            for idx, this_value in enumerate(value):
                pad_number = str(idx + 1).zfill(pad_digits)
                group_name = f"{group}/{basename}_{pad_number}"
                if issubclass(type(this_value), ExportableClassMixin):
                    this_value._write_hdf5_contents(file, group_name)
                else:
                    ExportableClassMixin._dump_h5_entry(
                        file, group_name, this_value
                    )

    attributes["type"] = H5StoreTypes.Sequence.name
    attributes["timestamp"] = datetime.datetime.now().isoformat()
    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.Sequence)
def load_generic_sequence(file: str, group: str) -> List[Any]:
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        basename = group.split("/")[-1]
        items = [k for k in entry]
        malformed_entries = [
            k for k in items if not re.match(rf"{basename}_[0-9]+", k)
        ]
        if len(malformed_entries) > 0:
            msg = f"Failed to load {group} because it was a dictionary, but contained non-item groups {','.join(malformed_entries)}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        results = []

        for item in items:
            item_group = entry[item]

            _, item_result = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{item}"
            )

            results.append(item_result)
    return results


#### python enum ####
@_register(CUSTOM_H5_WRITERS, lambda x: isinstance(x, Enum))
def dump_python_enum(
    file: str, group: str, value: Type[Enum]
) -> Dict[str, Any]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        new_entry["enum_class"] = str(value.__class__.__name__)
        new_entry["enum_module"] = str(value.__class__.__module__)
        new_entry["enum_value"] = str(value.name)
        attributes["type"] = H5StoreTypes.Enumerator.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()

    return attributes


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.Enumerator)
def load_python_enum(file: str, group: str) -> Type[Enum]:
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        try:
            enum_class_ = entry["enum_class"][()].decode("utf-8")
            enum_value_ = entry["enum_value"][()].decode("utf-8")
            enum_module_ = entry["enum_module"][()].decode("utf-8")
        except KeyError:
            msg = f"Failed to load {group} because it was a enum entry, but didnt have the enum name or value!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)

        enum_module = importlib.import_module(enum_module_)
        enum_class = getattr(enum_module, enum_class_, None)
        if enum_class is None:
            msg = f"Failed to load {group} because it was a enum entry, but the enum module {enum_module_} did not have the enum class {enum_class_}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        try:
            return enum_class[enum_value_]
        except KeyError:
            msg = f"Failed to load {group} because it was a enum entry, but the enum class {enum_class_} did not have the enum entry {enum_value_}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
