# -*- coding: utf-8 -*-
from __future__ import annotations

import datetime
import importlib
import re
import traceback
from collections.abc import Iterable
from enum import Enum
from typing import Any, Dict, List, Sequence, Type

import h5py
import loguru
import numpy as np

from prepper import H5StoreException
from prepper.caching import _HashedSeq, _make_key
from prepper.enums import H5StoreTypes
from prepper.exportable import ExportableClassMixin
from prepper.utils import check_equality

__all__ = [
    "dump_custom_h5_type",
    "read_h5_attr",
    "write_h5_attr",
    "load_custom_h5_type",
    "dump_class_constructor",
    "register_loader",
    "register_writer",
    "get_hdf5_compression",
    "set_hdf5_compression",
]

_NONE_TYPE_SENTINEL = "__python_None_sentinel__"
_EMPTY_TYPE_SENTINEL = "__python_Empty_sentinel__"

PYTHON_BASIC_TYPES = (int, float, str)
NUMPY_NUMERIC_TYPES = (np.int32, np.int64, np.float32, np.float64)
DEFAULT_H5_WRITERS = {}
DEFAULT_H5_LOADERS = {}
CUSTOM_H5_WRITERS = {}
CUSTOM_H5_LOADERS = {}
HDF5_COMPRESSION = {
    "compression": "gzip",
    "compression_opts": 9,
    "shuffle": True,
    "fletcher32": True,
}


def get_hdf5_compression():
    return HDF5_COMPRESSION


def set_hdf5_compression(compression: Dict[str, Any]):
    global HDF5_COMPRESSION
    HDF5_COMPRESSION = compression


def write_h5_attr(base: h5py.Group, name: str, value: Any):
    if value is None:
        base.attrs[name] = _NONE_TYPE_SENTINEL
    else:
        try:
            base.attrs[name] = value
        except TypeError as exc:
            if isinstance(value, Iterable) and all(
                isinstance(v, PYTHON_BASIC_TYPES) for v in value
            ):
                try:
                    base.attrs[name] = np.asarray(value)
                except TypeError as exc2:
                    raise H5StoreException(
                        f"Could not write attribute {name} with type {type(value)}!"
                    ) from exc2
            raise H5StoreException(
                f"Could not write attribute {name} with type {type(value)}!"
            ) from exc


def read_h5_attr(base: h5py.Group, name: str):
    try:
        value = base.attrs[name]
    except KeyError as exc:
        msg = f"Could not find attribute {name} in group {base.name}!"
        raise KeyError(msg) from exc

    if value == _NONE_TYPE_SENTINEL:
        value = None
    return value


def register_loader(validator):
    def decorator(func):
        CUSTOM_H5_LOADERS[func.__name__] = (validator, func)
        return func

    return decorator


def register_writer(validator):
    def decorator(func):
        CUSTOM_H5_WRITERS[func.__name__] = (validator, func)
        return func

    return decorator


def dump_custom_h5_type(
    file: str, group: str, value: Any, existing_groups: Dict[str, Any]
):
    writers = {}
    writers.update(CUSTOM_H5_WRITERS)
    writers.update(DEFAULT_H5_WRITERS)
    # Check to see if this class has already been written to the file
    class_already_written = False
    clone_group = None
    for k, v in existing_groups.items():
        try:
            is_equal = check_equality(value, v)
        except Exception:
            is_equal = False
        if is_equal:
            class_already_written = True
            clone_group = (
                k  # This is the group that this class is already written to
            )
            break
    if class_already_written:
        # This class has already been written to the file, so we just need to write a reference to it
        with h5py.File(file, mode="a", track_order=True) as hdf5_file:
            hdf5_file[group] = h5py.SoftLink(clone_group)
        return existing_groups

    for validator, writer in writers.values():
        if validator(value):
            attrs, existing_groups = writer(
                file, group, value, existing_groups
            )

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

            return existing_groups
    raise H5StoreException(
        f"None of the custom HDF5 writer functions supported a value of type {type(value)}!"
    )


def load_custom_h5_type(file: str, group: str, entry_type: H5StoreTypes):
    loaders = {}
    loaders.update(CUSTOM_H5_LOADERS)
    loaders.update(DEFAULT_H5_LOADERS)
    for loader_type, loader in loaders.values():
        if loader_type == entry_type:
            return loader(file, group)
    msg = (
        f"No loader found for group {group} with HDF5 store type {entry_type}!"
    )
    loguru.logger.error(msg, exc_info=True)
    raise H5StoreException(msg)


#### Internal functions below here


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

        return name

    try:
        name = str(key)
    except Exception:
        return None

    if "/" in name:
        return None

    return name


#### NONE ####
@_register(DEFAULT_H5_WRITERS, lambda x: x is None)
def dump_None(
    file: str, group: str, value: None, existing_groups: Dict[str, Any]
) -> Dict[str, Any]:
    attributes = {}
    attributes["type"] = H5StoreTypes.Null.name
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Null)
def load_None(file: str, group: str):
    return None


#### FunctionCache ####
@_register(DEFAULT_H5_LOADERS, H5StoreTypes.FunctionCache)
def load_hdf5_function_cache(file: str, group: str) -> Dict[_HashedSeq, Any]:
    function_calls = {}
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        function_group = hdf5_file[group]
        for function_call in function_group:
            arg_group_name = f"{group}/{function_call}/args"
            kwarg_group_name = f"{group}/{function_call}/kwargs"
            value_group_name = f"{group}/{function_call}/value"
            _, args = ExportableClassMixin._load_h5_entry(file, arg_group_name)
            _, kwargs = ExportableClassMixin._load_h5_entry(
                file, kwarg_group_name
            )
            _, value = ExportableClassMixin._load_h5_entry(
                file, value_group_name
            )

            key = _make_key(tuple(args), dict(sorted(kwargs.items())))
            function_calls[key] = value

    return function_calls


#### HDF5 Group ######
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, h5py.Group))
def dump_hdf5_group(
    file: str,
    group: str,
    value: h5py.Group,
    existing_groups: Dict[str, Any],
) -> Dict[str, Any]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        value.copy(source="/", dest=new_entry, name="value")
        attributes["type"] = H5StoreTypes.HDF5Group.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()

    existing_groups[group] = value
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.HDF5Group)
def load_hdf5_group(file: str, group: str):
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        return hdf5_file[group]["value"]


#### python class ####
@_register(
    DEFAULT_H5_WRITERS, lambda x: issubclass(type(x), ExportableClassMixin)
)
def dump_exportable_class(
    file: str,
    group: str,
    value: Type[ExportableClassMixin],
    existing_groups: Dict[str, Any],
) -> Dict[str, Any]:
    attributes = {}
    existing_groups = value._write_hdf5_contents(
        file,
        group=group,
        existing_groups=existing_groups,
    )
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.PythonClass)
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


#### python enum ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, Enum))
def dump_python_enum(
    file: str, group: str, value: Type[Enum], existing_groups: Dict[str, Any]
) -> Dict[str, Any]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        new_entry["enum_class"] = str(value.__class__.__name__)
        new_entry["enum_module"] = str(value.__class__.__module__)
        new_entry["enum_value"] = str(value.name)
        attributes["type"] = H5StoreTypes.Enumerator.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Enumerator)
def load_python_enum(file: str, group: str) -> Type[Enum]:
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        try:
            enum_class_ = entry["enum_class"][()].decode("utf-8")
            enum_value_ = entry["enum_value"][()].decode("utf-8")
            enum_module_ = entry["enum_module"][()].decode("utf-8")
        except KeyError as exc:
            msg = f"Failed to load {group} because it was a enum entry, but didnt have the enum name or value!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc

        enum_module = importlib.import_module(enum_module_)
        enum_class = getattr(enum_module, enum_class_, None)
        if enum_class is None:
            msg = f"Failed to load {group} because it was a enum entry, but the enum module {enum_module_} did not have the enum class {enum_class_}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        try:
            return enum_class[enum_value_]
        except KeyError as exc:
            msg = f"Failed to load {group} because it was a enum entry, but the enum class {enum_class_} did not have the enum entry {enum_value_}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc


#### generic HDF5 dataset ####
@_register(
    DEFAULT_H5_WRITERS,
    lambda x: isinstance(
        x, (PYTHON_BASIC_TYPES, NUMPY_NUMERIC_TYPES, np.ndarray)
    )
    or (
        isinstance(x, Iterable)
        and (not isinstance(x, dict))
        and all(
            isinstance(v, PYTHON_BASIC_TYPES + NUMPY_NUMERIC_TYPES) for v in x
        )
        and all(isinstance(v, type(x[0])) for v in x)
    ),
)
def dump_python_types_or_ndarray(
    file: str, group: str, value: Any, existing_groups: Dict[str, Any]
) -> Dict[str, Any]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        try:
            _ = hdf5_file.create_dataset(name=group, data=value)
        except TypeError as exc:
            msg = f"Failed to store {group} because it was of type {type(value)} which is not natively supported in HDF5!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc
        attributes["type"] = H5StoreTypes.HDF5Dataset.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()
    existing_groups[group] = value

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.HDF5Dataset)
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
    file: str,
    group: str,
    value: ExportableClassMixin,
    existing_groups: Dict[str, Any],
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
        existing_groups = value._dump_h5_entry(
            file,
            f"{group}/{value.__class__.__name__}/{k}",
            v,
            existing_groups=existing_groups,
        )
    return existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.ClassConstructor)
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
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, dict))
def dump_dictionary(
    file: str,
    group: str,
    value: Dict[Any, Any],
    existing_groups: Dict[str, Any],
) -> Dict[str, Any]:
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
                    existing_groups = this_value._write_hdf5_contents(
                        file,
                        f"{group_name}/value",
                        existing_groups=existing_groups,
                    )
                else:
                    existing_groups = ExportableClassMixin._dump_h5_entry(
                        file,
                        f"{group_name}/value",
                        this_value,
                        existing_groups=existing_groups,
                    )

                if issubclass(type(key), ExportableClassMixin):
                    existing_groups = key._write_hdf5_contents(
                        file,
                        f"{group_name}/key",
                        existing_groups=existing_groups,
                    )
                else:
                    existing_groups = ExportableClassMixin._dump_h5_entry(
                        file,
                        f"{group_name}/key",
                        key,
                        existing_groups=existing_groups,
                    )

    attributes["type"] = H5StoreTypes.Dictionary.name
    attributes["timestamp"] = datetime.datetime.now().isoformat()

    existing_groups[group] = value

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Dictionary)
def load_dictionary(file: str, group: str):
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]

        items = list(entry)

        results = {}

        for item in items:
            _, item_result = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{item}/value"
            )

            _, item_key = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{item}/key"
            )

            results[item_key] = item_result

    return results


#### python sequence ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, Iterable))
def dump_generic_sequence(
    file: str,
    group: str,
    value: Sequence[Any],
    existing_groups: Dict[str, Any],
) -> Dict[str, Any]:
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
                    existing_groups = this_value._write_hdf5_contents(
                        file, group_name, existing_groups=existing_groups
                    )
                else:
                    existing_groups = ExportableClassMixin._dump_h5_entry(
                        file,
                        group_name,
                        this_value,
                        existing_groups=existing_groups,
                    )

    attributes["type"] = H5StoreTypes.Sequence.name
    attributes["timestamp"] = datetime.datetime.now().isoformat()
    existing_groups[group] = value

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Sequence)
def load_generic_sequence(file: str, group: str) -> List[Any]:
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        entry = hdf5_file[group]
        basename = group.split("/")[-1]
        items = list(entry)
        malformed_entries = [
            k for k in items if not re.match(rf"{basename}_[0-9]+", k)
        ]
        if len(malformed_entries) > 0:
            msg = f"Failed to load {group} because it was a dictionary, but contained non-item groups {','.join(malformed_entries)}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        results = []

        for item in items:
            _, item_result = ExportableClassMixin._load_h5_entry(
                file, f"{group}/{item}"
            )

            results.append(item_result)
    return results
