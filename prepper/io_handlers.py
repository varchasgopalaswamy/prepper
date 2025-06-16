from __future__ import annotations

from collections.abc import Iterable, Sequence
import contextlib
import datetime
from enum import Enum
import importlib
import numbers
from pathlib import Path
import re
import tempfile
import traceback
from typing import TYPE_CHECKING, Any

import h5py
import loguru
import numpy as np

from prepper.caching import _HashedSeq, _make_key
from prepper.enums import H5StoreTypes
from prepper.exceptions import H5StoreException
from prepper.exportable import ExportableClassMixin
from prepper.utils import check_equality, get_element_from_number_and_weight

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    import periodictable as pt
except ImportError:
    pt = None

try:
    from auto_uncertainties import Uncertainty, nominal_values
except ImportError:
    Uncertainty = None
    nominal_values = None

try:
    import arviz as az
except ImportError:
    az = None


try:
    import pint

    ur = pint.get_application_registry()
except ImportError:
    ur = None

if TYPE_CHECKING:
    from arviz import InferenceData
    from auto_uncertainties import Uncertainty
    from periodictable.core import Element, Isotope
    from xarray import Dataset

__all__ = [
    "dump_class_constructor",
    "dump_custom_h5_type",
    "get_hdf5_compression",
    "load_custom_h5_type",
    "read_h5_attr",
    "register_loader",
    "register_writer",
    "set_hdf5_compression",
    "write_h5_attr",
]

_NONE_TYPE_SENTINEL = "__python_None_sentinel__"
_EMPTY_TYPE_SENTINEL = "__python_Empty_sentinel__"

PYTHON_BASIC_TYPES = (int, float, str)
TYPES_TO_SKIP_DUPLICATE_CHECKING = (*PYTHON_BASIC_TYPES, bool)
if az is not None:
    TYPES_TO_SKIP_DUPLICATE_CHECKING += (az.InferenceData,)

NUMPY_NUMERIC_TYPES = (np.int32, np.int64, np.float32, np.float64)

ALL_VALID_DATASET_TYPES = PYTHON_BASIC_TYPES + NUMPY_NUMERIC_TYPES
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


def set_hdf5_compression(compression: dict[str, Any]):
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
                    msg = f"Could not write attribute {name} with type {type(value)}!"
                    raise H5StoreException(msg) from exc2
                except KeyError:
                    if name in base.attrs:
                        pass
            msg = f"Could not write attribute {name} with type {type(value)}!"
            raise H5StoreException(msg) from exc
        except KeyError:
            if name in base.attrs:
                pass


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
    file: Path, group: str, value: Any, existing_groups: dict[str, Any]
):
    writers = {}
    writers.update(CUSTOM_H5_WRITERS)
    writers.update(DEFAULT_H5_WRITERS)
    # Check to see if this class has already been written to the file
    class_already_written = False
    clone_group = None
    if not isinstance(value, TYPES_TO_SKIP_DUPLICATE_CHECKING):
        for k, v in existing_groups.items():
            if isinstance(v, type(value)):
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

    for name, (validator, writer) in writers.items():
        try:
            is_valid = validator(value)
        except Exception:
            msg = f"Failed to check condition {name} for value {value} of type {type(value)}!"
            loguru.logger.error(msg, exc_info=True)
            is_valid = False
        if is_valid:
            attrs, existing_groups = writer(file, group, value, existing_groups)

            with h5py.File(file, mode="a", track_order=True) as hdf5_file:
                try:
                    entry = hdf5_file[group]
                except KeyError:
                    entry = hdf5_file.require_group(group)
                for k, v in attrs.items():
                    try:
                        write_h5_attr(entry, k, v)  # type: ignore
                    except H5StoreException:
                        msg = f"Failed to write attribute {k} to group {group}!"
                        loguru.logger.error(msg, exc_info=True)

            return existing_groups
    msg = f"None of the custom HDF5 writer functions supported a value of type {type(value)}!"
    raise H5StoreException(msg)


def load_custom_h5_type(file: Path, group: str, entry_type: H5StoreTypes) -> Any:
    loaders = {}
    loaders.update(CUSTOM_H5_LOADERS)
    loaders.update(DEFAULT_H5_LOADERS)
    for loader_type, loader in loaders.values():
        if loader_type == entry_type:
            return loader(file, group)
    msg = f"No loader found for group {group} with HDF5 store type {entry_type}!"
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
    file: Path, group: str, value: None, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes = {}
    attributes["type"] = H5StoreTypes.Null.name
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Null)
def load_None(file: Path, group: str):
    return None


#### FunctionCache ####
@_register(DEFAULT_H5_LOADERS, H5StoreTypes.FunctionCache)
def load_hdf5_function_cache(file: Path, group: str) -> dict[_HashedSeq, Any]:
    function_calls = {}
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        function_group = get_group(hdf5_file, group)
        for function_call in function_group:  # type: ignore
            arg_group_name = f"{group}/{function_call}/args"
            kwarg_group_name = f"{group}/{function_call}/kwargs"
            value_group_name = f"{group}/{function_call}/value"
            _, args = ExportableClassMixin._load_h5_entry(file, arg_group_name)
            _, kwargs = ExportableClassMixin._load_h5_entry(file, kwarg_group_name)
            _, value = ExportableClassMixin._load_h5_entry(file, value_group_name)

            key = _make_key(tuple(args), dict(sorted(kwargs.items())))
            function_calls[key] = value
    return function_calls


#### HDF5 Group ######
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, h5py.Group))
def dump_hdf5_group(
    file: Path,
    group: str,
    value: h5py.Group,
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        value.copy(source="/", dest=new_entry, name="value")
        attributes["type"] = H5StoreTypes.HDF5Group.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()

    existing_groups[group] = value
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.HDF5Group)
def load_hdf5_group(file: Path, group: str):
    with (
        tempfile.TemporaryFile() as tf,
        h5py.File(file, "r", track_order=True) as hdf5_file,
    ):
        f = h5py.File(tf, "w")
        hdf5_file.copy(source=f"{group}/value", dest=f["/"], name="value")

        return f["/value"]


#### python class ####
@_register(DEFAULT_H5_WRITERS, lambda x: issubclass(type(x), ExportableClassMixin))
def dump_exportable_class(
    file: Path,
    group: str,
    value: ExportableClassMixin,
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes = {}
    existing_groups = value._write_hdf5_contents(
        file=file,
        group=group,
        existing_groups=existing_groups,
    )
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.PythonClass)
def load_exportable_class(file: Path, group: str) -> ExportableClassMixin:
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = get_group(hdf5_file, group)
        if "module" not in entry.attrs:
            msg = f"Failed to load {group} because it was a class entry, but didnt have the class module path in the attribute list!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        class_module = importlib.import_module(str(entry.attrs["module"]))
        if "class" not in entry.attrs:
            msg = f"Failed to load {group} because it was a class entry, but didnt have the class name in the attribute list!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        class_definition = getattr(class_module, str(entry.attrs["class"]))
        return class_definition.from_hdf5(file, group)


#### python enum ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, Enum))
def dump_python_enum(
    file: Path, group: str, value: type[Enum], existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
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
def load_python_enum(file: Path, group: str) -> type[Enum]:
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = get_group(hdf5_file, group)
        try:
            enum_class_ = get_dataset(entry, "enum_class")[()].decode("utf-8")
            enum_value_ = get_dataset(entry, "enum_value")[()].decode("utf-8")
            enum_module_ = get_dataset(entry, "enum_module")[()].decode("utf-8")
        except KeyError as exc:
            msg = f"Failed to load {group} because it was a enum entry, but didnt have the enum name or value!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc
        except Exception as exc:
            error = traceback.format_exc()
            msg = f"Failed to load enum from {group} because of error '{error}'!"
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
    lambda x: isinstance(x, (PYTHON_BASIC_TYPES, NUMPY_NUMERIC_TYPES, np.ndarray))  # noqa: UP038
    or (
        isinstance(x, Iterable)
        and (not isinstance(x, dict))
        and all(
            isinstance(v, ALL_VALID_DATASET_TYPES)  # type: ignore
            for v in x  # type: ignore
        )
        and all(isinstance(v, type(x[0])) for v in x)  # type: ignore
    ),
)
def dump_python_types_or_ndarray(
    file: Path, group: str, value: Any, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        try:
            hdf5_file[group] = value
        except TypeError as exc:
            msg = f"Failed to store {group} because it was of type {type(value)} which is not natively supported in HDF5!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc
        attributes["type"] = H5StoreTypes.HDF5Dataset.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()
    existing_groups[group] = value

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.HDF5Dataset)
def load_python_types_or_ndarray(file: Path, group: str):
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = get_dataset(hdf5_file, group)
        try:
            return entry.asstr("utf-8")[()]
        except TypeError:
            val = entry[()]
            if isinstance(val, np.bool_):
                val = bool(val)
            return val


#### class constructor ####
def dump_class_constructor(
    file: Path,
    group: str,
    value: ExportableClassMixin,
    existing_groups: dict[str, Any],
) -> dict[str, Any]:
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        my_group = hdf5_file.require_group(group)
        constructor_group = my_group.require_group(value.__class__.__name__)
        constructor_attributes = {}
        constructor_attributes["module"] = value.__class__.__module__
        constructor_attributes["class"] = value.__class__.__name__
        constructor_attributes["timestamp"] = datetime.datetime.now().isoformat()
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
def load_class_constructor(file: Path, group: str):
    kwargs = {}
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        my_group = get_group(hdf5_file, group)
        for key in my_group:
            kwargs[key] = ExportableClassMixin._load_h5_entry(file, f"{group}/{key}")[1]

    return kwargs


#### python dict ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, dict))
def dump_dictionary(
    file: Path,
    group: str,
    value: dict[Any, Any],
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
                    if group_name in get_group(hdf5_file, group):
                        group_name = f"{group}/{group}_{pad_number}"
                else:
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
def load_dictionary(file: Path, group: str):
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        entry = get_group(hdf5_file, group)

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
    file: Path,
    group: str,
    value: Sequence[Any],
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
def load_generic_sequence(file: Path, group: str) -> list[Any]:
    with h5py.File(file, mode="r", track_order=True) as hdf5_file:
        entry = get_group(hdf5_file, group)
        basename = group.split("/")[-1]
        items = list(entry)
        malformed_entries = [k for k in items if not re.match(rf"{basename}_[0-9]+", k)]
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


if xr is not None:
    #### xarray ####
    @register_writer(lambda x: isinstance(x, xr.Dataset))
    def dump_xarray(
        file: Path,
        group: str,
        value: Dataset,
        existing_groups: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attributes = {}

        compression_args = {}
        for k in value.data_vars:
            compression_args[k] = get_hdf5_compression()

        value.astype(np.float32).to_netcdf(
            file,
            group=group,
            format="NETCDF4",
            mode="a",
            engine="h5netcdf",
            invalid_netcdf=True,
            encoding=compression_args,
        )
        value.close()
        attributes["type"] = H5StoreTypes.XArrayDataset.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()
        existing_groups[group] = value

        return attributes, existing_groups

    @register_loader(H5StoreTypes.XArrayDataset)
    def load_xarray(file: Path, group: str) -> Dataset:
        return xr.load_dataset(file, group=group, format="NETCDF4", engine="h5netcdf")


if pt is not None:

    @register_writer(
        lambda x: isinstance(x, pt.core.Isotope | pt.core.Element),
    )
    def dump_periodictable_element(
        file: Path,
        group: str,
        value: Isotope | Element,
        existing_groups: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attributes = {}
        with h5py.File(file, mode="a", track_order=True) as hdf5_file:
            new_entry = hdf5_file.require_group(group)
            new_entry["element_name"] = str(value.name)
            new_entry["element_A"] = value.mass
            new_entry["element_Z"] = value.number
            attributes["type"] = H5StoreTypes.PeriodicTableElement.name
            attributes["timestamp"] = datetime.datetime.now().isoformat()

        return attributes, existing_groups

    @register_loader(H5StoreTypes.PeriodicTableElement)
    def load_periodictable_element(file: Path, group: str) -> Isotope | Element:
        with h5py.File(file, "r", track_order=True) as hdf5_file:
            entry = get_group(hdf5_file, group)
            atomic_weight = float(get_dataset(entry, "element_A")[()])
            atomic_number = float(get_dataset(entry, "element_Z")[()])
            return get_element_from_number_and_weight(z=atomic_number, a=atomic_weight)


if az is not None:
    #### arviz ####
    @register_writer(lambda x: isinstance(x, az.InferenceData))
    def dump_inferencedata(
        file: Path,
        group: str,
        value: InferenceData,
        existing_groups: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attributes = {}

        value.to_netcdf(
            str(file),
            engine="h5netcdf",
            base_group=group,
            overwrite_existing=False,
        )

        attributes["type"] = H5StoreTypes.ArViz.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()
        existing_groups[group] = value

        return attributes, existing_groups

    @register_loader(H5StoreTypes.ArViz)
    def load_inferencedata(file: Path, group: str) -> InferenceData:
        return az.InferenceData.from_netcdf(file, base_group=group, engine="h5netcdf")


#### ndarray with units/error ####
@register_writer(
    lambda x: isinstance(x, Uncertainty) or hasattr(x, "units"),
)
def dump_unit_or_error_ndarrays(
    file: Path, group: str, value: Any, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(value, "units"):
        u = str(value.units)
        v = value.magnitude
    else:
        u = None
        v = value

    if hasattr(v, "error"):
        e = v.error
        v = v.value
    else:
        e = np.zeros_like(v)

    attributes = {}
    with h5py.File(file, mode="a", track_order=True) as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        if np.size(v) > 1:
            compression = get_hdf5_compression()
            if not issubclass(v.dtype.type, numbers.Integral) and issubclass(
                v.dtype.type, numbers.Real
            ):
                v = v.astype(np.float32)
                e = e.astype(np.float32)
        else:
            compression = {}
        new_entry.create_dataset(name="value", data=v, **compression)
        if np.any(e > 0):
            new_entry.create_dataset(name="error", data=e, **compression)

        if u is not None:
            attributes["unit"] = u
        attributes["type"] = H5StoreTypes.DimensionalNDArray.name
        attributes["timestamp"] = datetime.datetime.now().isoformat()
        attributes["ndim"] = np.ndim(v)

    existing_groups[group] = value
    return attributes, existing_groups


@register_loader(H5StoreTypes.DimensionalNDArray)
def load_unit_or_error_ndarrays(file: Path, group: str):
    with h5py.File(file, "r", track_order=True) as hdf5_file:
        entry = get_group(hdf5_file, group)
        v = get_dataset(entry, "value")[()]

        if "error" in entry:
            if Uncertainty is not None:
                e = get_dataset(entry, "error")[()]
                if np.any(e > 0):
                    v = Uncertainty(v, e)
            else:
                msg = f"Need auto_uncertainties to load uncertainty arrays from group {entry}!"
                raise ImportError(msg)
        try:
            unit = read_h5_attr(entry, "unit")
            if unit is not None:
                if ur is None:
                    msg = "The dataset had unit information, but pint is not installed!"
                    raise ValueError(msg)
                else:
                    v *= ur(unit)
        except KeyError:
            pass

        ndim = read_h5_attr(entry, "ndim")

        if ndim == 0:
            if nominal_values is not None:
                sz = np.size(nominal_values(v))
            else:
                sz = np.size(v)
            if sz > 1:
                raise H5StoreException
            with contextlib.suppress(IndexError, TypeError):
                v = v[0]
    return v


def get_group(hdf5_handle: h5py.File | h5py.Group, name: str) -> h5py.Group:
    if name not in hdf5_handle:
        msg = f"Could not find group {name}!"
        raise KeyError(msg)
    else:
        ret = hdf5_handle[name]
        if isinstance(ret, h5py.Group):
            return ret
        else:
            msg = f"Expected group {name} but found {type(ret)}!"
            raise TypeError(msg)


def get_dataset(hdf5_handle: h5py.File | h5py.Group, name: str) -> h5py.Dataset:
    if name not in hdf5_handle:
        msg = f"Could not find group {name}!"
        raise KeyError(msg)
    else:
        ret = hdf5_handle[name]
        if isinstance(ret, h5py.Dataset):
            return ret
        else:
            msg = f"Expected group {name} but found {type(ret)}!"
            raise TypeError(msg)
