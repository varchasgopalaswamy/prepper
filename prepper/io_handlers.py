from bisect import bisect_left, bisect_right, insort
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import MISSING, fields
import datetime as dt
from enum import Enum, Flag
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar
import uuid

import h5py
import loguru
import numpy as np

from prepper._storage import FileSource, open_hdf5
from prepper.caching import _HashedSeq, _make_key
from prepper.config import get_settings
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
    import pint

    ur = pint.get_application_registry()
except ImportError:
    ur = None

try:
    import equinox as eqx
    import jax
except ImportError:
    eqx = None
    jax = None

if TYPE_CHECKING:
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


NUMPY_NUMERIC_TYPES = (np.int32, np.int64, np.float32, np.float64)

ALL_VALID_DATASET_TYPES = PYTHON_BASIC_TYPES + NUMPY_NUMERIC_TYPES
F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_H5_WRITERS: dict[str, tuple[Any, Callable[..., Any]]] = {}
DEFAULT_H5_LOADERS: dict[str, tuple[Any, Callable[..., Any]]] = {}
CUSTOM_H5_WRITERS: dict[str, tuple[Any, Callable[..., Any]]] = {}
CUSTOM_H5_LOADERS: dict[str, tuple[Any, Callable[..., Any]]] = {}
HDF5_COMPRESSION = {
    "compression": "gzip",
    "compression_opts": 9,
    "shuffle": True,
    "fletcher32": True,
}
_INTERNAL_HANDLERS: set[Callable[..., Any]] = set()
_ACTIVE_LOADS: ContextVar[frozenset[tuple[str, int]]] = ContextVar(
    "prepper_active_loads", default=frozenset()
)
_EXACT_DEDUP: ContextVar[bool] = ContextVar("prepper_exact_dedup", default=False)


@contextmanager
def exact_dedup() -> Generator[None]:
    """Restrict deduplication to identical objects when writing exact cache keys."""
    token = _EXACT_DEDUP.set(True)
    try:
        yield
    finally:
        _EXACT_DEDUP.reset(token)


def _comparison_bucket(value: Any) -> tuple[Any, ...]:
    if isinstance(value, np.ndarray):
        return type(value), value.shape, value.dtype.str
    if isinstance(value, (list, tuple, dict)):
        return type(value), len(value)
    return (type(value),)


def _array_magnitude(value: Any) -> float | None:
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biufc":
        return None
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=np.complex128 if value.dtype.kind == "c" else np.float64)
        if not np.all(np.isfinite(array)):
            return None
        magnitude = float(np.max(np.abs(array), initial=0))
    return magnitude if np.isfinite(magnitude) else None


class _WriteIndex(dict[str, Any]):
    """Keep identity lookups separate from approximate content comparisons."""

    def __init__(self, values: dict[str, Any]) -> None:
        super().__init__()
        self.identities: dict[int, str] = {}
        self.buckets: dict[tuple[Any, ...], list[str]] = {}
        self.magnitudes: dict[tuple[Any, ...], list[tuple[float, str]]] = {}
        self.other_values: dict[tuple[Any, ...], list[str]] = {}
        self.positions: dict[str, int] = {}
        self.active: set[int] = set()
        for name, value in values.items():
            self[name] = value

    def __setitem__(self, name: str, value: Any) -> None:
        super().__setitem__(name, value)
        self.positions.setdefault(name, len(self.positions))
        self.identities.setdefault(id(value), name)
        bucket = _comparison_bucket(value)
        self.buckets.setdefault(bucket, []).append(name)
        magnitude = _array_magnitude(value)
        if magnitude is None:
            self.other_values.setdefault(bucket, []).append(name)
        else:
            insort(self.magnitudes.setdefault(bucket, []), (magnitude, name))

    def candidates(self, value: Any) -> list[str]:
        bucket = _comparison_bucket(value)
        magnitude = _array_magnitude(value)
        rtol = get_settings().compare_rtol
        if magnitude is None or rtol >= 1:
            return self.buckets.get(bucket, [])
        if value.dtype.kind not in "fc":
            rtol = 0
        # Relative equality requires min(maxabs(a), maxabs(b)) >= (1-r)*max(...).
        # Widen the interval slightly to make the index conservative at float boundaries.
        padding = magnitude * (32 * np.finfo(np.float64).eps)
        lower = max(0.0, magnitude * (1 - rtol) - padding)
        upper = magnitude / (1 - rtol) + padding
        items = self.magnitudes.get(bucket, [])
        start = bisect_left(items, (lower, ""))
        stop = bisect_right(items, (upper, chr(0x10FFFF)))
        selected = {name for _, name in items[start:stop]}
        selected.update(self.other_values.get(bucket, ()))
        # Keep insertion order, so the first equal value remains the canonical copy.
        return sorted(selected, key=self.positions.__getitem__)


def _handler_source(file: FileSource, handler: Callable[..., Any]) -> FileSource:
    if handler in _INTERNAL_HANDLERS or not isinstance(file, h5py.File):
        return file
    # Existing plugins open the filename themselves. Flush before handing it over.
    file.flush()
    return Path(file.filename)


def _metadata(entry_type: H5StoreTypes) -> dict[str, Any]:
    return {"type": entry_type.name, "timestamp": dt.datetime.now().astimezone().isoformat()}


def get_hdf5_compression() -> dict[str, Any]:
    return HDF5_COMPRESSION


def set_hdf5_compression(compression: dict[str, Any]) -> None:
    global HDF5_COMPRESSION  # noqa: PLW0603 - the setter replaces the shared configuration.
    HDF5_COMPRESSION = compression


def write_h5_attr(base: h5py.Group | h5py.Dataset, name: str, value: Any) -> None:
    marker = f"__prepper_none__:{name}"
    try:
        if value is None:
            base.attrs[name] = _NONE_TYPE_SENTINEL
            base.attrs[marker] = True
        else:
            base.attrs[name] = value
            if isinstance(value, str) and value == _NONE_TYPE_SENTINEL:
                base.attrs[marker] = False
            elif marker in base.attrs:
                del base.attrs[marker]
    except (TypeError, ValueError) as exc:
        msg = f"Could not write attribute {name} with type {type(value)}!"
        raise H5StoreException(msg) from exc


def read_h5_attr(base: h5py.Group | h5py.Dataset, name: str) -> Any:
    try:
        value = base.attrs[name]
    except KeyError as exc:
        msg = f"Could not find attribute {name} in group {base.name}!"
        raise KeyError(msg) from exc

    if (
        isinstance(value, str)
        and value == _NONE_TYPE_SENTINEL
        and base.attrs.get(f"__prepper_none__:{name}", True)
    ):
        value = None
    return value


def register_loader(validator: H5StoreTypes) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        CUSTOM_H5_LOADERS[func.__name__] = (validator, func)
        return func

    return decorator


def register_writer(validator: Callable[[Any], bool]) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        CUSTOM_H5_WRITERS[func.__name__] = (validator, func)
        return func

    return decorator


def dump_custom_h5_type(
    file: FileSource, group: str, value: Any, existing_groups: dict[str, Any]
) -> dict[str, Any]:
    index = (
        existing_groups
        if isinstance(existing_groups, _WriteIndex)
        else _WriteIndex(existing_groups)
    )
    canonical_group = "/" + "/".join(part for part in group.split("/") if part)
    clone_group = index.identities.get(id(value))
    ancestor = clone_group is not None and (
        canonical_group == clone_group.rstrip("/")
        or canonical_group.startswith(clone_group.rstrip("/") + "/")
    )
    if id(value) in index.active or ancestor:
        msg = f"Cyclic object references are not supported at {group}."
        raise H5StoreException(msg)
    if not isinstance(value, TYPES_TO_SKIP_DUPLICATE_CHECKING):
        if clone_group is None and not _EXACT_DEDUP.get():
            for candidate in index.candidates(value):
                try:
                    same = check_equality(value, index[candidate])
                except Exception:  # noqa: BLE001 -- User equality methods may raise arbitrary errors.
                    same = False
                if same:
                    clone_group = candidate
                    break
        if clone_group is not None:
            with open_hdf5(file, "a") as hdf5_file:
                hdf5_file[group] = h5py.SoftLink(clone_group)
            return index

    index.active.add(id(value))
    try:
        with open_hdf5(file, "a") as hdf5_file:
            for validator, writer in (*CUSTOM_H5_WRITERS.values(), *DEFAULT_H5_WRITERS.values()):
                try:
                    is_valid = validator(value)
                except Exception:  # noqa: BLE001 -- Plugin predicates may reject arbitrary objects.
                    is_valid = False
                if is_valid:
                    attrs, updated = writer(
                        _handler_source(hdf5_file, writer), group, value, index
                    )
                    if updated is not index:
                        for name, item in updated.items():
                            if name not in index:
                                index[name] = item
                    entry = hdf5_file[group]
                    for name, item in attrs.items():
                        write_h5_attr(entry, name, item)
                    # Every completed object is indexed, including plugin and enum values.
                    if group not in index:
                        index[group] = value
                    return index
    finally:
        index.active.remove(id(value))
    msg = f"None of the custom HDF5 writer functions supported a value of type {type(value)}!"
    raise H5StoreException(msg)


def load_custom_h5_type(file: FileSource, group: str, entry_type: H5StoreTypes) -> Any:
    with open_hdf5(file) as hdf5_file:
        entry = hdf5_file[group]
        key = str(hdf5_file.filename), hash(entry.id)
        active = _ACTIVE_LOADS.get()
        if key in active:
            msg = f"Cyclic HDF5 references are not supported at {group}."
            raise H5StoreException(msg)
        token = _ACTIVE_LOADS.set(active | {key})
        try:
            for loader_type, loader in (*CUSTOM_H5_LOADERS.values(), *DEFAULT_H5_LOADERS.values()):
                if loader_type == entry_type:
                    return loader(_handler_source(hdf5_file, loader), group)
        finally:
            _ACTIVE_LOADS.reset(token)
    msg = f"No loader found for group {group} with HDF5 store type {entry_type}!"
    loguru.logger.error(msg, exc_info=True)
    raise H5StoreException(msg)


#### Internal functions below here


def _register(
    store: dict[str, tuple[Any, Callable[..., Any]]], validator: Any
) -> Callable[[F], F]:
    def decorator(func: F) -> F:
        store[func.__name__] = (validator, func)
        _INTERNAL_HANDLERS.add(func)
        return func

    return decorator


def key_to_group_name(key: Any) -> str | None:
    if isinstance(key, Enum):
        name = key.name

        if "/" in name:
            return None

        return name

    try:
        name = str(key)
    except Exception:  # noqa: BLE001 - user objects and callbacks may raise arbitrary errors.
        return None

    if "/" in name:
        return None

    return name


#### NONE ####
@_register(DEFAULT_H5_WRITERS, lambda x: x is None)
def dump_None(
    file: FileSource, group: str, value: None, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    with open_hdf5(file, "a") as handle:
        handle.require_group(group)
    return {"type": H5StoreTypes.Null.name}, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Null)
def load_None(file: FileSource, group: str) -> None:
    return None


#### FunctionCache ####
@_register(DEFAULT_H5_LOADERS, H5StoreTypes.FunctionCache)
def load_hdf5_function_cache(file: FileSource, group: str) -> dict[_HashedSeq, Any]:
    function_calls = {}
    with open_hdf5(file) as hdf5_file:
        function_group = get_group(hdf5_file, group)
        for function_call in function_group:
            arg_group_name = f"{group}/{function_call}/args"
            kwarg_group_name = f"{group}/{function_call}/kwargs"
            value_group_name = f"{group}/{function_call}/value"
            _, args = ExportableClassMixin._load_h5_entry(hdf5_file, arg_group_name)
            _, kwargs = ExportableClassMixin._load_h5_entry(hdf5_file, kwarg_group_name)
            _, value = ExportableClassMixin._load_h5_entry(hdf5_file, value_group_name)

            key = _make_key(tuple(args), kwargs)
            function_calls[key] = value
    return function_calls


#### HDF5 Group ######
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, h5py.Group))
def dump_hdf5_group(
    file: FileSource,
    group: str,
    value: h5py.Group,
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes: dict[str, Any] = {}
    with open_hdf5(file, "a") as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        value.copy(source=value, dest=new_entry, name="value")
        attributes["type"] = H5StoreTypes.HDF5Group.name
        attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()

    existing_groups[group] = value
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.HDF5Group)
def load_hdf5_group(file: FileSource, group: str) -> h5py.Group | h5py.Dataset:
    with open_hdf5(file) as hdf5_file:
        # The HDF5 core driver owns its memory; no Python backing file can close early.
        memory_file = h5py.File(str(uuid.uuid4()), "w", driver="core", backing_store=False)
        try:
            hdf5_file.copy(source=f"{group}/value", dest=memory_file["/"], name="value")
            return memory_file["/value"]
        except BaseException:
            memory_file.close()
            raise


#### python class ####
@_register(DEFAULT_H5_WRITERS, lambda x: issubclass(type(x), ExportableClassMixin))
def dump_exportable_class(
    file: FileSource,
    group: str,
    value: ExportableClassMixin,
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes: dict[str, Any] = {}
    existing_groups = value._write_hdf5_contents(
        file=file,
        group=group,
        existing_groups=existing_groups,
    )
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.PythonClass)
def load_exportable_class(file: FileSource, group: str) -> ExportableClassMixin:
    with open_hdf5(file) as hdf5_file:
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
        class_definition: Any = class_module
        for component in str(entry.attrs.get("qualname", entry.attrs["class"])).split("."):
            class_definition = getattr(class_definition, component, None)
        if not isinstance(class_definition, type) or not issubclass(
            class_definition, ExportableClassMixin
        ):
            msg = f"The class stored at {group} is not an importable ExportableClassMixin."
            raise H5StoreException(msg)
        return class_definition.from_hdf5(hdf5_file, group)


#### python enum ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, Enum))
def dump_python_enum(
    file: FileSource, group: str, value: Enum, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    attributes: dict[str, Any] = {}
    with open_hdf5(file, "a") as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        new_entry["enum_class"] = str(value.__class__.__qualname__)
        new_entry["enum_module"] = str(value.__class__.__module__)
        new_entry["enum_value"] = str(value.name)
        if isinstance(value, Flag):
            new_entry["enum_underlying_value"] = value.value
        attributes["type"] = H5StoreTypes.Enumerator.name
        attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Enumerator)
def load_python_enum(file: FileSource, group: str) -> Enum:
    with open_hdf5(file) as hdf5_file:
        entry = get_group(hdf5_file, group)
        try:
            enum_class_ = get_dataset(entry, "enum_class")[()].decode("utf-8")
            enum_value_ = get_dataset(entry, "enum_value")[()].decode("utf-8")
            enum_module_ = get_dataset(entry, "enum_module")[()].decode("utf-8")
        except KeyError as exc:
            msg = f"Failed to load {group} because it was a enum entry, but didnt have the enum name or value!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc
        enum_module = importlib.import_module(enum_module_)
        enum_class: Any = enum_module
        for component in enum_class_.split("."):
            enum_class = getattr(enum_class, component, None)
            if enum_class is None:
                break
        if enum_class is None:
            msg = f"Failed to load {group} because it was a enum entry, but the enum module {enum_module_} did not have the enum class {enum_class_}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg)
        try:
            if "enum_underlying_value" in entry:
                value = get_dataset(entry, "enum_underlying_value")[()]
                if isinstance(value, np.generic):
                    value = value.item()
                return enum_class(value)
            return enum_class[enum_value_]
        except (KeyError, ValueError) as exc:
            msg = f"Failed to load {group} because it was a enum entry, but the enum class {enum_class_} did not have the enum entry {enum_value_}!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc


#### generic HDF5 dataset ####
def _is_native_dataset(value: Any) -> bool:
    if jax is not None and isinstance(value, jax.Array):
        return True
    if isinstance(value, (*PYTHON_BASIC_TYPES, bytes, complex, np.generic, np.ndarray)):
        return True
    if isinstance(value, (list, tuple)):
        return not value or (
            type(value[0]) in {int, float, complex, bool, str}
            and all(type(item) is type(value[0]) for item in value)
        )
    return False


@_register(DEFAULT_H5_WRITERS, _is_native_dataset)
def dump_python_types_or_ndarray(
    file: FileSource, group: str, value: Any, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    original_value = value
    if jax is not None and isinstance(value, jax.Array):
        value = np.asarray(value)
    attributes: dict[str, Any] = {}
    with open_hdf5(file, "a") as hdf5_file:
        if isinstance(value, bytes) and not isinstance(value, np.generic):
            kind = "bytes"
            data = np.frombuffer(value, dtype=np.uint8)
        elif isinstance(value, np.ndarray):
            kind = "ndarray"
            data = value
        elif isinstance(value, np.generic):
            kind = "numpy_scalar"
            data = value
        else:
            kind = type(value).__name__
            data = value
            if isinstance(value, (list, tuple)):
                data = np.asarray(value)
                if data.dtype.kind == "U":
                    data = data.astype(h5py.string_dtype("utf-8"))
        attributes["python_kind"] = kind
        if isinstance(value, (np.ndarray, np.generic)):
            attributes["numpy_dtype"] = value.dtype.str
            if value.dtype.kind == "U":
                data = np.asarray(value, dtype=h5py.string_dtype("utf-8"))
            elif value.dtype.kind in "mM":
                data = value.view(np.int64)
        try:
            hdf5_file[group] = data
        except (TypeError, ValueError) as exc:
            msg = f"Failed to store {group} because it was of type {type(value)} which is not natively supported in HDF5!"
            loguru.logger.error(msg)
            raise H5StoreException(msg) from exc
        attributes["type"] = H5StoreTypes.HDF5Dataset.name
        attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()
    existing_groups[group] = original_value

    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.HDF5Dataset)
def load_python_types_or_ndarray(file: FileSource, group: str) -> Any:
    with open_hdf5(file) as hdf5_file:
        entry = get_dataset(hdf5_file, group)
        kind = entry.attrs.get("python_kind")
        if kind is not None and kind not in {
            "bytes",
            "ndarray",
            "numpy_scalar",
            "list",
            "tuple",
            "bool",
            "int",
            "float",
            "complex",
            "str",
        }:
            msg = f"Unknown native value kind {kind!r} at {group}."
            raise H5StoreException(msg)
        if kind == "bytes":
            return entry[()].tobytes()
        dtype = np.dtype(entry.attrs["numpy_dtype"]) if "numpy_dtype" in entry.attrs else None
        try:
            val = (
                entry[()] if dtype is not None and dtype.kind == "S" else entry.asstr("utf-8")[()]
            )
        except TypeError:
            val = entry[()]
        if kind in {"ndarray", "numpy_scalar"}:
            assert dtype is not None
            if dtype.kind in "mM":
                array = np.asarray(val).view(dtype)
            else:
                array = np.asarray(val, dtype=entry.dtype if entry.dtype.names else dtype)
            return array if kind == "ndarray" else array[()]
        if kind in {"list", "tuple"}:
            values = np.asarray(val).tolist()
            return tuple(values) if kind == "tuple" else values
        if kind == "bool" or (kind is None and isinstance(val, np.bool_)):
            return bool(val)
        if kind == "int":
            return int(val)
        if kind == "float":
            return float(val)
        if kind == "complex":
            return complex(val)
        return val


#### class constructor ####
def dump_class_constructor(
    file: FileSource,
    group: str,
    value: ExportableClassMixin,
    existing_groups: dict[str, Any],
) -> dict[str, Any]:
    with open_hdf5(file, "a") as handle:
        constructor = handle.require_group(group).require_group(value.__class__.__name__)
        attributes = _metadata(H5StoreTypes.ClassConstructor)
        attributes.update(module=value.__class__.__module__, **{"class": value.__class__.__name__})
        call = getattr(value, "_constructor_call", None)
        if call is not None:
            attributes["constructor_kind"] = "call"
        for name, item in attributes.items():
            write_h5_attr(constructor, name, item)
        entries = (
            {"__args__": call[0], "__kwargs__": call[1]}
            if call is not None
            else value._constructor_args
        )
        for name, item in entries.items():
            existing_groups = value._dump_h5_entry(
                handle,
                f"{constructor.name}/{name}",
                item,
                existing_groups=existing_groups,
            )
    return existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.ClassConstructor)
def load_class_constructor(
    file: FileSource, group: str
) -> dict[str, Any] | tuple[tuple[Any, ...], dict[str, Any]]:
    with open_hdf5(file) as handle:
        entry = get_group(handle, group)
        values = {
            name: ExportableClassMixin._load_h5_entry(handle, f"{group}/{name}")[1]
            for name in entry
        }
        kind = entry.attrs.get("constructor_kind")
        if kind == "call":
            if (
                set(values) != {"__args__", "__kwargs__"}
                or not isinstance(values["__args__"], tuple)
                or not isinstance(values["__kwargs__"], dict)
            ):
                msg = f"Malformed constructor arguments at {group}."
                raise H5StoreException(msg)
            return values["__args__"], values["__kwargs__"]
        if kind is not None:
            msg = f"Unknown constructor argument kind {kind!r} at {group}."
            raise H5StoreException(msg)
        return values


#### python dict ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, dict))
def dump_dictionary(
    file: FileSource,
    group: str,
    value: dict[Any, Any],
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with open_hdf5(file, "a") as handle:
        handle.require_group(group)
        width = len(str(max(len(value), 1)))
        for index, (key, item) in enumerate(value.items(), 1):
            item_group = f"{group}/item_{index:0{width}d}"
            for name, child in (("key", key), ("value", item)):
                with exact_dedup() if name == "key" else nullcontext():
                    existing_groups = ExportableClassMixin._dump_h5_entry(
                        handle, f"{item_group}/{name}", child, existing_groups=existing_groups
                    )
    existing_groups[group] = value
    return _metadata(H5StoreTypes.Dictionary), existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Dictionary)
def load_dictionary(file: FileSource, group: str) -> dict[Any, Any]:
    with open_hdf5(file) as handle:
        entry = get_group(handle, group)
        result = {}
        for name in entry:
            item = get_group(entry, name)
            if set(item) != {"key", "value"}:
                msg = f"Malformed dictionary item at {item.name}."
                raise H5StoreException(msg)
            key = ExportableClassMixin._load_h5_entry(handle, f"{group}/{name}/key")[1]
            value = ExportableClassMixin._load_h5_entry(handle, f"{group}/{name}/value")[1]
            try:
                if key in result:
                    msg = f"Duplicate dictionary key at {item.name}."
                    raise H5StoreException(msg)
                result[key] = value
            except TypeError as exc:
                msg = f"Unhashable dictionary key at {item.name}."
                raise H5StoreException(msg) from exc
    return result


#### python sequence ####
@_register(DEFAULT_H5_WRITERS, lambda x: isinstance(x, (list, tuple, set, frozenset)))
def dump_generic_sequence(
    file: FileSource,
    group: str,
    value: Sequence[Any] | set[Any] | frozenset[Any],
    existing_groups: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    with open_hdf5(file, "a") as handle:
        handle.require_group(group)
        width = len(str(max(len(value), 1)))
        for index, item in enumerate(value, 1):
            existing_groups = ExportableClassMixin._dump_h5_entry(
                handle, f"{group}/item_{index:0{width}d}", item, existing_groups=existing_groups
            )
    attributes = _metadata(H5StoreTypes.Sequence)
    attributes["python_kind"] = type(value).__name__
    attributes["sequence_length"] = len(value)
    existing_groups[group] = value
    return attributes, existing_groups


@_register(DEFAULT_H5_LOADERS, H5StoreTypes.Sequence)
def load_generic_sequence(
    file: FileSource, group: str
) -> list[Any] | tuple[Any, ...] | set[Any] | frozenset[Any]:
    with open_hdf5(file) as handle:
        entry = get_group(handle, group)
        names = list(entry)
        kind = entry.attrs.get("python_kind")
        if kind is not None and kind not in {"list", "tuple", "set", "frozenset"}:
            msg = f"Unknown sequence kind {kind!r} at {group}."
            raise H5StoreException(msg)
        if "sequence_length" in entry.attrs:
            length = int(entry.attrs["sequence_length"])
            width = len(str(max(length, 1)))
            expected = [f"item_{index:0{width}d}" for index in range(1, length + 1)]
            if length < 0 or set(names) != set(expected):
                msg = f"Malformed sequence entries at {group}."
                raise H5StoreException(msg)
            names = expected
        elif any(not name.rpartition("_")[2].isdigit() for name in names):
            msg = f"Malformed legacy sequence entries at {group}."
            raise H5StoreException(msg)
        # Legacy files used the path basename as a prefix. Soft links can change
        # the apparent basename, so use their stored order without reinterpreting it.
        result = [
            ExportableClassMixin._load_h5_entry(handle, f"{group}/{name}")[1] for name in names
        ]
        kind = entry.attrs.get("python_kind")
        if kind == "tuple":
            return tuple(result)
        if kind == "set":
            return set(result)
        if kind == "frozenset":
            return frozenset(result)
        return result


if xr is not None:
    #### xarray ####
    @_register(CUSTOM_H5_WRITERS, lambda x: isinstance(x, xr.Dataset))
    def dump_xarray(
        file: FileSource,
        group: str,
        value: Dataset,
        existing_groups: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attributes: dict[str, Any] = {}

        compression_args = {}
        for k in value.data_vars:
            if value[k].ndim > 0 and value[k].size > 0 and value[k].dtype.kind in "biufc":
                compression_args[k] = get_hdf5_compression()

        with open_hdf5(file, "a") as handle:
            handle.require_group(group)
            handle.flush()
            value.to_netcdf(
                handle.filename,
                group=f"{group}/value",
                format="NETCDF4",
                mode="a",
                engine="h5netcdf",
                invalid_netcdf=True,
                encoding=compression_args,
            )
        attributes["type"] = H5StoreTypes.XArrayDataset.name
        attributes["xarray_group"] = "value"
        attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()
        existing_groups[group] = value

        return attributes, existing_groups

    @_register(CUSTOM_H5_LOADERS, H5StoreTypes.XArrayDataset)
    def load_xarray(file: FileSource, group: str) -> Dataset:
        with open_hdf5(file) as handle:
            entry = get_group(handle, group)
            child = entry.attrs.get("xarray_group")
            result = xr.load_dataset(
                handle.filename,
                group=f"{group}/{child}" if child else group,
                engine="h5netcdf",
            )
            if not child:
                # Old files placed serialization metadata on the Dataset itself.
                for name in ("type", "timestamp", "bound_class"):
                    result.attrs.pop(name, None)
            return result


if pt is not None:

    @_register(
        CUSTOM_H5_WRITERS,
        lambda x: isinstance(x, pt.core.Isotope | pt.core.Element),  # pyright: ignore[reportOptionalMemberAccess]
    )
    def dump_periodictable_element(
        file: FileSource,
        group: str,
        value: Isotope | Element,
        existing_groups: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attributes: dict[str, Any] = {}
        with open_hdf5(file, "a") as hdf5_file:
            new_entry = hdf5_file.require_group(group)
            new_entry["element_name"] = str(value.name)
            new_entry["element_A"] = value.mass
            new_entry["element_Z"] = value.number
            if isinstance(value, pt.core.Isotope):
                attributes["element_kind"] = "isotope"
                attributes["isotope_number"] = value.isotope
            else:
                attributes["element_kind"] = "element"
            attributes["type"] = H5StoreTypes.PeriodicTableElement.name
            attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()

        return attributes, existing_groups

    @_register(CUSTOM_H5_LOADERS, H5StoreTypes.PeriodicTableElement)
    def load_periodictable_element(file: FileSource, group: str) -> Isotope | Element:
        with open_hdf5(file) as hdf5_file:
            entry = get_group(hdf5_file, group)
            kind = entry.attrs.get("element_kind")
            if kind is not None:
                try:
                    element = pt.elements[int(get_dataset(entry, "element_Z")[()])]
                    if kind == "element":
                        return element
                    if kind == "isotope":
                        return element[int(entry.attrs["isotope_number"])]
                except (KeyError, TypeError, ValueError) as exc:
                    msg = f"Invalid periodic table entry at {group}."
                    raise H5StoreException(msg) from exc
                msg = f"Unknown periodic table value kind {kind!r} at {group}."
                raise H5StoreException(msg)
            atomic_weight = float(get_dataset(entry, "element_A")[()])
            atomic_number = float(get_dataset(entry, "element_Z")[()])
            return get_element_from_number_and_weight(z=atomic_number, a=atomic_weight)


#### ndarray with units/error ####
@_register(
    CUSTOM_H5_WRITERS,
    lambda x: (Uncertainty is not None and isinstance(x, Uncertainty)) or hasattr(x, "units"),
)
def dump_unit_or_error_ndarrays(
    file: FileSource, group: str, value: Any, existing_groups: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if hasattr(value, "units"):
        u = str(value.units)
        v = value.magnitude
    else:
        u = None
        v = value

    has_uncertainty = hasattr(v, "error")
    e: Any = None
    if has_uncertainty:
        e = v.error
        v = v.value

    attributes: dict[str, Any] = {}
    with open_hdf5(file, "a") as hdf5_file:
        new_entry = hdf5_file.require_group(group)
        compression = get_hdf5_compression() if np.size(v) > 1 else {}
        new_entry.create_dataset(name="value", data=v, **compression)
        if has_uncertainty:
            new_entry.create_dataset(name="error", data=e, **compression)

        if u is not None:
            attributes["unit"] = u
        attributes["type"] = H5StoreTypes.DimensionalNDArray.name
        attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()
        attributes["ndim"] = np.ndim(v)

    existing_groups[group] = value
    return attributes, existing_groups


#### Equinox Modules ####
if eqx is not None:

    @_register(CUSTOM_H5_WRITERS, lambda x: isinstance(x, eqx.Module))
    def dump_equinox_module(
        file: FileSource, group: str, value: eqx.Module, existing_groups: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attributes: dict[str, Any] = {}
        with open_hdf5(file, "a") as hdf5_file:
            _ = hdf5_file.require_group(group)
            for field in fields(value):
                if not field.init:
                    continue
                item = getattr(value, field.name)
                with exact_dedup():
                    existing_groups = ExportableClassMixin._dump_h5_entry(
                        hdf5_file,
                        f"{group}/{field.name}",
                        item,
                        existing_groups=existing_groups,
                    )
            attributes["type"] = H5StoreTypes.Equinox.name
            attributes["module_class"] = f"{type(value).__module__}.{type(value).__name__}"
            attributes["timestamp"] = dt.datetime.now().astimezone().isoformat()

        existing_groups[group] = value
        return attributes, existing_groups

    @_register(CUSTOM_H5_LOADERS, H5StoreTypes.Equinox)
    def load_equinox_module(file: FileSource, group: str) -> eqx.Module:
        with open_hdf5(file) as hdf5_file:
            entry = get_group(hdf5_file, group)
            module_class = None
            if "module_class" in entry.attrs:
                module_class_path = entry.attrs["module_class"]
                module_name, class_name = module_class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                module_class = getattr(module, class_name, None)
            if module_class is None or not issubclass(module_class, eqx.Module):
                msg = f"Failed to load {group} because it was an Equinox entry, but the module class {module_class} is not a valid Equinox Module."
                raise H5StoreException(msg)
            field_values = {}
            for field in fields(module_class):
                if not field.init:
                    continue
                item_group = f"{group}/{field.name}"
                if field.name not in entry:
                    if field.default is not MISSING or field.default_factory is not MISSING:
                        continue
                    msg = f"Missing required Equinox field {field.name!r} at {group}."
                    raise H5StoreException(msg)
                _, item_value = ExportableClassMixin._load_h5_entry(hdf5_file, item_group)
                field_values[field.name] = item_value
            return module_class(**field_values)


@_register(CUSTOM_H5_LOADERS, H5StoreTypes.DimensionalNDArray)
def load_unit_or_error_ndarrays(file: FileSource, group: str) -> Any:
    with open_hdf5(file) as hdf5_file:
        entry = get_group(hdf5_file, group)
        v = get_dataset(entry, "value")[()]

        if "error" in entry:
            if Uncertainty is not None:
                e = get_dataset(entry, "error")[()]
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
                v = ur.Quantity(v, unit)
        except KeyError:
            pass

        ndim = read_h5_attr(entry, "ndim")

        if ndim == 0:
            sz = np.size(nominal_values(v)) if nominal_values is not None else np.size(v)
            if sz > 1:
                raise H5StoreException
            if np.ndim(v) > 0:
                v = v.reshape(())
    return v


def get_group(hdf5_handle: h5py.File | h5py.Group, name: str) -> h5py.Group:
    if name not in hdf5_handle:
        msg = f"Could not find group {name}!"
        raise KeyError(msg)
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
    ret = hdf5_handle[name]
    if isinstance(ret, h5py.Dataset):
        return ret
    else:
        msg = f"Expected group {name} but found {type(ret)}!"
        raise TypeError(msg)
