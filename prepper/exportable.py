"""Persist decorated Python classes and their populated caches."""

from abc import ABCMeta
from annotationlib import Format
import copy
import datetime as dt
import importlib.metadata
from inspect import signature
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar

import h5py

from prepper._storage import FileSource, open_hdf5
from prepper.caching import break_key, make_cache_name
from prepper.enums import H5StoreTypes
from prepper.exceptions import H5StoreException
from prepper.utils import check_equality

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ExportableClassMixin", "saveable_class"]


class ExportableClassMixin(metaclass=ABCMeta):
    """Save constructor arguments, selected properties, and method caches in HDF5."""

    _constructor_args: dict[str, Any]
    _constructor_call: tuple[tuple[Any, ...], dict[str, Any]]
    api_version: float | str
    _exportable_attributes: ClassVar[list[str]] = []
    _exportable_functions: ClassVar[list[str]] = []
    __hash__ = None

    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        instance = object.__new__(cls)
        sig = signature(instance.__init__, annotation_format=Format.FORWARDREF)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        instance._constructor_args = dict(bound.arguments)
        instance._constructor_call = (bound.args, bound.kwargs)
        instance._initialized_from_file = False
        return instance

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def __copy__(self) -> Self:
        clone = object.__new__(type(self))
        clone.__dict__.update(self.__dict__)
        self._copy_slots(clone)
        return clone

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        clone = object.__new__(type(self))
        memo[id(self)] = clone
        clone.__dict__.update(copy.deepcopy(self.__dict__, memo))
        self._copy_slots(clone, memo)
        return clone

    def _copy_slots(self, clone: Self, memo: dict[int, Any] | None = None) -> None:
        for parent in type(self).__mro__:
            slots = getattr(parent, "__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for slot_name in slots:
                name = slot_name
                if name in {"__dict__", "__weakref__"}:
                    continue
                if name.startswith("__") and not name.endswith("__"):
                    name = f"_{parent.__name__.lstrip('_')}{name}"
                if hasattr(self, name):
                    value = getattr(self, name)
                    setattr(clone, name, value if memo is None else copy.deepcopy(value, memo))

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        return self._constructor_call

    def __eq__(self, other: object) -> bool:
        if type(self) is not type(other) or not isinstance(other, ExportableClassMixin):
            return NotImplemented
        if not check_equality(self._constructor_args, other._constructor_args):
            return False
        missing = object()
        for symbol in self._exportable_attributes:
            if not check_equality(
                self.__dict__.get(symbol, missing), other.__dict__.get(symbol, missing)
            ):
                return False
        return all(
            check_equality(
                self.__dict__.get(make_cache_name(symbol), missing),
                other.__dict__.get(make_cache_name(symbol), missing),
            )
            for symbol in self._exportable_functions
        )

    @property
    def initialized_from_file(self) -> bool:
        return self._initialized_from_file

    @classmethod
    def from_hdf5(cls, path: FileSource, group: str = "/") -> Self:
        """Validate metadata before constructing and restoring an instance."""
        from prepper.io_handlers import read_h5_attr  # noqa: PLC0415 -- Circular dependency.

        with open_hdf5(path) as handle:
            base = cls._validated_group(handle, group)
            stored_version = read_h5_attr(base, "api_version")
            if str(stored_version) != str(cls.api_version):
                msg = f"Stored API version {stored_version!r} does not match {cls.api_version!r}"
                raise H5StoreException(msg)
            cls._member_bindings(base)
            constructor = f"{group.rstrip('/')}/{cls.__name__}"
            if constructor in handle:
                kind, data = cls._load_h5_entry(handle, constructor)
                if kind != H5StoreTypes.ClassConstructor:
                    msg = f"Invalid constructor entry at {constructor}"
                    raise H5StoreException(msg)
                if isinstance(data, tuple):
                    args, kwargs = data
                else:
                    args, kwargs = (), data
            else:
                args, kwargs = (), {}
            instance = cls(*args, **kwargs)
            instance._read_hdf5_contents(handle, group)
        return instance

    @classmethod
    def _validated_group(cls, handle: h5py.File, group: str) -> h5py.Group:
        from prepper.io_handlers import get_group, read_h5_attr  # noqa: PLC0415

        base = get_group(handle, group)
        if cls._get_group_type(base) != H5StoreTypes.PythonClass:
            msg = f"HDF5 group {group!r} does not contain an exportable class"
            raise H5StoreException(msg)
        stored_class = read_h5_attr(base, "class")
        if stored_class != cls.__name__:
            msg = f"Stored class {stored_class!r} does not match {cls.__name__!r}"
            raise H5StoreException(msg)
        stored_module = base.attrs.get("module")
        stored_qualname = base.attrs.get("qualname")
        if stored_module is not None and stored_module != cls.__module__:
            msg = f"Stored module {stored_module!r} does not match {cls.__module__!r}"
            raise H5StoreException(msg)
        if stored_qualname is not None and stored_qualname != cls.__qualname__:
            msg = f"Stored qualified name {stored_qualname!r} does not match {cls.__qualname__!r}"
            raise H5StoreException(msg)
        return base

    @staticmethod
    def _member_bindings(base: h5py.Group) -> dict[str, str]:
        try:
            bindings = json.loads(base.attrs.get("__prepper_bindings__", "{}"))
        except (TypeError, ValueError) as exc:
            msg = f"Invalid member bindings at {base.name}"
            raise H5StoreException(msg) from exc
        if not isinstance(bindings, dict) or not all(
            isinstance(name, str) and isinstance(symbol, str) for name, symbol in bindings.items()
        ):
            msg = f"Invalid member bindings at {base.name}"
            raise H5StoreException(msg)
        return bindings

    def _read_hdf5_contents(self, file: FileSource, group: str) -> None:
        with open_hdf5(file) as handle:
            base = self._validated_group(handle, group)
            bindings = self._member_bindings(base)
            for name in base:
                if name == type(self).__name__:
                    continue
                path = f"{group.rstrip('/')}/{name}"
                kind, value = self._load_h5_entry(handle, path)
                if name in bindings:
                    symbol = bindings[name]
                else:
                    member_name = handle[path].attrs.get("member_name", name)
                    symbol = f"{self._get_bound_name(handle, path)}.{member_name}"
                if kind == H5StoreTypes.FunctionCache:
                    symbol = make_cache_name(symbol)
                self.__dict__[symbol] = value
        self._initialized_from_file = True

    @staticmethod
    def _load_h5_entry(file: FileSource, group: str) -> tuple[H5StoreTypes, Any]:
        from prepper.io_handlers import load_custom_h5_type  # noqa: PLC0415

        with open_hdf5(file) as handle:
            if group not in handle:
                msg = f"Could not find HDF5 entry {group!r}"
                raise H5StoreException(msg)
            kind = ExportableClassMixin._get_group_type(handle[group])
            return kind, load_custom_h5_type(handle, group, kind)

    def to_hdf5(self, path: str | Path) -> None:
        """Write a complete temporary file, then atomically replace the destination."""
        path = Path(path)
        if not path.parent.is_dir():
            msg = f"The parent directory for {path} does not exist"
            raise FileNotFoundError(msg)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        os.close(descriptor)
        temporary_path = Path(temporary)
        try:
            with open_hdf5(temporary_path, "w") as handle:
                self._write_hdf5_contents(handle, "/", {})
                handle.flush()
            with temporary_path.open("rb") as stream:
                os.fsync(stream.fileno())
            if path.exists():
                temporary_path.chmod(stat.S_IMODE(path.stat().st_mode))
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _write_hdf5_contents(
        self,
        file: FileSource,
        group: str,
        existing_groups: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from prepper.io_handlers import dump_class_constructor, exact_dedup, write_h5_attr  # noqa: PLC0415

        existing_groups[group] = self
        code_name = type(self).__module__.partition(".")[0]
        try:
            version = importlib.metadata.version(code_name)
        except importlib.metadata.PackageNotFoundError:
            version = ""
        metadata = dict(attributes or {})
        metadata.update(
            module=type(self).__module__,
            qualname=type(self).__qualname__,
            **{"class": type(self).__name__},
            timestamp=dt.datetime.now().astimezone().isoformat(),
            version=version,
            code=code_name,
            type=H5StoreTypes.PythonClass.name,
            api_version=self.api_version,
        )
        with open_hdf5(file, "a") as handle:
            base = handle.require_group(group)
            symbols = set(self._exportable_attributes) | set(self._exportable_functions)
            names: dict[str, list[str]] = {}
            for symbol in symbols:
                names.setdefault(symbol.rsplit(".", maxsplit=1)[-1], []).append(symbol)

            def entry_name(symbol: str) -> str:
                name = symbol.rsplit(".", maxsplit=1)[-1]
                if name == type(self).__name__ or len(names[name]) > 1:
                    return symbol
                return name

            bindings: dict[str, str] = {}
            for name, value in metadata.items():
                write_h5_attr(base, name, value)
            with exact_dedup():
                existing_groups = dump_class_constructor(handle, group, self, existing_groups)
            for symbol in self._exportable_attributes:
                if symbol in self.__dict__:
                    disk_name = entry_name(symbol)
                    existing_groups = self._dump_h5_entry(
                        handle,
                        f"{group.rstrip('/')}/{disk_name}",
                        self.__dict__[symbol],
                        existing_groups,
                    )
                    # Bindings belong to references, since values may be deduplicated links.
                    bindings[disk_name] = symbol
            for symbol in self._exportable_functions:
                cache = self.__dict__.get(make_cache_name(symbol))
                if cache is None:
                    continue
                bound_class, name = symbol.rsplit(".", maxsplit=1)
                disk_name = entry_name(symbol)
                function_path = f"{group.rstrip('/')}/{disk_name}"
                function_group = handle.require_group(function_path)
                write_h5_attr(function_group, "type", H5StoreTypes.FunctionCache.name)
                write_h5_attr(function_group, "bound_class", bound_class)
                write_h5_attr(function_group, "member_name", name)
                bindings[disk_name] = symbol
                width = len(str(max(len(cache), 1)))
                for index, (key, value) in enumerate(cache.items(), start=1):
                    args, kwargs = break_key(key)
                    call_path = f"{function_path}/{name}_call{index:0{width}d}"
                    for part, item in (
                        ("args", tuple(args)),
                        ("kwargs", kwargs),
                        ("value", value),
                    ):
                        if part == "value":
                            existing_groups = self._dump_h5_entry(
                                handle, f"{call_path}/{part}", item, existing_groups
                            )
                        else:
                            with exact_dedup():
                                existing_groups = self._dump_h5_entry(
                                    handle, f"{call_path}/{part}", item, existing_groups
                                )
            write_h5_attr(base, "__prepper_bindings__", json.dumps(bindings))
        return existing_groups

    @staticmethod
    def _dump_h5_entry(
        file: FileSource,
        entry_name: str,
        value: Any,
        existing_groups: dict[str, Any],
        attributes: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        from prepper.io_handlers import dump_custom_h5_type, write_h5_attr  # noqa: PLC0415

        with open_hdf5(file, "a") as handle:
            existing_groups = dump_custom_h5_type(handle, entry_name, value, existing_groups)
            for name, item in (attributes or {}).items():
                write_h5_attr(handle[entry_name], name, item)
        return existing_groups

    @staticmethod
    def _get_group_type(group: h5py.Group | h5py.Dataset) -> H5StoreTypes:
        from prepper.io_handlers import read_h5_attr  # noqa: PLC0415

        value = read_h5_attr(group, "type")
        try:
            return H5StoreTypes[value]
        except (KeyError, TypeError) as exc:
            msg = f"Unknown HDF5 entry type {value!r} at {group.name}"
            raise H5StoreException(msg) from exc

    @staticmethod
    def _get_bound_name(file: FileSource, groupname: str) -> str:
        from prepper.io_handlers import read_h5_attr  # noqa: PLC0415

        with open_hdf5(file) as handle:
            return read_h5_attr(handle[groupname], "bound_class")


E = TypeVar("E", bound=ExportableClassMixin)


def saveable_class(
    api_version: float | str,
    attributes: list[str] | None = None,
    functions: list[str] | None = None,
) -> Callable[[type[E]], type[E]]:
    """Declare the cached properties and methods included in persisted state."""

    def decorator(cls: type[E]) -> type[E]:
        if not issubclass(cls, ExportableClassMixin):
            msg = "Only subclasses of ExportableClassMixin can be decorated with saveable_class"
            raise TypeError(msg)
        inherited_attributes = list(getattr(cls, "_exportable_attributes", ()))
        inherited_functions = list(getattr(cls, "_exportable_functions", ()))
        for names, collected, description in (
            (attributes or [], inherited_attributes, "property/attribute"),
            (functions or [], inherited_functions, "function"),
        ):
            # A selected parent binding remains exportable when an override adds its own cache.
            overridden_names = [
                symbol.rsplit(".", maxsplit=1)[-1]
                for symbol in collected
                if symbol.rsplit(".", maxsplit=1)[-1] in cls.__dict__
            ]
            for name in [*overridden_names, *names]:
                if not hasattr(cls, name):
                    msg = f"{cls} and its parents does not have {description} {name}"
                    raise ValueError(msg)
                member = getattr(cls, name)
                if not hasattr(member, "__qualname__"):
                    msg = f"{cls}.{name} must be a cached property or method"
                    raise ValueError(msg)
                collected.append(member.__qualname__)
        cls._exportable_attributes = list(dict.fromkeys(inherited_attributes))
        cls._exportable_functions = list(dict.fromkeys(inherited_functions))
        cls.api_version = api_version
        return cls

    return decorator
