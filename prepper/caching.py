from annotationlib import Format
from dataclasses import dataclass, field
import functools
from functools import wraps
import inspect
import pickle
from typing import (
    TYPE_CHECKING,
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Self,
    TypeVar,
    cast,
    overload,
)

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

__all__ = [
    "break_key",
    "cached_property",
    "local_cache",
    "make_cache_name",
]

KWD_SENTINEL = "__prepper_kwd_sentinel__"

Instance = TypeVar("Instance")  # class using the descriptor
Arguments = ParamSpec("Arguments")  # parameter specs of the decorated method
Value = TypeVar("Value")  # return value of the decorated method


def _freeze_argument(value: Any) -> tuple[Any, ...]:
    """Snapshot arguments without delegating key equality to mutable objects."""
    if type(value) is np.ndarray and not value.dtype.hasobject:
        # The same immutable payload backs both equality and reconstructed arguments.
        return (
            "ndarray",
            pickle.dumps(value.dtype, protocol=5),
            value.shape,
            value.tobytes(order="C"),
        )
    if isinstance(value, np.generic):
        # HDF5 scalar loaders return Python values; retain compatibility with those keys.
        value = value.item()
    elif isinstance(value, np.ndarray):
        # Pickle object elements and subclass state, never their raw object pointers.
        value = value.copy(order="C")
    return ("pickle", pickle.dumps(value, protocol=5))


def _thaw_argument(value: tuple[Any, ...]) -> Any:
    """Reconstruct a stored argument for export from its private immutable snapshot."""
    if value[0] == "ndarray":
        dtype = pickle.loads(value[1])  # noqa: S301 -- Snapshot produced by _freeze_argument.
        if dtype.itemsize == 0:
            return np.empty(value[2], dtype=dtype)
        return np.frombuffer(value[3], dtype=dtype).reshape(value[2])
    return pickle.loads(value[1])  # noqa: S301 -- Snapshot produced by _freeze_argument.


@dataclass(frozen=True, slots=True, init=False)
class _HashedSeq:
    """An immutable call snapshot with separate positional and ordered keyword arguments."""

    _args: tuple[tuple[Any, ...], ...]
    _kwargs: tuple[tuple[str, tuple[Any, ...]], ...]
    hashvalue: int = field(compare=False)

    def __init__(self, tup: Iterable[Any], kwds: dict[str, Any] | None = None) -> None:
        args = tuple(_freeze_argument(value) for value in tup)
        kwargs = tuple((name, _freeze_argument(value)) for name, value in (kwds or {}).items())
        object.__setattr__(self, "_args", args)
        object.__setattr__(self, "_kwargs", kwargs)
        object.__setattr__(self, "hashvalue", hash((args, kwargs)))

    def __hash__(self) -> int:
        return self.hashvalue

    def __reduce__(self) -> tuple[type[_HashedSeq], tuple[list[Any], dict[str, Any]]]:
        # Recompute Python's process-dependent byte hashes when loading a pickled key.
        return _HashedSeq, break_key(self)


def break_key(key: Any) -> tuple[list[Any], dict[str, Any]]:
    """Return snapshotted call arguments for export, accepting legacy flat keys too."""
    if isinstance(key, _HashedSeq):
        return (
            [_thaw_argument(value) for value in key._args],
            {name: _thaw_argument(value) for name, value in key._kwargs},
        )
    values = list(key)
    for index, value in enumerate(values):
        if isinstance(value, str) and value == KWD_SENTINEL:
            keyword_values = values[index + 1 :]
            if len(keyword_values) % 2:
                msg = "Malformed legacy cache key: missing a keyword argument value."
                raise ValueError(msg)
            return values[:index], dict(chunks(keyword_values, 2))
    return values, {}


def make_cache_name(name: str) -> str:
    return f"__cache_{name}__"


def _make_key(args: tuple[Any, ...], kwds: dict[str, Any]) -> _HashedSeq:
    """Build an exact cache key that preserves ndarray metadata and keyword order."""
    return _HashedSeq(args, kwds)


def chunks(lst: list[Any], n: int) -> Iterator[list[Any]]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def _cache_wrapper(
    user_function: Callable[Concatenate[Instance, Arguments], Value],
) -> Callable[Concatenate[Instance, Arguments], Value]:
    # Constants shared by all lru cache instances:

    @wraps(user_function)
    def wrapper(instance: Instance, /, *args: Arguments.args, **kwds: Arguments.kwargs) -> Value:
        cache = object.__getattribute__(instance, "__dict__")
        key = _make_key(args, kwds)
        fname = make_cache_name(user_function.__qualname__)
        if fname in cache:
            function_cache = cache[fname]
            if key in function_cache:
                return function_cache[key]
        else:
            cache[fname] = {}
        try:
            result = user_function(instance, *args, **kwds)
        except Exception as e:
            tb = e.__traceback__
            if tb:
                tb = tb.tb_next
            e.__traceback__ = tb
            del e, tb
            raise
        cache[fname][key] = result
        return result

    return wrapper


class cached_property(Generic[Instance, Value]):  # noqa: N801 -- Established decorator API.
    """
    A property computed once per instance, stored under its qualified name.
    Assigning overrides the cached value, and deleting it permits recomputation.
    Qualified names preserve separate cache entries when overrides use super().
    """

    __slots__ = ("__name__", "__qualname__", "__type_params__", "func")
    func: Callable[[Instance], Value]

    def __init__(self, func: Callable[[Instance], Value]) -> None:
        self.__name__ = func.__name__
        self.__qualname__ = func.__qualname__
        self.__type_params__ = func.__type_params__
        self.func = func

    @overload
    def __get__(self, instance: Instance, owner: object) -> Value: ...

    @overload
    def __get__(self, instance: None, owner: object) -> Self: ...

    def __get__(self, instance: Instance | None, owner: object) -> Self | Value:
        try:
            if instance is None:
                return self
            qualname = self.func.__qualname__
            if qualname in instance.__dict__:
                return instance.__dict__[qualname]

            instance.__dict__[self.func.__qualname__] = self.func(instance)

            return instance.__dict__[self.func.__qualname__]
        except Exception as e:
            tb = e.__traceback__
            if tb:
                tb = tb.tb_next
            e.__traceback__ = tb
            del e, tb
            raise

    def __set__(self, instance: Instance, value: Value) -> None:
        instance.__dict__[self.func.__qualname__] = value

    def __delete__(self, instance: Instance) -> None:
        # Deleting an uncached property is harmless and never evaluates the getter.
        instance.__dict__.pop(self.func.__qualname__, None)


class local_cache(Generic[Instance, Arguments, Value]):  # noqa: N801 -- Established decorator API.
    """
    Caches the result of a function call locally to the instance.
    This is different from functools.cache, which caches the function result to the class, which means that
    the cache is not invalidated even if the instance is deleted.
    The functools implementation is good for i.e. HTTPS lookups,
    but causes memory leaks if the class being cached is very large
    """

    user_func: Callable[Concatenate[Instance, Arguments], Value]

    def __init__(self, wrapped_func: Callable[Concatenate[Instance, Arguments], Value]) -> None:
        self.user_func = wrapped_func
        self.display_formatter = functools.partial(display_formatter, func=wrapped_func)

    @overload
    def __get__(self, instance: Instance, owner: type[Instance]) -> Callable[Arguments, Value]: ...

    @overload
    def __get__(
        self, instance: None, owner: type[Instance]
    ) -> Callable[Concatenate[Instance, Arguments], Value]: ...

    def __get__(
        self, instance: Instance | None, owner: type[Instance]
    ) -> Callable[Arguments, Value] | Callable[Concatenate[Instance, Arguments], Value]:
        if instance is None:
            return self.user_func
        else:
            partial_function = functools.update_wrapper(
                functools.partial(_cache_wrapper(self.user_func), instance),
                self.user_func,
            )
            partial_function.__repr__ = self.display_formatter
            partial_function.__str__ = self.display_formatter
            # Jupyter discovers this dynamically attached display hook.
            partial_function._repr_html_ = self.display_formatter  # pyrefly: ignore[missing-attribute]
            if self.user_func.__doc__ is not None:
                partial_function.__doc__ = self.user_func.__doc__
            else:
                partial_function.__doc__ = (
                    f"Cached version of {self.user_func.__module__}.{self.user_func.__qualname__}"
                )

            return cast("Callable[Arguments, Value]", partial_function)

    def __set__(self, obj: Instance, value: tuple[_HashedSeq, Value]) -> None:
        fname = make_cache_name(self.user_func.__qualname__)
        if fname not in obj.__dict__:
            obj.__dict__[fname] = {}

        key, return_value = value
        if isinstance(key, _HashedSeq):
            obj.__dict__[fname][key] = return_value
        else:
            msg = f"Can't assign {value} to the cache of {self.user_func.__qualname__}!"
            raise TypeError(msg)


def display_formatter(func: Callable[..., Any]) -> str:
    is_in_main = func.__module__ == "__main__"
    display_name = f"{func.__module__}.{func.__qualname__}"

    sig = inspect.signature(func, eval_str=False, annotation_format=Format.FORWARDREF)
    function_signature = str(sig).replace("(self, ", "(").replace(" ", "")
    display_name = f"{display_name}{function_signature}"
    if is_in_main:
        display_name = f"<{display_name}>"
    return display_name
