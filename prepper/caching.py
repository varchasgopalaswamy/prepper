from annotationlib import Format
import functools
from functools import wraps
import inspect
from typing import (
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    Self,
    TypeVar,
    cast,
    overload,
)
lazy from collections.abc import Callable, Iterable, Iterator

from joblib import hash as joblib_hash
from numpy import ndarray

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


class _HashedSeq(list[Any]):
    """
    This class guarantees that hash() will be called no more than once
    per element.  This is important because the lru_cache() will hash
    the key multiple times on a cache miss.
    """

    __slots__ = ["hashvalue"]

    def __init__(self, tup: Iterable[Any]) -> None:
        hash_values = []
        for v in tup:
            try:
                hash_values.append(v.item())
            except Exception:  # noqa: BLE001 -- Arbitrary objects may reject scalar conversion.
                hash_values.append(v)

        self[:] = hash_values

        digest = joblib_hash(hash_values, hash_name="sha1")
        if digest is None:
            msg = "Could not hash the cache key."
            raise TypeError(msg)
        self.hashvalue = int(digest, 16)

    def __hash__(self) -> int:
        return self.hashvalue


def break_key(key: Any) -> tuple[list[Any], dict[str, Any]]:
    """Breaks a function cache key into the args and kwargs"""
    args = []
    kwargs = {}
    if KWD_SENTINEL in key:
        kwd_split_idx = key.index(KWD_SENTINEL)
        args = key[0:kwd_split_idx]
        assert len(key[kwd_split_idx + 1 :]) % 2 == 0
        kwargs = dict(chunks(key[kwd_split_idx + 1 :], 2))
    else:
        args = key
    return args, kwargs


def make_cache_name(name: str) -> str:
    return f"__cache_{name}__"


def _make_key(args: tuple[Any, ...], kwds: dict[str, Any]) -> _HashedSeq:
    """
    Make a cache key from optionally typed positional and keyword arguments
    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.
    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.
    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = args
    if kwds:
        key += (KWD_SENTINEL,)
        for k, v in kwds.items():
            v2 = tuple(v.tolist()) if isinstance(v, ndarray) else v
            key += (k, v2)
    return _HashedSeq(key)


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
        key = _make_key(args, dict(sorted(kwds.items())))
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
    A property that is only computed once per instance and then replaces
    itself with an ordinary attribute. Deleting the attribute resets the
    property. Implementation adapted from https://github.com/pydanny/cached-property
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
