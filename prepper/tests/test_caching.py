from dataclasses import FrozenInstanceError
import pickle

import numpy as np
import pytest

from prepper import ExportableClassMixin, cached_property, local_cache, saveable_class
from prepper.caching import KWD_SENTINEL, _HashedSeq, _make_key, break_key, make_cache_name
from prepper.config import get_settings


class ArrayCache:
    def __init__(self):
        self.calls = 0

    @local_cache
    def describe(self, array):
        self.calls += 1
        return array.dtype, array.shape, array.copy()


@saveable_class("1", functions=["evaluate", "keywords"])
class PersistedArrayCache(ExportableClassMixin):
    def __init__(self):
        self.calls = 0

    @local_cache
    def evaluate(self, array):
        self.calls += 1
        return self.calls

    @local_cache
    def keywords(self, **kwargs):
        self.calls += 1
        return tuple(kwargs)


@pytest.mark.parametrize("keyword", (False, True))
def test_distinct_array_cache_keys_survive_approximate_storage_deduplication(
    tmp_path, monkeypatch, keyword
):
    monkeypatch.setattr(get_settings(), "compare_rtol", 1e-3)
    first = np.array([1.0, 2.0])
    second = np.array([1.000001, 2.000002])
    original = PersistedArrayCache()
    if keyword:
        assert original.evaluate(array=first) == 1
        assert original.evaluate(array=second) == 2
    else:
        assert original.evaluate(first) == 1
        assert original.evaluate(second) == 2
    filename = tmp_path / "cached-arrays.h5"
    original.to_hdf5(filename)
    restored = PersistedArrayCache.from_hdf5(filename)
    assert restored.calls == 0
    if keyword:
        assert restored.evaluate(array=first) == 1
        assert restored.evaluate(array=second) == 2
    else:
        assert restored.evaluate(first) == 1
        assert restored.evaluate(second) == 2
    assert restored.calls == 0
    cache = restored.__dict__[make_cache_name(PersistedArrayCache.evaluate.__qualname__)]
    assert len(cache) == 2


def test_keyword_order_survives_persisted_cache_roundtrip(tmp_path):
    original = PersistedArrayCache()
    assert original.keywords(a=1, b=2) == ("a", "b")
    assert original.keywords(b=2, a=1) == ("b", "a")
    filename = tmp_path / "ordered-keywords.h5"
    original.to_hdf5(filename)
    restored = PersistedArrayCache.from_hdf5(filename)
    assert restored.keywords(a=1, b=2) == ("a", "b")
    assert restored.keywords(b=2, a=1) == ("b", "a")
    assert restored.calls == 0


@pytest.mark.parametrize("keyword", (False, True))
@pytest.mark.parametrize(
    "array",
    (
        np.array(3),
        np.array([3]),
        np.arange(6).reshape(2, 3),
        np.arange(12).reshape(3, 4)[:, ::2],
        np.empty((0, 3)),
        np.array([np.nan, np.inf, -np.inf]),
        np.array([(1, 2.0)], dtype=[("integer", "i4"), ("real", "f8")]),
        np.array([{"value": [1, 2]}, "text"], dtype=object),
        np.empty((2,), dtype="V0"),
    ),
)
def test_array_calls_reuse_cache_for_equal_copies(array, keyword):
    instance = ArrayCache()
    if keyword:
        first = instance.describe(array=array)
        second = instance.describe(array=array.copy())
    else:
        first = instance.describe(array)
        second = instance.describe(array.copy())
    assert first is second
    assert instance.calls == 1
    assert first[0] == array.dtype
    assert first[1] == array.shape


@pytest.mark.parametrize("keyword", (False, True))
@pytest.mark.parametrize(
    ("first", "second"),
    (
        (np.array([1], dtype=np.int8), np.array([1], dtype=np.int64)),
        (np.array(1), np.array([1])),
        (np.array([1]), np.array([[1]])),
        (np.empty((0,)), np.empty((0, 3))),
        (np.array([1, 2]), np.array([1, 3])),
        (
            np.array([(1,)], dtype=[("first", "i4")]),
            np.array([(1,)], dtype=[("other", "i4")]),
        ),
        (np.array([0.0]), np.array([-0.0])),
    ),
)
def test_array_metadata_and_content_are_part_of_cache_identity(first, second, keyword):
    instance = ArrayCache()
    if keyword:
        result = instance.describe(array=first)
        other = instance.describe(array=second)
    else:
        result = instance.describe(first)
        other = instance.describe(second)
    assert result is not other
    assert instance.calls == 2
    assert other[0] == second.dtype
    assert other[1] == second.shape


def test_array_key_snapshot_survives_argument_mutation_and_export():
    array = np.arange(6).reshape(2, 3)
    expected = array.copy()
    key = _make_key((array,), {"other": array})
    original_hash = hash(key)
    cache = {key: "original"}
    array[:] = -1

    assert hash(key) == original_hash
    assert cache[_make_key((expected,), {"other": expected})] == "original"
    assert _make_key((array,), {"other": array}) not in cache
    args, kwargs = break_key(key)
    np.testing.assert_array_equal(args[0], expected)
    np.testing.assert_array_equal(kwargs["other"], expected)
    assert not args[0].flags.writeable


def test_object_array_snapshot_preserves_objects_instead_of_pointer_addresses():
    array = np.array([{"nested": [1, 2]}], dtype=object)
    key = _make_key((array,), {})
    array[0]["nested"].append(3)
    args, _ = break_key(key)
    assert args[0][0] == {"nested": [1, 2]}
    assert key == _make_key((np.array([{"nested": [1, 2]}], dtype=object),), {})


def test_mutable_container_keys_hold_independent_snapshots():
    value = {"items": [1, 2], "array": np.array([3, 4])}
    key = _make_key((value,), {})
    value["items"].append(5)
    value["array"][0] = 9
    args, _ = break_key(key)
    assert args[0]["items"] == [1, 2]
    np.testing.assert_array_equal(args[0]["array"], [3, 4])
    args[0]["items"].append(7)
    assert break_key(key)[0][0]["items"] == [1, 2]


def test_pickled_key_recomputes_cached_hash_from_argument_snapshots():
    key = _make_key((np.arange(3),), {"value": [1, 2]})
    expected = _make_key((np.arange(3),), {"value": [1, 2]})
    # Model a cached hash from another process, whose Python hash seed differs.
    object.__setattr__(key, "hashvalue", hash(expected) + 1)
    loaded = pickle.loads(pickle.dumps(key))  # noqa: S301 -- Roundtrip of the local test fixture.
    assert loaded == expected
    assert {loaded: "restored"}[expected] == "restored"


@pytest.mark.parametrize("scalar", (np.float64(1.25), np.int64(2), np.True_))
def test_numpy_scalars_match_loaded_python_scalars(scalar):
    assert _make_key((scalar,), {}) == _make_key((scalar.item(),), {})


class OrderedCache:
    def __init__(self):
        self.calls = 0

    @local_cache
    def describe(self, *args, **kwargs):
        self.calls += 1
        return args, tuple(kwargs.items())


def test_sentinel_string_is_an_ordinary_positional_or_keyword_value():
    instance = OrderedCache()
    positional = instance.describe(KWD_SENTINEL, "x", 1)
    keyword = instance.describe(x=1)
    assert positional == ((KWD_SENTINEL, "x", 1), ())
    assert keyword == ((), (("x", 1),))
    assert instance.calls == 2
    assert break_key(_make_key((KWD_SENTINEL,), {"value": KWD_SENTINEL})) == (
        [KWD_SENTINEL],
        {"value": KWD_SENTINEL},
    )


def test_keyword_order_is_preserved_by_cache_and_export():
    instance = OrderedCache()
    assert instance.describe(a=1, b=2) == ((), (("a", 1), ("b", 2)))
    assert instance.describe(b=2, a=1) == ((), (("b", 2), ("a", 1)))
    assert instance.describe(a=1, b=2) == ((), (("a", 1), ("b", 2)))
    assert instance.calls == 2
    _, kwargs = break_key(_make_key((), {"b": 2, "a": 1}))
    assert tuple(kwargs) == ("b", "a")


def test_key_is_immutable_and_exact_even_when_hashes_collide():
    first = _make_key(("first",), {})
    second = _make_key(("second",), {})
    with pytest.raises(FrozenInstanceError):
        first.hashvalue = 0
    # A hash collision must still compare the full immutable argument snapshots.
    object.__setattr__(second, "hashvalue", hash(first))
    cache = {first: 1, second: 2}
    assert len(cache) == 2
    assert cache[first] == 1
    assert cache[second] == 2


def test_legacy_flat_keys_remain_readable():
    assert break_key([1, KWD_SENTINEL, "value", 2]) == ([1], {"value": 2})
    assert break_key([1, 2]) == ([1, 2], {})
    with pytest.raises(ValueError, match="Malformed legacy cache key"):
        break_key([1, KWD_SENTINEL, "value"])


def test_method_exceptions_are_not_cached():
    class SometimesFails:
        calls = 0

        @local_cache
        def value(self, argument):
            self.calls += 1
            if self.calls == 1:
                msg = "try again"
                raise ValueError(msg)
            return argument

    instance = SometimesFails()
    with pytest.raises(ValueError, match="try again"):
        instance.value(3)
    assert instance.value(3) == 3
    assert instance.value(3) == 3
    assert instance.calls == 2


def test_cache_descriptor_accepts_loaded_entries_and_rejects_invalid_keys():
    instance = OrderedCache()
    key = _make_key((3,), {"a": 1})
    instance.describe = key, "loaded result"
    assert instance.describe(3, a=1) == "loaded result"
    assert instance.calls == 0
    with pytest.raises(TypeError, match="Can't assign"):
        instance.describe = "invalid key", "value"
    cache = instance.__dict__[make_cache_name(OrderedCache.describe.__qualname__)]
    assert cache == {key: "loaded result"}
    assert isinstance(key, _HashedSeq)


def test_cached_property_assignment_deletion_and_recomputation():
    class Example:
        calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            return self.calls

    instance = Example()
    del instance.value
    assert instance.calls == 0
    assert instance.value == 1
    assert instance.value == 1
    instance.value = 50
    assert instance.value == 50
    assert instance.__dict__[Example.value.__qualname__] == 50
    del instance.value
    assert instance.value == 2


def test_inherited_cached_property_deletion_keeps_parent_cache_independent():
    class Parent:
        calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            return self.calls

    class Child(Parent):
        @cached_property
        def value(self):
            return 2 * super().value

    instance = Child()
    assert instance.value == 2
    instance.value = 99
    assert instance.value == 99
    assert Parent.value.__get__(instance, Child) == 1
    del instance.value
    assert instance.value == 2
    assert instance.calls == 1
    Parent.value.__delete__(instance)
    del instance.value
    assert instance.value == 4
    assert instance.calls == 2


def test_cached_property_failure_is_retried():
    class SometimesFails:
        calls = 0

        @cached_property
        def value(self):
            self.calls += 1
            if self.calls == 1:
                msg = "try again"
                raise ValueError(msg)
            return self.calls

    instance = SometimesFails()
    with pytest.raises(ValueError, match="try again"):
        _ = instance.value
    assert instance.value == 2
    assert instance.value == 2
