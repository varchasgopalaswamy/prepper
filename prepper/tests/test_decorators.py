from __future__ import annotations

from hypothesis import given, strategies
from hypothesis.extra import numpy as hnp
import pytest

from prepper import (
    ExportableClassMixin,
    cached_property,
    local_cache,
    saveable_class,
)
from prepper.caching import _make_key


class SimpleSaveableClass(ExportableClassMixin):
    """
    A simple saveable class, used to test saving as an attribute of another
    saveable class
    """

    def __init__(self):
        self.test_int = 1

    @cached_property
    def test_string(self):
        return "test string SimpleSaveableClass"

    @local_cache
    def square(self, x):
        return x**2


class SimpleSaveableClass2(SimpleSaveableClass):
    """
    A simple saveable class, used to test saving as an attribute of another
    saveable class
    """

    def __init__(self):
        self.test_int = 1

    @cached_property
    def test_string(self):
        base = super().test_string
        return base + " test string SimpleSaveableClass2"

    @local_cache
    def square(self, x):
        return 2 * super().square(x)


class NotASaveableClass:
    """
    This class is not decorated to be saveable, and an exception
    should be raised if you try
    """

    def __init__(self):
        self.test_int = 1


def test_saveable_class():
    decorator = saveable_class(
        api_version="0.0.1",
        attributes=["test_string"],
        functions=["square"],
    )
    bad_decorators = [
        saveable_class(
            api_version="0.0.1",
            attributes=["test_string"],
            functions=["square2"],
        ),
        saveable_class(
            api_version="0.0.1",
            attributes=["test_int", "test_string"],
            functions=["square"],
        ),
    ]
    pytest.raises(TypeError, decorator, NotASaveableClass)
    decorated = decorator(SimpleSaveableClass)
    assert decorated._exportable_functions == ["SimpleSaveableClass.square"]
    assert decorated._exportable_attributes == ["SimpleSaveableClass.test_string"]

    for d in bad_decorators:
        pytest.raises(ValueError, d, SimpleSaveableClass)


def test_cached_property():
    test_class = SimpleSaveableClass2()

    # Make sure __dict__ doesn't have the cached property
    assert not any("test_string" in k for k in test_class.__dict__)

    # Make sure the cached property works correctly with the super() call
    assert (
        test_class.test_string
        == "test string SimpleSaveableClass test string SimpleSaveableClass2"
    )

    # Make sure the cached property stores the parent and child calls
    assert (
        test_class.__dict__["SimpleSaveableClass.test_string"]
        == "test string SimpleSaveableClass"
    )
    assert (
        test_class.__dict__["SimpleSaveableClass2.test_string"]
        == "test string SimpleSaveableClass test string SimpleSaveableClass2"
    )


@given(
    hnp.arrays(
        elements=strategies.floats(allow_nan=False, allow_infinity=False, width=32),
        shape=(10,),
        dtype=float,
    )
)
def test_local_cache(x):
    test_class2 = SimpleSaveableClass2()

    for x_ in x:
        assert test_class2.square(x_) == 2 * x_**2
        key = _make_key((x_,), {})

        # Make sure the cache is stores the parent and child calls
        assert (
            test_class2.__dict__["__cache_SimpleSaveableClass.square__"][key] == x_**2
        )
        assert (
            test_class2.__dict__["__cache_SimpleSaveableClass2.square__"][key]
            == 2 * x_**2
        )
