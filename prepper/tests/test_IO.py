# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import os
import tempfile

import attr
import h5py
import numpy as np
import pytest
from hypothesis import given, strategies
from hypothesis.extra import numpy as hnp

from prepper import (
    cached_property,
    ExportableClassMixin,
    local_cache,
    saveable_class,
)

NATIVE_DTYPES = [str, int, float]


@saveable_class("0.0.1", attributes=["a"], functions=["mult"])
class SimpleSaveableClass(ExportableClassMixin):
    """
    A simple saveable class, used to test saving as an attribute of another
    saveable class
    """

    def __init__(self, a):
        self._a = a

    @cached_property
    def a(self):
        return self._a

    @local_cache
    def mult(self, factor):
        return self.a * factor


def roundtrip(obj: ExportableClassMixin):
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = os.path.join(tmpdir, "test.hdf5")
        obj.to_hdf5(filename)

        with h5py.File(filename, "r") as hdf5_file:
            # Check that all the header info is there
            ensure_required_attributes(obj, hdf5_file)
            # Check that all the keys have been written
            for key in obj._exportable_functions:
                assert key.split(".")[1] in hdf5_file
            for key in obj._exportable_attributes:
                assert key.split(".")[1] in hdf5_file

        new_obj = obj.from_hdf5(filename)

    assert obj == new_obj
    return new_obj


@given(
    strategies.lists(
        strategies.text(
            min_size=1,
            alphabet=strategies.characters(
                blacklist_characters="\x00", blacklist_categories=("Cs",)
            ),
        ),
        min_size=1,
    )
)
def test_with_str_list(x):
    test_class = SimpleSaveableClass(x)
    _ = test_class.mult(2)
    new_class = roundtrip(test_class)


@given(
    hnp.arrays(
        elements=strategies.floats(
            allow_nan=False, allow_infinity=False, width=32
        ),
        shape=hnp.array_shapes(min_dims=1, max_dims=4),
        dtype=float,
    )
)
def test_with_floats(x):

    test_class = SimpleSaveableClass(x)
    _ = test_class.mult(2)
    new_class = roundtrip(test_class)


@given(
    hnp.arrays(
        elements=strategies.integers(min_value=-100, max_value=100),
        shape=hnp.array_shapes(min_dims=1, max_dims=4),
        dtype=int,
    )
)
def test_with_ints(x):

    test_class = SimpleSaveableClass(x)
    _ = test_class.mult(2)
    new_class = roundtrip(test_class)


@given(
    strategies.lists(
        strategies.floats(allow_nan=False, allow_infinity=False, width=32),
        min_size=1,
    )
)
def test_with_float_list(x):
    test_class = SimpleSaveableClass(x)
    _ = test_class.mult(2)
    new_class = roundtrip(test_class)


@given(
    strategies.lists(
        strategies.integers(min_value=-100, max_value=100),
        min_size=1,
    )
)
def test_with_int_list(x):
    test_class = SimpleSaveableClass(x)
    _ = test_class.mult(2)
    new_class = roundtrip(test_class)


# Lists of attributes required for different types of objects
req_class_attrs = [
    "module",
    "class",
    "timestamp",
    "version",
    "code",
    "type",
    "api_version",
]

req_dataset_attrs = ["timestamp", "type"]

req_none_attrs = ["type"]


def ensure_required_attributes(var, group):
    """
    Given an object, returns the attributes requried to be exported with that
    object

    var : Variable that was saved to this file

    group: h5py.Group object where the object is stored

    """
    if inspect.isclass(var):
        req_attrs = req_class_attrs
    elif var is None:
        req_attrs = ["type"]
    else:
        req_attrs = req_dataset_attrs

    for attr in req_attrs:
        assert attr in group.attrs.keys()