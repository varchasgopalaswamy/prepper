# -*- coding: utf-8 -*-
from __future__ import annotations

import inspect
import os

import h5py
import numpy as np
import pytest

from prepper import (
    cached_property,
    ExportableClassMixin,
    local_cache,
    saveable_class,
)


@saveable_class("0.0.1", save=["test_string"])
class SimpleSaveableClass(ExportableClassMixin):
    """
    A simple saveable class, used to test saving as an attribute of another
    saveable class
    """

    def __init__(self):
        self.test_string = "test string SimpleSaveableClass"


class NotASaveableClass:
    """
    This class is not decorated to be saveable, and an exception
    should be raised if you try
    """

    def __init__(self):
        self.test_int = 1


# Varaible types to try and save
saveable_variables = [
    (None),  # None
    (1),  # int
    (1.1),  # float
    ("testing testing"),  # string
    (np.float32(1.1)),  # np.float32
    (np.ones(10, dtype=np.float32)),  # ndarray float32
    (np.random.random((5, 3, 2))),  # ndarray 3D
    ({"key1": 1, 2: "two", 3.3: 3}),  # dict with string, int, float
    ([1, 2, 3]),  # list (all the same datatype)
    ([1, "two", 3.3]),  # List (all different datatypes)
    (SimpleSaveableClass()),  # Saveable class
]

# Object, error, msg
unsaveable_variables = [
    (
        NotASaveableClass(),
        ValueError,
        "that does not support being saved to an HDF5 file!",
    ),  # Not a saveable class
]


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


@pytest.mark.parametrize("var", saveable_variables)
def test_export_import(var, tmp_path):
    """
    Test that you can export and import an object with every supported type
    """

    @saveable_class("0.0.1", save=["test_var"])
    class ThisClassHasAVariable(ExportableClassMixin):
        def __init__(self):
            self.test_var = var

    path = os.path.join(tmp_path, "tmp.hdf5")

    obj1 = ThisClassHasAVariable()
    obj1.to_hdf5(path)

    obj2 = ThisClassHasAVariable.from_hdf5(path)

    # Assert that ExportableClassMixin thinks the two objects are equal
    assert obj1 == obj2

    # Check that the imported object actually has the same type as the
    # original
    #
    # TODO: Confirm that we actually want to be saving python int and float
    # as numpy types
    if type(var) == int:
        assert type(obj2.test_var) == np.int32
    elif type(var) == float:
        assert type(obj2.test_var) == np.float64
    else:
        assert type(obj2.test_var) == type(var)


@pytest.mark.parametrize("var", saveable_variables)
def test_export_hdf5_format(var, tmp_path):
    """
    Test that you can export and import an object with every supported type
    """

    @saveable_class("0.0.1", save=["test_var"])
    class ThisClassHasAVariable(ExportableClassMixin):
        def __init__(self):
            self.test_var = var

    path = os.path.join(tmp_path, "tmp.hdf5")

    obj1 = ThisClassHasAVariable()
    obj1.to_hdf5(path)

    with h5py.File(path, "r") as f:
        # Test that the base level attributes are all there
        ensure_required_attributes(obj1, f)

        # Test that the exportable attribute was exported
        assert "test_var" in f.keys()

        # Test that the exportable attribute has all the required attributes
        ensure_required_attributes(var, f["test_var"])


@pytest.mark.parametrize("var,error,msg", unsaveable_variables)
def test_error_during_export(var, error, msg, tmp_path):
    """
    Verify that expected exceptions are raised
    """
    path = os.path.join(tmp_path, "tmp.hdf5")

    @saveable_class("0.0.1", save=["test_var"])
    class ThisClassHasABadVariable(ExportableClassMixin):
        def __init__(self):
            self.test_var = var

    obj = ThisClassHasABadVariable()

    with pytest.raises(error, match=msg):
        obj.to_hdf5(path)


def test_export_import_property(tmp_path):
    """
    Test exporting and importing a cached property
    """

    @saveable_class("0.0.1", save=["answer"])
    class ClassWithProperty(ExportableClassMixin):
        """
        This class has a cached property
        """

        @property
        def answer(self):
            return 42

    path = os.path.join(tmp_path, "tmp.hdf5")

    obj1 = ClassWithProperty()
    obj1.to_hdf5(path)

    # Assert that everything was exported correctly
    with h5py.File(path, "r") as f:
        ensure_required_attributes(obj1, f)
        ensure_required_attributes(obj1, f["answer"])

    # Import and verify equality
    obj2 = ClassWithProperty.from_hdf5(path)
    assert obj1 == obj2


def test_export_import_cached_property(tmp_path):
    """
    Test exporting and importing a cached property
    """

    @saveable_class("0.0.1", save=["expensive_fcn"])
    class ClassWithCachedProperty(ExportableClassMixin):
        """
        This class has a cached property
        """

        @cached_property
        def expensive_fcn(self):
            return "This string took 1000 hours to calculate"

    path = os.path.join(tmp_path, "tmp.hdf5")

    obj1 = ClassWithCachedProperty()
    obj1.to_hdf5(path)

    # So far we haven't actually called the cached_property, so it should
    # NOT be saved?

    with h5py.File(path, "r") as f:
        print(list(f["expensive_fcn"].attrs.keys()))
        print(f["expensive_fcn"][...])
