from enum import IntEnum
import math

import loguru
import numpy as np
import pytest

from prepper.caching import _make_key
from prepper.config import configure, get_settings
from prepper.utils import check_equality, get_element_from_number_and_weight


@pytest.fixture(autouse=True)
def restore_settings():
    previous = get_settings().model_dump()
    configure(compare_rtol=1e-5)
    yield
    configure(**previous)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (1.0, 1.000001, True),
        (1.0, 1.001, False),
        (0.0, 1e-100, False),
        (1e-100, 2e-100, False),
        (math.nan, math.nan, True),
        (math.nan, 1.0, False),
        (math.inf, math.inf, True),
        (math.inf, -math.inf, False),
        (math.inf, 1e308, False),
        (1e308, -1e308, False),
        (1e308, 1.000001e308, True),
        (5e-324, 1e-323, False),
        (1 + 2j, 1.000001 + 2j, True),
        (1_000_000, 1_000_001, False),
        (np.int64(1_000_000), np.int64(1_000_001), False),
        (True, False, False),
        (None, None, True),
        ("hello", "different", False),
    ),
)
def test_scalar_equality_is_symmetric(left, right, expected):
    assert check_equality(left, right) is expected
    assert check_equality(right, left) is expected


def test_runtime_tolerance_and_exact_mode():
    assert not check_equality(1.0, 1.01)
    configure(compare_rtol=0.02)
    assert check_equality(1.0, 1.01)
    assert not check_equality(0.0, 1e-100)
    assert not check_equality(100, 101)
    configure(compare_rtol=0)
    assert not check_equality(1.0, np.nextafter(1.0, 2.0))
    assert check_equality(np.array([np.nan, 1.0]), np.array([np.nan, 1.0]))


def test_relative_tolerance_is_symmetric_at_boundary():
    configure(compare_rtol=0.1)
    assert check_equality(0.9, 1.0)
    assert check_equality(1.0, 0.9)
    assert not check_equality(0.89, 1.0)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    (
        (np.array([1.0, 2.0]), np.array([1.000001, 2.000001]), True),
        (np.array([[1.0], [1.0]]), np.array([[1.0, 1.0]]), False),
        (np.ones((1, 2)), np.ones((2, 2)), False),
        (np.array([1], dtype="i4"), np.array([1], dtype="i8"), False),
        (np.array([1.0], dtype="f4"), np.array([1.0], dtype="f8"), False),
        (np.array([1_000_000]), np.array([1_000_001]), False),
        (np.array([True]), np.array([False]), False),
        (np.array(["a", "b"]), np.array(["a", "c"]), False),
        (np.array([np.nan, np.inf]), np.array([np.nan, np.inf]), True),
        (np.array([1.0]), [1.0], False),
    ),
)
def test_arrays_preserve_shape_dtype_and_discrete_values(left, right, expected):
    assert check_equality(left, right) is expected
    assert check_equality(right, left) is expected


def test_nested_containers_and_cache_keys():
    left = {"x": [np.array([1.0, 2.0]), {"v": (3.0, None)}]}
    right = {"x": [np.array([1.000001, 2.0]), {"v": (3.000001, None)}]}
    assert check_equality(left, right)
    assert not check_equality(left, {"y": right["x"]})
    assert not check_equality(left, {"x": []})
    key1 = _make_key((np.array([1.0, 2.0]),), {"scale": 2})
    key2 = _make_key((np.array([1.0, 2.0]),), {"scale": 2})
    assert check_equality({key1: np.array([3.0])}, {key2: np.array([3.000001])})
    assert not check_equality({key1: 1}, {key2: 2})


def test_mapping_keys_and_sequence_kinds_remain_exact():
    configure(compare_rtol=0.1)
    assert check_equality({1.0: 2.0}, {1.0: 2.001})
    assert not check_equality({1.0: "value"}, {1.0000001: "value"})
    assert not check_equality([1.0, 2.0], (1.0, 2.0))
    assert not check_equality({"value": [1.0]}, {"value": (1.0,)})


def test_object_and_structured_arrays():
    left = np.empty(2, dtype=object)
    right = np.empty(2, dtype=object)
    left[:] = [{"x": 1.0}, [np.array([2.0])]]
    right[:] = [{"x": 1.000001}, [np.array([2.000001])]]
    assert check_equality(left, right)
    dtype = [("number", "i8"), ("value", "f8")]
    assert check_equality(np.array([(1, 2.0)], dtype), np.array([(1, 2.000001)], dtype))
    assert not check_equality(np.array([(1, 2.0)], dtype), np.array([(2, 2.0)], dtype))


def test_recursive_container_cycles():
    left, right = [], []
    left.append(left)
    right.append(right)
    assert check_equality(left, right)
    right.append(1)
    assert not check_equality(left, right)


def test_enumerators_are_exact():
    class Code(IntEnum):
        FIRST = 100_000
        SECOND = 100_001

    configure(compare_rtol=1)
    assert check_equality(Code.FIRST, Code.FIRST)
    assert not check_equality(Code.FIRST, Code.SECOND)
    assert not check_equality(Code.FIRST, 100_000)


def test_quantities_preserve_units():
    pint = pytest.importorskip("pint")
    registry = pint.UnitRegistry()
    assert check_equality(
        registry.Quantity([1.0, 2.0], "m"), registry.Quantity([1.000001, 2.0], "m")
    )
    assert not check_equality(registry.Quantity(1.0, "m"), registry.Quantity(1.0, "s"))
    assert not check_equality(registry.Quantity(1.0, "m"), registry.Quantity(100.0, "cm"))
    assert not check_equality(registry.Quantity(1.0, "m"), 1.0)


def test_uncertainties_compare_both_values_and_errors():
    au = pytest.importorskip("auto_uncertainties")
    assert check_equality(au.Uncertainty(1.0, 0.1), au.Uncertainty(1.000001, 0.1000001))
    assert not check_equality(au.Uncertainty(1.0, 0.1), au.Uncertainty(1.0, 0.2))
    assert not check_equality(au.Uncertainty(1.0, 0.1), 1.0)
    pint = pytest.importorskip("pint")
    registry = pint.UnitRegistry()
    assert not check_equality(
        registry.Quantity(au.Uncertainty([1.0, 2.0], [0.1, 0.2]), "m"),
        registry.Quantity(au.Uncertainty([1.0, 2.0], [0.2, 0.2]), "m"),
    )


def test_xarray_preserves_coordinates_and_metadata():
    xr = pytest.importorskip("xarray")
    left = xr.DataArray([1.0], dims=["x"], coords={"x": [0]}, attrs={"unit": "m"})
    assert check_equality(left, left.copy())
    assert not check_equality(left, left.assign_attrs(unit="s"))
    assert not check_equality(left, left.assign_coords(x=[1]))
    assert not check_equality(left, left.to_dataset(name="value"))
    assert check_equality(left.to_dataset(name="value"), left.to_dataset(name="value"))


def test_arbitrary_equality_errors_remain_visible_and_logging_state_is_untouched(monkeypatch):
    def unexpected_logging_change(*args):
        pytest.fail("Comparison changed the caller's logging configuration")

    monkeypatch.setattr(loguru.logger, "disable", unexpected_logging_change)
    monkeypatch.setattr(loguru.logger, "enable", unexpected_logging_change)
    assert check_equality(1, 1, log=True)
    assert not check_equality({"x": 1}, {"x": 2}, log=True)

    class BrokenEquality:
        __hash__ = None

        def __eq__(self, other):
            msg = "Cannot compare this value"
            raise ValueError(msg)

    with pytest.raises(TypeError, match="Cannot compare values of types BrokenEquality"):
        check_equality(BrokenEquality(), BrokenEquality(), log=True)


def test_isotope_lookup_and_invalid_input(monkeypatch):
    pt = pytest.importorskip("periodictable")
    assert get_element_from_number_and_weight(1, pt.elements.H[2].mass) is pt.elements.H[2]
    assert get_element_from_number_and_weight(6, pt.elements.C.mass) is pt.elements.C
    for z, a in ((999, 1), (1, math.nan), (math.inf, 1)):
        with pytest.raises(ValueError, match="Could not find a matching element"):
            get_element_from_number_and_weight(z, a)
    monkeypatch.setattr("prepper.utils.elements", None)
    with pytest.raises(ImportError, match="Could not import periodictable"):
        get_element_from_number_and_weight(1, 1)
