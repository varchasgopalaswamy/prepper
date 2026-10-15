from collections.abc import Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

from aenum import Enum as ExtendedEnum
import loguru
import numpy as np

from prepper.config import get_settings

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    from auto_uncertainties import Uncertainty
except ImportError:
    Uncertainty = None

try:
    from pint import Quantity
except ImportError:
    Quantity = None

try:
    from periodictable import elements
except ImportError:
    elements = None

if TYPE_CHECKING:
    from periodictable.core import Element, Isotope

__all__ = ["check_equality", "get_element_from_number_and_weight"]


def check_equality(value1: Any, value2: Any, *, log: bool = False) -> bool:
    """
    Compare persisted values without discarding shape, dtype, units or uncertainties.

    Floating and complex numbers use the symmetric relative criterion
    ``abs(a-b) <= compare_rtol * max(abs(a), abs(b))``, with no absolute tolerance.
    Zero tolerance requests exact comparison. Matching NaNs are equal; infinities
    must match exactly. Arrays require equal shapes and dtypes, and discrete values
    compare exactly. Containers compare recursively; sequence types and mapping
    keys must match exactly, while mapping values use the configured tolerance.
    Quantities retain their units and uncertainty values retain their errors.
    Xarray objects use their metadata-aware ``identical`` comparison.

    Arbitrary objects use their equality implementation; comparison errors remain
    visible, and logging never changes the caller's Loguru configuration.
    """
    same = _check_equality(value1, value2, get_settings().compare_rtol, set())
    if not same and log:
        loguru.logger.debug("Values are different: {} and {}", value1, value2)
    return same


def _check_equality(value1: Any, value2: Any, rtol: float, active: set[tuple[int, int]]) -> bool:
    if value1 is value2:
        return True
    pair = (id(value1), id(value2))
    if pair in active:
        return True
    active.add(pair)
    try:
        return _compare_values(value1, value2, rtol, active)
    finally:
        active.remove(pair)


def _compare_values(value1: Any, value2: Any, rtol: float, active: set[tuple[int, int]]) -> bool:
    if Quantity is not None and (isinstance(value1, Quantity) or isinstance(value2, Quantity)):
        return (
            isinstance(value1, Quantity)
            and isinstance(value2, Quantity)
            and value1.units == value2.units
            and _check_equality(value1.magnitude, value2.magnitude, rtol, active)
        )
    if Uncertainty is not None and (
        isinstance(value1, Uncertainty) or isinstance(value2, Uncertainty)
    ):
        return (
            isinstance(value1, Uncertainty)
            and isinstance(value2, Uncertainty)
            and _check_equality(value1.value, value2.value, rtol, active)
            and _check_equality(value1.error, value2.error, rtol, active)
        )
    if xr is not None and (
        isinstance(value1, xr.Dataset | xr.DataArray)
        or isinstance(value2, xr.Dataset | xr.DataArray)
    ):
        if isinstance(value1, xr.Dataset) and isinstance(value2, xr.Dataset):
            return value1.identical(value2)
        if isinstance(value1, xr.DataArray) and isinstance(value2, xr.DataArray):
            return value1.identical(value2)
        return False
    if isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
        if not isinstance(value1, np.ndarray) or not isinstance(value2, np.ndarray):
            return False
        if value1.shape != value2.shape or value1.dtype != value2.dtype:
            return False
        if value1.dtype.names is not None:
            return all(
                _check_equality(value1[name], value2[name], rtol, active)
                for name in value1.dtype.names
            )
        if value1.dtype.kind == "O":
            return all(
                _check_equality(left, right, rtol, active)
                for left, right in zip(value1.flat, value2.flat, strict=True)
            )
        if value1.dtype.kind in "fc":
            return _floating_equal(value1, value2, rtol)
        return bool(np.array_equal(value1, value2))
    if isinstance(value1, Mapping) or isinstance(value2, Mapping):
        if not isinstance(value1, Mapping) or not isinstance(value2, Mapping):
            return False
        return _mapping_equal(value1, value2, rtol, active)
    if isinstance(value1, list | tuple) or isinstance(value2, list | tuple):
        if not isinstance(value1, list | tuple) or not isinstance(value2, list | tuple):
            return False
        if type(value1) is not type(value2):
            return False
        return len(value1) == len(value2) and all(
            _check_equality(left, right, rtol, active)
            for left, right in zip(value1, value2, strict=True)
        )
    if isinstance(value1, Enum | ExtendedEnum) or isinstance(value2, Enum | ExtendedEnum):
        return type(value1) is type(value2) and bool(value1 == value2)
    if isinstance(value1, float | complex | np.floating | np.complexfloating) and isinstance(
        value2, float | complex | np.floating | np.complexfloating
    ):
        return _floating_equal(np.asarray(value1), np.asarray(value2), rtol)
    try:
        same = value1 == value2
        if same is NotImplemented:
            return False
        return bool(np.all(same)) if isinstance(same, np.ndarray) else bool(same)
    except (TypeError, ValueError) as exc:
        msg = f"Cannot compare values of types {type(value1).__name__} and {type(value2).__name__}"
        raise TypeError(msg) from exc


def _floating_equal(value1: np.ndarray, value2: np.ndarray, rtol: float) -> bool:
    exact = (value1 == value2) | (np.isnan(value1) & np.isnan(value2))
    if bool(np.all(exact)) or rtol == 0:
        return bool(np.all(exact))
    finite = np.isfinite(value1) & np.isfinite(value2)
    # Normalize before subtracting to avoid overflow and underflow at extreme magnitudes.
    scale = np.maximum(
        np.maximum(np.abs(value1.real), np.abs(value1.imag)),
        np.maximum(np.abs(value2.real), np.abs(value2.imag)),
    )
    valid = finite & (scale != 0)
    dtype = np.result_type(value1.dtype, value2.dtype, np.float64)
    left = np.divide(value1, scale, out=np.zeros_like(value1, dtype=dtype), where=valid)
    right = np.divide(value2, scale, out=np.zeros_like(value2, dtype=dtype), where=valid)
    with np.errstate(over="ignore"):
        close = np.abs(left - right) <= rtol * np.maximum(np.abs(left), np.abs(right))
    return bool(np.all(exact | (finite & close)))


def _mapping_equal(
    value1: Mapping[Any, Any],
    value2: Mapping[Any, Any],
    rtol: float,
    active: set[tuple[int, int]],
) -> bool:
    if len(value1) != len(value2):
        return False
    if all(isinstance(key, str | bytes) for key in (*value1, *value2)):
        return value1.keys() == value2.keys() and all(
            _check_equality(value, value2[key], rtol, active) for key, value in value1.items()
        )
    unmatched = list(value2.items())
    for key, value in value1.items():
        for index, (other_key, other_value) in enumerate(unmatched):
            if _check_equality(key, other_key, 0, active) and _check_equality(
                value, other_value, rtol, active
            ):
                unmatched.pop(index)
                break
        else:
            return False
    return True


def get_element_from_number_and_weight(z: float, a: float) -> Isotope | Element:
    """Find the isotope nearest mass ``a`` among the isotopes of atomic number ``z``."""
    if elements is None:
        msg = "Could not import periodictable"
        raise ImportError(msg)
    try:
        element = elements[int(z)]
    except (KeyError, ValueError, OverflowError) as exc:
        msg = f"Could not find a matching element for A = {a} and Z = {z}"
        raise ValueError(msg) from exc
    elm = min(element, key=lambda isotope: abs(a - isotope.mass), default=None)
    if elm is None or not np.isfinite(a):
        msg = f"Could not find a matching element for A = {a} and Z = {z}"
        raise ValueError(msg)
    if abs(elm.element.mass - elm.mass) < 0.3:
        return elm.element
    return elm
