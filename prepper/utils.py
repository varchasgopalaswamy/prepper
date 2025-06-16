from __future__ import annotations

from typing import TYPE_CHECKING, Any

import loguru
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None

try:
    from periodictable import elements
except ImportError:
    elements = None

if TYPE_CHECKING:
    from periodictable.core import Element, Isotope

__all__ = ["check_equality", "get_element_from_number_and_weight"]


def check_equality(value1: Any, value2: Any, *, log: bool = False) -> bool:
    """
    Check if two objects are equal
    """

    # Need a special check for xarray Datasets...
    if xr is not None:
        # If both are xarray datasets, check to see if they are the same
        for xr_type in [xr.Dataset, xr.DataArray]:
            if isinstance(value1, xr_type) and isinstance(value2, xr_type):
                return value1.identical(value2)

    # Just try to do a comparison
    try:
        loguru.logger.disable("prepper")
        same = value1 == value2
        same = bool(same)
        loguru.logger.enable("prepper")

        if not same and log:
            loguru.logger.debug(f"Values are different: {value1} and {value2}")

    except Exception:
        # Maybe it's a numpy array
        # check if the dimensions are compatible
        try:
            if np.ndim(value1) != np.ndim(value2):
                if log:
                    loguru.logger.debug(
                        f"Dims are different: {np.ndim(value1)} and {np.ndim(value2)} for values {value1} and {value2}"
                    )
                return False
            if hasattr(value1, "units") and not value1.is_compatible_with(value2):
                if log:
                    loguru.logger.debug(
                        f"Units are different: {getattr(value1, 'units', '')} and {getattr(value2, 'units', '')}"
                    )
                return False
            # do a numpy comparison
            try:
                same = np.allclose(value1, value2)
            except Exception:
                same = all(value1 == value2)
            if not same and log:
                loguru.logger.debug(
                    f"Numpy check: values are different: {value1} and {value2}"
                )
        except Exception as e:
            if isinstance(value1, type(value2)):
                msg = f"Cannot compare {value1} and {value2} of type {type(value1)} and {type(value2)}"
                raise TypeError(msg) from e

            if log:
                loguru.logger.debug(
                    f"Types are different: {type(value1)} and {type(value2)} for values {value1} and {value2}"
                )
            return False

        else:
            return same
    else:
        return same


def get_element_from_number_and_weight(z: float, a: float) -> Isotope | Element:
    """
    This function takes in a float value 'z' representing the atomic number,
    and another float value 'a' representing the atomic mass, and returns
    an object of type ElementType.
    """

    # Initialize variables
    elm = None
    mindist = np.inf

    # Iterates over elements in the periodic table to
    # find the element that matches the atomic number and weight.
    if elements is None:
        msg = "Could not import periodictable"
        raise ImportError(msg)

    for element in elements:
        for iso in element:
            e_z = iso.number  # atomic number of the current element
            e_a = iso.mass  # atomic mass of the current element
            if int(z) == int(e_z) and np.abs(a - e_a) < mindist:
                # updates the element object if the difference in
                # mass between the requested and the found element
                # is less than the minimum distance.
                mindist = np.abs(a - e_a)
                elm = iso
    # If we didnt find an element, raise a ValueError exception
    if elm is None:
        msg = f"Could not find a matching element for A = {a} and Z = {z}"
        raise ValueError(msg)

    # If the found element is the base element, just return the base element
    if np.abs(elm.element.mass - elm.mass) < 0.3:
        elm = elm.element

    return elm
