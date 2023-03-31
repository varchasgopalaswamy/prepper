# -*- coding: utf-8 -*-
from __future__ import annotations

import warnings

import loguru
import numpy as np

try:
    import xarray as xr
except ImportError:
    xr = None


def check_equality(value1, value2, log=False):
    """
    Check if two objects are equal
    """

    # Need a special check for xarray Datasets...
    if xr is not None:
        if isinstance(value1, (xr.Dataset)):
            return value1.identical(value2)
        elif isinstance(value2, (xr.Dataset)):
            return value2.identical(value1)

    # Just try to do a comparison
    try:
        same = value1 == value2
        same = bool(same)
        if not same:
            if log:
                loguru.logger.debug(
                    f"Values are different: {value1} and {value2}"
                )
        return same
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
            if hasattr(value1, "units") and not value1.is_compatible_with(
                value2
            ):
                if log:
                    loguru.logger.debug(
                        f"Units are different: {getattr(value1,'units','')} and {getattr(value2,'units','')}"
                    )
                return False
            # do a numpy comparison
            try:
                same = np.allclose(value1, value2)
            except Exception:
                same = all(value1 == value2)
            if not same:
                if log:
                    loguru.logger.debug(
                        f"Numpy check: values are different: {value1} and {value2}"
                    )
            return same
        except Exception as e:
            if not isinstance(value1, type(value2)):
                if log:
                    loguru.logger.debug(
                        f"Types are different: {type(value1)} and {type(value2)} for values {value1} and {value2}"
                    )
                return False

            raise ValueError(
                f"Cannot compare {value1} and {value2} of type {type(value1)} and {type(value2)}"
            ) from e


def get_element_from_number_and_weight(z: float, a: float):
    """

    :param z:
    :param a:

    """
    import periodictable

    elm = None
    mindist = np.inf
    for element in periodictable.elements:
        for iso in element:
            e_z = iso.number
            e_a = iso.mass
            if int(z) == int(e_z) and np.abs(a - e_a) < mindist:
                mindist = np.abs(a - e_a)
                elm = iso

    # If this is the base element, just return the base element
    if np.abs(elm.element.mass - elm.mass) < 0.3:
        elm = elm.element
    if elm is None:
        raise ValueError(
            f"Could not find a matching element for A = {a} and Z = {z}"
        )
    return elm
