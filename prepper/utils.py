# -*- coding: utf-8 -*-
from __future__ import annotations

import loguru
import numpy as np


def check_equality(value1, value2, log=False):
    """
    Check if two objects are equal
    """

    # Just try to do a comparison
    try:
        same = bool(value1 == value2)
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
