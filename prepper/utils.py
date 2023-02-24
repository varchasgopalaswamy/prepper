# -*- coding: utf-8 -*-
from __future__ import annotations

import loguru
import numpy as np


def check_equality(a, b, log=False):
    """
    Check if two objects are equal
    """
    from prepper.exportable import ExportableClassMixin

    # Just try to do a comparison
    try:
        same = bool(a == b)
        if not same:
            if log:
                loguru.logger.debug(f"Values are different: {a} and {b}")
        return same
    except Exception:
        # Maybe it's a numpy array
        # check if the dimensions are compatible
        try:
            if np.ndim(a) != np.ndim(b):
                if log:
                    loguru.logger.debug(
                        f"Dims are different: {np.ndim(a)} and {np.ndim(b)} for values {a} and {b}"
                    )
                return False
            if hasattr(a, "units") and not a.is_compatible_with(b):
                if log:
                    loguru.logger.debug(
                        f"Units are different: {getattr(a,'units','')} and {getattr(b,'units','')}"
                    )
                return False
            # do a numpy comparison
            try:
                same = np.allclose(a, b)
            except Exception:
                same = all(a == b)
            if not same:
                if log:
                    loguru.logger.debug(
                        f"Numpy check: values are different: {a} and {b}"
                    )
            return same
        except Exception:
            if type(a) != type(b):
                if log:
                    loguru.logger.debug(
                        f"Types are different: {type(a)} and {type(b)} for values {a} and {b}"
                    )
                return False
            else:
                raise ValueError(
                    f"Cannot compare {a} and {b} of type {type(a)} and {type(b)}"
                )
