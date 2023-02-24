from tkinter import E
import numpy as np

def check_equality(a,b):
    """
    Check if two objects are equal
    """
    from prepper.exportable import ExportableClassMixin

    if type(a) != type(b):
        return False

    # Just try to do a comparison
    try:
        return bool(a == b)
    except Exception:
        # Maybe it's a numpy array
        # check if the dimensions are compatible
        if np.ndim(a) != np.ndim(b):
            return False
        if hasattr(a,'units') and not a.is_compatible_with(b):
            return False
        # do a numpy comparison
        try:
            return np.allclose(a,b)
        except Exception:
            raise ValueError(f"Cannot compare {a} and {b} of type {type(a)} and {type(b)}")
