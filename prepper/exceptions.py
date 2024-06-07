from __future__ import annotations

__all__ = ["H5StoreException"]


class H5StoreException(Exception):
    "An exception for when the HDF5 store does not meet spec"
