__all__ = ["H5StoreException"]


class H5StoreException(Exception):  # noqa: N818 -- Established public exception name.
    """An exception for when the HDF5 store does not meet spec"""
