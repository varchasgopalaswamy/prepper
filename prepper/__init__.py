# -*- coding: utf-8 -*-
from __future__ import annotations

import h5py

__all__ = [
    "cached_property",
    "local_cache",
    "ExportableClassMixin",
    "saveable_class",
]


# Make h5py write groups in order
h5py.get_config().track_order = True


class H5StoreException(Exception):
    "An exception for when the HDF5 store does not meet spec"


from .caching import cached_property, local_cache
from .exportable import ExportableClassMixin, saveable_class
