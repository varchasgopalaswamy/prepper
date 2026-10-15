"""Store Python objects in HDF5 files."""

import h5py

__public__ = ("caching", "exceptions", "exportable")
__protected__ = ("enums", "io_handlers", "utils")

# Make h5py write groups in order.
h5py.get_config().track_order = True

# <geninit>
lazy from . import caching
lazy from . import enums
lazy from . import exceptions
lazy from . import exportable
lazy from . import io_handlers
lazy from . import utils
lazy from .caching import break_key
lazy from .caching import cached_property
lazy from .caching import local_cache
lazy from .caching import make_cache_name
lazy from .exceptions import H5StoreException
lazy from .exportable import ExportableClassMixin
lazy from .exportable import saveable_class

__all__ = (
    "ExportableClassMixin",
    "H5StoreException",
    "break_key",
    "cached_property",
    "caching",
    "enums",
    "exceptions",
    "exportable",
    "io_handlers",
    "local_cache",
    "make_cache_name",
    "saveable_class",
    "utils",
)
# </geninit>
