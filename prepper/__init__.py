
from loguru import logger
import sys

logger.add(sys.stdout, backtrace=True, diagnose=True, colorize=True)

# Make h5py write groups in order
import h5py

h5py.get_config().track_order = True

class H5StoreException(Exception):
    "An exception for when the HDF5 store does not meet spec"

from .exportable import ExportableClassMixin
from .caching import local_cache, cached_property
from .io_handlers import saveable_class
