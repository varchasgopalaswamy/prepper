# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from loguru import logger

logger.add(sys.stdout, backtrace=True, diagnose=True, colorize=True)

# Make h5py write groups in order
import h5py

h5py.get_config().track_order = True


class H5StoreException(Exception):
    "An exception for when the HDF5 store does not meet spec"


from .caching import cached_property, local_cache
from .exportable import ExportableClassMixin
from .io_handlers import saveable_class
