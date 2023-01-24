# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import TYPE_CHECKING

from aenum import auto, Enum, unique

__all__ = ["H5StoreTypes"]


@unique
class H5StoreTypes(Enum):
    Sequence = auto()
    Dictionary = auto()
    ClassConstructor = auto()
    PythonClass = auto()
    HDF5Dataset = auto()
    HDF5Group = auto()
    DimensionalNDArray = auto()
    XArrayDataset = auto()
    FunctionCache = auto()
    PeriodicTableElement = auto()
    Null = auto()
    Enumerator = auto()
