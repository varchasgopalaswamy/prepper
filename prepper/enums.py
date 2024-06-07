from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from enum import Enum, unique

    from aenum import extend_enum
else:
    from aenum import Enum, extend_enum, unique

__all__ = ["H5StoreTypes", "add_enum_item"]


@unique
class H5StoreTypes(Enum):
    Sequence = 1
    Dictionary = 2
    ClassConstructor = 3
    PythonClass = 4
    HDF5Dataset = 5
    HDF5Group = 6
    FunctionCache = 7
    Null = 8
    Enumerator = 9
    DimensionalNDArray = 10
    XArrayDataset = 11
    PeriodicTableElement = 12
    ArViz = 13


def add_enum_item(name: str) -> None:
    extend_enum(H5StoreTypes, name, len(H5StoreTypes) + 1)
