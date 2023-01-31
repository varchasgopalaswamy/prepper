# -*- coding: utf-8 -*-
from __future__ import annotations

from aenum import auto, Enum, extend_enum, unique

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


N_PREPPER_ENUMS = 9


def add_enum_item(name: str):
    global N_PREPPER_ENUMS
    N_PREPPER_ENUMS += 1
    extend_enum(H5StoreTypes, name, N_PREPPER_ENUMS)
