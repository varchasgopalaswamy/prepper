# -*- coding: utf-8 -*-
from __future__ import annotations

from aenum import auto, Enum, extend_enum, unique

__all__ = ["H5StoreTypes", "add_enum_item"]


@unique
class H5StoreTypes(Enum):
    Sequence = auto()
    Dictionary = auto()
    ClassConstructor = auto()
    PythonClass = auto()
    HDF5Dataset = auto()
    HDF5Group = auto()
    FunctionCache = auto()
    Null = auto()
    Enumerator = auto()


def add_enum_item(name: str):
    extend_enum(H5StoreTypes, name, auto())
