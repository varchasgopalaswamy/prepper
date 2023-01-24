
from aenum import Enum, unique, auto

__all__ = [
    'H5StoreTypes'
]

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

