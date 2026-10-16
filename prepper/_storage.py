"""Share open HDF5 handles throughout a serialization operation."""

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import h5py

if TYPE_CHECKING:
    from collections.abc import Generator

type FileSource = str | Path | h5py.File


@contextmanager
def open_hdf5(source: FileSource, mode: str = "r") -> Generator[h5py.File]:
    """Borrow an existing handle, or open and close a file owned by this call."""
    if isinstance(source, h5py.File):
        yield source
    else:
        with h5py.File(source, mode=mode, track_order=True) as handle:
            yield handle
