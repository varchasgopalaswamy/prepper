import h5py
import pytest

from prepper._storage import open_hdf5


def interrupt(handle=None):
    if handle is not None:
        handle["value"] = 7
    msg = "interrupt"
    raise RuntimeError(msg)


def test_borrowed_file_remains_open_after_context_and_exception(tmp_path):
    path = tmp_path / "borrowed.h5"
    with h5py.File(path, "w") as owner:
        with open_hdf5(owner, "a") as borrowed:
            assert borrowed is owner
            borrowed["value"] = 3
        assert owner.id.valid
        with pytest.raises(RuntimeError, match="interrupt"), open_hdf5(owner):
            interrupt()
        assert owner.id.valid
        assert owner["value"][()] == 3
    assert not owner.id.valid


def test_owned_file_closes_after_context_and_exception(tmp_path):
    path = tmp_path / "owned.h5"
    with pytest.raises(RuntimeError, match="interrupt"), open_hdf5(path, "w") as owned:
        interrupt(owned)
    assert not owned.id.valid
    with open_hdf5(str(path)) as reopened:
        assert reopened["value"][()] == 7
    assert not reopened.id.valid
