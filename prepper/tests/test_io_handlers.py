from enum import Enum, IntFlag
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest

from prepper import ExportableClassMixin, H5StoreException, saveable_class
from prepper import io_handlers as io
from prepper.config import get_settings
from prepper.enums import H5StoreTypes


@saveable_class("1")
class Box(ExportableClassMixin):
    def __init__(self, payload):
        self.payload = payload


class Color(Enum):
    RED = "red"


class Permission(IntFlag):
    READ = 1
    WRITE = 2


def roundtrip(value, tmp_path):
    destination = tmp_path / "value.h5"
    Box(value).to_hdf5(destination)
    return Box.from_hdf5(destination).payload


@pytest.mark.parametrize("value", (None, [None], {"missing": None}, (None,)))
def test_null_values_roundtrip(value, tmp_path):
    assert roundtrip(value, tmp_path) == value


def test_dictionary_keys_and_container_kinds_roundtrip(tmp_path):
    value = {
        "a/b": 1,
        "": 2,
        ".": 3,
        "..": 4,
        1: "integer",
        "1": "string",
        ("tuple", 2): {"nested/slash": [1, "different types", None]},
    }
    restored = roundtrip(value, tmp_path)
    assert restored == value
    assert list(restored) == list(value)
    assert isinstance(restored[("tuple", 2)]["nested/slash"], list)


@pytest.mark.parametrize(
    "value", ((), [], (1, 2), [1, 2], (1, "a"), [1, "a"], {1, 2}, frozenset({1, 2}))
)
def test_sequence_type_is_preserved(value, tmp_path):
    restored = roundtrip(value, tmp_path)
    assert type(restored) is type(value)
    assert restored == value


@pytest.mark.parametrize(
    "value",
    (
        np.array(3.5),
        np.array([1e-50, 1e50]),
        np.array([True, False]),
        np.array(["hello", "λ"]),
        np.array([b"\xff", b"\x00"], dtype="S1"),
        np.array([1, 2], dtype="datetime64[ns]"),
        np.array([(1, 2.5)], dtype=[("a", "i4"), ("b", "f8")]),
        np.bool_(1),
        np.float32(3.5),
        b"\x00\xffraw bytes",
    ),
)
def test_native_dtypes_shapes_and_bytes_are_preserved(value, tmp_path):
    restored = roundtrip(value, tmp_path)
    assert type(restored) is type(value)
    np.testing.assert_array_equal(restored, value)
    if isinstance(value, np.ndarray | np.generic):
        assert restored.dtype == value.dtype
        assert restored.shape == value.shape


@pytest.mark.parametrize("member", (Color.RED, Permission(0), Permission.READ | Permission.WRITE))
def test_enum_and_flag_members_roundtrip(member, tmp_path):
    assert roundtrip(member, tmp_path) is member


def test_metadata_arrays_none_and_literal_sentinel(tmp_path):
    with h5py.File(tmp_path / "metadata.h5", "w") as handle:
        io.write_h5_attr(handle, "array", np.array([1, 2]))
        np.testing.assert_array_equal(io.read_h5_attr(handle, "array"), [1, 2])
        io.write_h5_attr(handle, "none", None)
        assert io.read_h5_attr(handle, "none") is None
        io.write_h5_attr(handle, "literal", "__python_None_sentinel__")
        assert io.read_h5_attr(handle, "literal") == "__python_None_sentinel__"
        handle.attrs["legacy_none"] = "__python_None_sentinel__"
        assert io.read_h5_attr(handle, "legacy_none") is None
        io.write_h5_attr(handle, "none", "ordinary")
        assert io.read_h5_attr(handle, "none") == "ordinary"
        with pytest.raises(H5StoreException, match="Could not write attribute"):
            io.write_h5_attr(handle, "unsupported", object())
        with pytest.raises(KeyError, match="Could not find attribute"):
            io.read_h5_attr(handle, "absent")


def test_deduplication_honors_tolerance_and_preserves_shape_dtype(tmp_path, monkeypatch):
    first = np.array([1.0, 2.0])
    second = np.array([1.0001, 2.0001])
    monkeypatch.setattr(get_settings(), "compare_rtol", 1e-3)
    path = tmp_path / "dedup.h5"
    values = [first, second, first.reshape(1, 2), first.astype("float32")]
    ExportableClassMixin._dump_h5_entry(path, "/payload", values, {})
    restored = ExportableClassMixin._load_h5_entry(path, "/payload")[1]
    np.testing.assert_array_equal(restored[1], first)
    assert restored[2].shape == (1, 2)
    assert restored[3].dtype == np.float32
    monkeypatch.setattr(get_settings(), "compare_rtol", 0)
    path.unlink()
    ExportableClassMixin._dump_h5_entry(path, "/payload", [first, second], {})
    restored = ExportableClassMixin._load_h5_entry(path, "/payload")[1]
    np.testing.assert_array_equal(restored[1], second)


def test_repeated_heterogeneous_sequence_links_load_independent_of_path(tmp_path):
    restored = roundtrip({"first": [1, "two"], "second": [1, "two"]}, tmp_path)
    assert restored == {"first": [1, "two"], "second": [1, "two"]}


def test_recursive_objects_raise_clear_errors_without_publishing(tmp_path):
    cycle = []
    cycle.append(cycle)
    destination = tmp_path / "value.h5"
    with pytest.raises(H5StoreException, match="Cyclic object references"):
        Box(cycle).to_hdf5(destination)
    assert not destination.exists()
    box = Box(None)
    box._constructor_call = ((box,), {})
    with pytest.raises(H5StoreException, match="Cyclic object references"):
        box.to_hdf5(destination)
    assert not destination.exists()


def test_malformed_dictionary_and_sequence_are_rejected(tmp_path):
    destination = tmp_path / "malformed.h5"
    with h5py.File(destination, "w") as handle:
        dictionary = handle.create_group("dictionary")
        dictionary.attrs["type"] = "Dictionary"
        dictionary.create_group("bad_item")
        sequence = handle.create_group("sequence")
        sequence.attrs.update(type="Sequence", python_kind="list", sequence_length=1)
        sequence.create_group("unexpected")
        with pytest.raises(H5StoreException, match="Malformed dictionary"):
            ExportableClassMixin._load_h5_entry(handle, "/dictionary")
        with pytest.raises(H5StoreException, match="Malformed sequence"):
            ExportableClassMixin._load_h5_entry(handle, "/sequence")


def test_cyclic_file_links_raise_clear_error(tmp_path):
    with h5py.File(tmp_path / "cycle.h5", "w") as handle:
        group = handle.create_group("cycle")
        group.attrs.update(type="Sequence", python_kind="list", sequence_length=1)
        group["item_1"] = h5py.SoftLink("/cycle")
        with pytest.raises(H5StoreException, match="Cyclic HDF5 references"):
            ExportableClassMixin._load_h5_entry(handle, "/cycle")


def test_custom_handlers_continue_to_receive_paths(tmp_path, monkeypatch):
    class Custom:
        pass

    monkeypatch.setattr(io, "CUSTOM_H5_WRITERS", dict(io.CUSTOM_H5_WRITERS))
    monkeypatch.setattr(io, "CUSTOM_H5_LOADERS", dict(io.CUSTOM_H5_LOADERS))

    @io.register_writer(lambda value: isinstance(value, Custom))
    def plugin_writer(file, group, value, existing_groups):
        assert isinstance(file, Path)
        with h5py.File(file, "a") as handle:
            handle.create_dataset(group, data=42)
        return {"type": H5StoreTypes.ArViz.name}, existing_groups

    @io.register_loader(H5StoreTypes.ArViz)
    def plugin_loader(file, group):
        assert isinstance(file, Path)
        with h5py.File(file) as handle:
            return handle[group][()]

    assert roundtrip(Custom(), tmp_path) == 42


def test_hdf5_subgroup_contents_and_lifetime(tmp_path):
    with h5py.File(tmp_path / "source.h5", "w") as source:
        source.create_dataset("unrelated", data=1)
        selected = source.create_group("selected")
        selected.create_dataset("wanted", data=2)
        restored = roundtrip(selected, tmp_path)
    assert set(restored) == {"wanted"}
    assert restored["wanted"][()] == 2
    restored.file.close()


def test_hdf5_group_cleanup_in_subprocess(tmp_path):
    code = """
import h5py
from prepper.exportable import ExportableClassMixin
with h5py.File(__import__('sys').argv[1], 'w') as source:
    selected = source.create_group('selected')
    selected.create_dataset('value', data=7)
    ExportableClassMixin._dump_h5_entry(__import__('sys').argv[2], '/payload', selected, {})
out = ExportableClassMixin._load_h5_entry(__import__('sys').argv[2], '/payload')[1]
assert out['value'][()] == 7
"""
    result = subprocess.run(  # noqa: S603 -- The interpreter and inline program are fixed.
        [sys.executable, "-c", code, str(tmp_path / "source.h5"), str(tmp_path / "target.h5")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_unit_and_uncertainty_values_preserve_precision(tmp_path):
    pint = pytest.importorskip("pint")
    uncertainty = pytest.importorskip("auto_uncertainties")
    magnitude = np.array([1e-50, 1e50], dtype=np.float64)
    error = np.array([1e-100, 0], dtype=np.float64)
    value = pint.get_application_registry().Quantity(
        uncertainty.Uncertainty(magnitude, error), "m"
    )
    restored = roundtrip(value, tmp_path)
    assert str(restored.units) == str(value.units)
    assert restored.magnitude.value.dtype == np.float64
    np.testing.assert_array_equal(restored.magnitude.value, magnitude)
    np.testing.assert_array_equal(restored.magnitude.error, error)
    zero_error = roundtrip(uncertainty.Uncertainty(magnitude, np.zeros_like(magnitude)), tmp_path)
    assert isinstance(zero_error, uncertainty.Uncertainty)
    np.testing.assert_array_equal(zero_error.error, [0, 0])


def test_xarray_mixed_variables_preserve_dtype_metadata_and_open_input(tmp_path):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("h5netcdf")
    value = xr.Dataset(
        {
            "precise": ("x", np.array([1e-50, 1e50])),
            "integer": ("x", np.array([2**60 + 1, 2**60 + 2], dtype="int64")),
            "scalar": 3.5,
            "label": ("x", ["a", "b"]),
        },
        coords={"x": [1, 2]},
        attrs={"description": "roundtrip"},
    )
    value["precise"].attrs["unit"] = "m"
    closed = []
    value.set_close(lambda: closed.append(True))
    restored = roundtrip(value, tmp_path)
    xr.testing.assert_identical(restored, value)
    assert not closed


def test_array_dedup_index_prunes_different_magnitudes(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "compare_rtol", 1e-5)
    comparisons = 0
    original = io.check_equality

    def count_comparison(left, right):
        nonlocal comparisons
        comparisons += 1
        return original(left, right)

    monkeypatch.setattr(io, "check_equality", count_comparison)
    values = [np.array([index, -index], dtype=float) for index in range(100)]
    values.append(np.array([50.0001, -50.0001]))
    path = tmp_path / "indexed.h5"
    ExportableClassMixin._dump_h5_entry(path, "/payload", values, {})
    restored = ExportableClassMixin._load_h5_entry(path, "/payload")[1]
    np.testing.assert_array_equal(restored[-1], values[50])
    assert 0 < comparisons < 20


def test_array_magnitude_boundaries_keep_approximate_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "compare_rtol", 0.01)
    for dtype in (np.float32, np.float64, np.complex128):
        for magnitude in (1e-20, 1.0, 1e20):
            first = np.array([magnitude, -magnitude], dtype=dtype)
            if np.issubdtype(dtype, np.complexfloating):
                first *= 1 + 1j
            second = first * (1 + 0.0099)
            assert io.check_equality(first, second)
            path = tmp_path / f"{np.dtype(dtype).name}-{magnitude}.h5"
            ExportableClassMixin._dump_h5_entry(path, "/payload", [first, second], {})
            restored = ExportableClassMixin._load_h5_entry(path, "/payload")[1]
            np.testing.assert_array_equal(restored[1], first)


def test_reads_legacy_sequence_through_a_soft_link(tmp_path):
    with h5py.File(tmp_path / "legacy.h5", "w") as handle:
        entry = handle.create_group("first")
        entry.attrs["type"] = "Sequence"
        item = entry.create_dataset("first_1", data=7)
        item.attrs["type"] = "HDF5Dataset"
        handle["second"] = h5py.SoftLink("/first")
        assert ExportableClassMixin._load_h5_entry(handle, "/second")[1] == [7]


def test_flag_zero_and_composite_load_in_a_cold_process(tmp_path):
    (tmp_path / "cold_flag_models.py").write_text(
        """from enum import IntFlag
from prepper import ExportableClassMixin, saveable_class

class Permission(IntFlag):
    READ = 1
    WRITE = 2

@saveable_class('1')
class Example(ExportableClassMixin):
    def __init__(self, payload):
        self.payload = payload
"""
    )
    programs = (
        """from pathlib import Path
from cold_flag_models import Example, Permission
Example([Permission(0), Permission.READ | Permission.WRITE]).to_hdf5(Path('flags.h5'))
""",
        """from pathlib import Path
from cold_flag_models import Example, Permission
assert 0 not in Permission._value2member_map_
assert 3 not in Permission._value2member_map_
restored = Example.from_hdf5(Path('flags.h5')).payload
assert type(restored[0]) is Permission and restored[0].value == 0
assert type(restored[1]) is Permission and restored[1].value == 3
""",
    )
    for code in programs:
        result = subprocess.run(  # noqa: S603 -- Fixed programs use the active Python interpreter.
            [sys.executable, "-B", "-c", code],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(("symbol", "mass_number"), (("C", 12), ("Fe", 56), ("O", 16)))
def test_periodictable_natural_elements_and_common_isotopes_are_distinct(
    symbol, mass_number, tmp_path
):
    pt = pytest.importorskip("periodictable")
    element = getattr(pt, symbol)
    isotope = element[mass_number]
    restored = roundtrip({"natural": element, "isotope": isotope}, tmp_path)
    assert restored["natural"] is element
    assert restored["isotope"] is isotope
    assert type(restored["natural"]) is pt.core.Element
    assert type(restored["isotope"]) is pt.core.Isotope
    assert restored["natural"].mass == element.mass
    assert restored["isotope"].mass == isotope.mass


@pytest.mark.parametrize(("symbol", "mass_number"), (("C", None), ("H", 2)))
def test_periodictable_legacy_entries_without_kind_still_load(symbol, mass_number, tmp_path):
    pt = pytest.importorskip("periodictable")
    value = getattr(pt, symbol)
    if mass_number is not None:
        value = value[mass_number]
    with h5py.File(tmp_path / "legacy_element.h5", "w") as handle:
        entry = handle.create_group("value")
        entry.attrs["type"] = H5StoreTypes.PeriodicTableElement.name
        entry.create_dataset("element_A", data=value.mass)
        entry.create_dataset("element_Z", data=value.number)
        assert ExportableClassMixin._load_h5_entry(handle, "/value")[1] is value
