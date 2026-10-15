from copy import copy, deepcopy
from pathlib import Path

import h5py
import numpy as np
import pytest

from prepper import (
    ExportableClassMixin,
    H5StoreException,
    cached_property,
    local_cache,
    saveable_class,
)


@saveable_class("1", attributes=["value"], functions=["calculate"])
class Lifecycle(ExportableClassMixin):
    constructions = 0

    def __init__(self, x=2):
        type(self).constructions += 1
        self.x = x

    @cached_property
    def value(self):
        return self.x

    @local_cache
    def calculate(self, factor=1):
        return self.x * factor


@saveable_class("1")
class SignatureKinds(ExportableClassMixin):
    def __init__(self, x, /, *values, scale=2, **options):
        self.x, self.values, self.scale, self.options = x, values, scale, options


@saveable_class("1")
class Payload(ExportableClassMixin):
    def __init__(self, data):
        self.data = data


@saveable_class("1", attributes=["value"], functions=["calculate"])
class InheritanceParent(ExportableClassMixin):
    def __init__(self):
        self.property_calls = 0
        self.method_calls = 0

    @cached_property
    def value(self):
        self.property_calls += 1
        return np.array([1, 2])

    @local_cache
    def calculate(self, argument):
        self.method_calls += 1
        return argument


@saveable_class("1")
class InheritanceChild(InheritanceParent):
    @cached_property
    def value(self):
        self.property_calls += 1
        return super().value + 1

    @local_cache
    def calculate(self, argument):
        self.method_calls += 1
        return super().calculate(argument) + 1


@saveable_class("1", attributes=["left"])
class AliasedPropertyParent(ExportableClassMixin):
    def __init__(self):
        self.calls = 0

    @cached_property
    def left(self):
        self.calls += 1
        return np.array([1, 2])


@saveable_class("1", attributes=["right"])
class AliasedPropertyChild(AliasedPropertyParent):
    @cached_property
    def right(self):
        self.calls += 1
        return np.array([1, 2])


@saveable_class("1", attributes=["ConstructorNamedProperty"])
class ConstructorNamedProperty(ExportableClassMixin):
    def __init__(self):
        self.calls = 0

    @cached_property
    def ConstructorNamedProperty(self):
        self.calls += 1
        return 7


def test_constructor_runs_once_and_defaults_are_canonical(tmp_path):
    Lifecycle.constructions = 0
    implicit = Lifecycle()
    explicit = Lifecycle(x=2)
    assert Lifecycle.constructions == 2
    assert implicit == explicit
    assert explicit == implicit
    with pytest.raises(TypeError, match="unhashable"):
        hash(implicit)
    path = tmp_path / "lifecycle.h5"
    implicit.to_hdf5(path)
    restored = Lifecycle.from_hdf5(path)
    assert Lifecycle.constructions == 3
    assert restored.initialized_from_file
    assert restored.x == 2


def test_persisted_defaults_survive_signature_changes(tmp_path, monkeypatch):
    path = tmp_path / "defaults.h5"
    Lifecycle().to_hdf5(path)
    monkeypatch.setattr(Lifecycle.__init__, "__defaults__", (10,))
    restored = Lifecycle.from_hdf5(path)
    assert restored.x == 2


def test_invalid_metadata_rejected_before_construction(tmp_path, monkeypatch):
    path = tmp_path / "invalid.h5"
    Lifecycle().to_hdf5(path)
    original = Lifecycle.constructions
    with h5py.File(path, "a") as file:
        file.attrs["class"] = "WrongClass"
    with pytest.raises(H5StoreException, match="Stored class"):
        Lifecycle.from_hdf5(path)
    assert Lifecycle.constructions == original
    with h5py.File(path, "a") as file:
        file.attrs["class"] = "Lifecycle"
    monkeypatch.setattr(Lifecycle, "api_version", "2")
    with pytest.raises(H5StoreException, match="API version"):
        Lifecycle.from_hdf5(path)
    assert Lifecycle.constructions == original


def test_all_constructor_argument_kinds_roundtrip(tmp_path):
    path = tmp_path / "signature.h5"
    item = SignatureKinds(1, "a", "b", scale=3, alpha=4, beta=5)
    item.to_hdf5(path)
    restored = SignatureKinds.from_hdf5(path)
    assert (restored.x, restored.values, restored.scale, restored.options) == (
        1,
        ("a", "b"),
        3,
        {"alpha": 4, "beta": 5},
    )
    assert item == restored


def test_equality_includes_populated_properties_and_function_caches():
    left, right = Lifecycle(), Lifecycle()
    assert left == right
    _ = left.value
    assert left != right
    _ = right.value
    assert left == right
    left.calculate(3)
    assert left != right
    right.calculate(3)
    assert left == right
    left.__dict__["__cache_Lifecycle.calculate__"].clear()
    assert left != right
    assert left != object()


def test_copies_preserve_state_shared_references_and_cycles():
    shared = []
    item = Payload([shared, shared])
    item.extra = 77
    item.self_reference = item
    shallow = copy(item)
    assert shallow.data is item.data
    assert shallow.extra == 77
    deep = deepcopy(item)
    assert deep is not item
    assert deep.data[0] is deep.data[1]
    assert deep.data[0] is not shared
    assert deep.self_reference is deep
    assert deep.extra == 77


def test_copy_preserves_slots():
    class Slotted(ExportableClassMixin):
        __slots__ = ("__private", "slot")

        def __init__(self):
            self.slot = []
            self.__private = 3

    item = Slotted()
    assert copy(item).slot is item.slot
    assert deepcopy(item).slot is not item.slot
    assert deepcopy(item)._Slotted__private == 3


def test_failed_replacement_preserves_destination_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "existing.h5"
    original = b"existing contents"
    path.write_bytes(original)

    def fail_replace(source, destination):
        assert Path(source).parent == path.parent
        assert Path(destination) == path
        msg = "simulated replacement failure"
        raise OSError(msg)

    monkeypatch.setattr("prepper.exportable.Path.replace", fail_replace)
    with pytest.raises(OSError, match="replacement failure"):
        Lifecycle().to_hdf5(path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_failed_serialization_preserves_destination(tmp_path):
    path = tmp_path / "existing.h5"
    path.write_bytes(b"original")
    with pytest.raises((H5StoreException, ValueError), match="support"):
        Payload(object()).to_hdf5(path)
    assert path.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [path]


def test_missing_paths_and_unknown_types(tmp_path):
    with pytest.raises(FileNotFoundError):
        Lifecycle.from_hdf5(tmp_path / "missing.h5")
    with pytest.raises(FileNotFoundError):
        Lifecycle().to_hdf5(tmp_path / "missing" / "file.h5")
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as file:
        file.attrs["type"] = "unknown"
    with pytest.raises(H5StoreException, match="Unknown HDF5"):
        Lifecycle.from_hdf5(path)


def test_reads_legacy_keyword_constructor(tmp_path):
    path = tmp_path / "legacy.h5"
    with h5py.File(path, "w") as file:
        file.attrs.update({"type": "PythonClass", "class": "Lifecycle", "api_version": "1"})
        constructor = file.create_group("Lifecycle")
        constructor.attrs["type"] = "ClassConstructor"
        argument = constructor.create_dataset("x", data=9)
        argument.attrs["type"] = "HDF5Dataset"
    assert Lifecycle.from_hdf5(path).x == 9


def test_large_arrays_and_cached_results_remain_exact(tmp_path):
    path = tmp_path / "array.h5"
    item = Lifecycle(np.array([1e-50, 1e50], dtype=np.float64))
    _ = item.value
    _ = item.calculate(2)
    item.to_hdf5(path)
    restored = Lifecycle.from_hdf5(path)
    np.testing.assert_array_equal(restored.x, item.x)
    np.testing.assert_array_equal(restored.value, item.value)
    np.testing.assert_array_equal(restored.calculate(2), item.calculate(2))
    assert restored.value.dtype == np.float64


def test_reuses_open_file_handles(tmp_path, monkeypatch):
    calls = 0
    original = h5py.File.__init__

    def count_file(self, *args, **kwargs):
        nonlocal calls
        if args and isinstance(args[0], (str, Path)):
            calls += 1
        original(self, *args, **kwargs)

    monkeypatch.setattr(h5py.File, "__init__", count_file)
    path = tmp_path / "opens.h5"
    item = Payload({f"k{i}": np.array([i, -i]) for i in range(100)})
    item.to_hdf5(path)
    assert calls == 1
    restored = Payload.from_hdf5(path)
    assert calls == 2
    np.testing.assert_array_equal(restored.data["k99"], [99, -99])


def test_overridden_inherited_members_preserve_both_populated_caches(tmp_path):
    item = InheritanceChild()
    np.testing.assert_array_equal(item.value, [2, 3])
    assert item.calculate(3) == 4
    assert item.property_calls == 2
    assert item.method_calls == 2
    assert InheritanceChild._exportable_attributes == [
        "InheritanceParent.value",
        "InheritanceChild.value",
    ]
    assert InheritanceChild._exportable_functions == [
        "InheritanceParent.calculate",
        "InheritanceChild.calculate",
    ]
    path = tmp_path / "inherited-caches.h5"
    item.to_hdf5(path)
    with h5py.File(path) as handle:
        for symbol in (
            "InheritanceParent.value",
            "InheritanceChild.value",
            "InheritanceParent.calculate",
            "InheritanceChild.calculate",
        ):
            assert symbol in handle
    restored = InheritanceChild.from_hdf5(path)
    np.testing.assert_array_equal(restored.value, [2, 3])
    np.testing.assert_array_equal(super(InheritanceChild, restored).value, [1, 2])
    assert restored.calculate(3) == 4
    assert super(InheritanceChild, restored).calculate(3) == 3
    assert restored.property_calls == 0
    assert restored.method_calls == 0
    assert item == restored


def test_deduplicated_inherited_properties_keep_separate_binding_metadata(tmp_path):
    item = AliasedPropertyChild()
    np.testing.assert_array_equal(item.left, item.right)
    path = tmp_path / "aliased-inherited-properties.h5"
    item.to_hdf5(path)
    with h5py.File(path) as handle:
        assert isinstance(handle.get("right", getlink=True), h5py.SoftLink)
    restored = AliasedPropertyChild.from_hdf5(path)
    assert "AliasedPropertyParent.left" in restored.__dict__
    assert "AliasedPropertyChild.right" in restored.__dict__
    assert "AliasedPropertyChild.left" not in restored.__dict__
    np.testing.assert_array_equal(restored.left, restored.right)
    assert restored.calls == 0
    assert item == restored


def test_member_named_after_class_does_not_overwrite_constructor(tmp_path):
    item = ConstructorNamedProperty()
    assert item.ConstructorNamedProperty == 7
    path = tmp_path / "reserved-constructor-name.h5"
    item.to_hdf5(path)
    with h5py.File(path) as handle:
        assert "ConstructorNamedProperty" in handle
        assert "ConstructorNamedProperty.ConstructorNamedProperty" in handle
    restored = ConstructorNamedProperty.from_hdf5(path)
    assert restored.ConstructorNamedProperty == 7
    assert restored.calls == 0


def test_reads_legacy_member_binding_attributes(tmp_path):
    path = tmp_path / "legacy-bindings.h5"
    item = Lifecycle()
    _ = item.value
    item.calculate(2)
    item.to_hdf5(path)
    with h5py.File(path, "a") as handle:
        del handle.attrs["__prepper_bindings__"]
        handle["value"].attrs["bound_class"] = "Lifecycle"
        del handle["calculate"].attrs["member_name"]
    restored = Lifecycle.from_hdf5(path)
    assert restored.__dict__["Lifecycle.value"] == 2
    assert len(restored.__dict__["__cache_Lifecycle.calculate__"]) == 1
    assert item == restored


def test_atomic_overwrite_preserves_file_permissions(tmp_path):
    path = tmp_path / "permissions.h5"
    path.write_bytes(b"old contents")
    path.chmod(0o664)
    Lifecycle().to_hdf5(path)
    assert path.stat().st_mode & 0o777 == 0o664
    assert Lifecycle.from_hdf5(path).x == 2


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (("module", "foreign_app", "Stored module"), ("qualname", "DifferentClass", "qualified name")),
)
def test_rejects_foreign_class_metadata_before_constructor(tmp_path, field, value, error):
    path = tmp_path / "foreign.h5"
    Lifecycle().to_hdf5(path)
    with h5py.File(path, "a") as file:
        file.attrs[field] = value
    before = Lifecycle.constructions
    with pytest.raises(H5StoreException, match=error):
        Lifecycle.from_hdf5(path)
    assert Lifecycle.constructions == before


@pytest.mark.parametrize("bindings", ("[invalid-json", '{"value": 42}', "[]"))
def test_rejects_malformed_member_bindings_before_constructor(tmp_path, bindings):
    path = tmp_path / "invalid-bindings.h5"
    Lifecycle().to_hdf5(path)
    with h5py.File(path, "a") as handle:
        handle.attrs["__prepper_bindings__"] = bindings
    before = Lifecycle.constructions
    with pytest.raises(H5StoreException, match="Invalid member bindings"):
        Lifecycle.from_hdf5(path)
    assert Lifecycle.constructions == before
