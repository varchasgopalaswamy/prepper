from dataclasses import field
import datetime as dt

import h5py
import numpy as np
import pytest

from prepper import ExportableClassMixin, H5StoreException, saveable_class
from prepper.enums import H5StoreTypes

eqx = pytest.importorskip("equinox")
jnp = pytest.importorskip("jax.numpy")


class Affine(eqx.Module):
    weight: object
    bias: float
    label: str = eqx.field(static=True, default="affine")
    _private: str = "default"
    derived: float = field(init=False)

    def __post_init__(self):
        self.derived = self.bias * 2

    def __call__(self, value):
        return self.weight @ value + self.bias


class Stack(eqx.Module):
    layers: tuple[Affine, ...]
    metadata: dict


class PrivateMeasurement(eqx.Module):
    _value: object
    _unit: str = eqx.field(static=True)

    @property
    def value(self):
        import pint

        return pint.get_application_registry().Quantity(self._value, self._unit)


class PrivateCollection(eqx.Module):
    _items: list[str] = field(default_factory=list)


class PrivateArrays(eqx.Module):
    _value: object
    _error: object
    _metadata: dict


@saveable_class("1")
class ModelBox(ExportableClassMixin):
    def __init__(self, model):
        self.model = model


def roundtrip(value, tmp_path):
    destination = tmp_path / "model.h5"
    ModelBox(value).to_hdf5(destination)
    return ModelBox.from_hdf5(destination).model


def test_equinox_writer_records_class_and_serializable_fields(tmp_path):
    value = Affine(np.array([[1, 2]], dtype=np.float32), 0.5)
    destination = tmp_path / "module.h5"
    ExportableClassMixin._dump_h5_entry(destination, "/model", value, {})
    with h5py.File(destination) as handle:
        entry = handle["model"]
        assert entry.attrs["type"] == H5StoreTypes.Equinox.name
        assert dt.datetime.fromisoformat(entry.attrs["timestamp"]).tzinfo is not None
        assert set(entry) == {"weight", "bias", "label", "_private"}
        assert entry.attrs["module_class"] == f"{__name__}.Affine"


def test_equinox_numpy_fields_and_static_metadata_roundtrip(tmp_path):
    value = Affine(np.array([[1, -2], [3, 4]], dtype=np.float32), 0.5, label="λ")
    restored = roundtrip(value, tmp_path)
    assert type(restored) is Affine
    assert type(restored.weight) is np.ndarray
    assert restored.weight.dtype == value.weight.dtype
    assert restored.weight.shape == value.weight.shape
    np.testing.assert_array_equal(restored.weight, value.weight)
    assert restored.bias == value.bias
    assert restored.label == value.label
    assert restored.derived == value.derived
    inputs = np.array([2, -1], dtype=np.float32)
    np.testing.assert_array_equal(restored(inputs), value(inputs))


def test_equinox_nested_modules_and_containers_roundtrip(tmp_path):
    value = Stack(
        (Affine(np.array([[1, 2]], dtype=np.float32), 0.5), Affine(np.array(3), -1)),
        {"description": "nested", "optional": None, "dimensions": [2, 1]},
    )
    restored = roundtrip(value, tmp_path)
    assert type(restored) is Stack
    assert type(restored.layers) is tuple
    assert len(restored.layers) == 2
    for actual, expected in zip(restored.layers, value.layers, strict=True):
        assert type(actual) is Affine
        np.testing.assert_array_equal(actual.weight, expected.weight)
        assert actual.bias == expected.bias
    assert restored.metadata == value.metadata


@pytest.mark.parametrize("shape", ((), (0,), (2, 3)))
def test_equinox_jax_array_fields_load_as_numpy_with_same_shape_dtype_and_values(shape, tmp_path):
    weight = jnp.ones(shape, dtype=jnp.float32)
    restored = roundtrip(Affine(weight, 0.5), tmp_path)
    assert type(restored.weight) is np.ndarray
    assert restored.weight.shape == weight.shape
    assert restored.weight.dtype == weight.dtype
    np.testing.assert_array_equal(restored.weight, weight)


def test_jax_arrays_in_nested_containers_load_as_numpy(tmp_path):
    weight = jnp.array([1, -2, 3], dtype=jnp.int32)
    restored = roundtrip({"arrays": [weight, weight]}, tmp_path)
    for actual in restored["arrays"]:
        assert type(actual) is np.ndarray
        assert actual.dtype == weight.dtype
        np.testing.assert_array_equal(actual, weight)


def test_equinox_private_field_preserves_saved_value(tmp_path):
    value = Affine(np.array([1.0, 2.0]), 0.5, _private="saved private value")

    restored = roundtrip(value, tmp_path)

    assert restored._private == "saved private value"
    assert restored.derived == value.derived


def test_equinox_required_private_fields_keep_computed_quantity_property(tmp_path):
    pint = pytest.importorskip("pint")
    value = PrivateMeasurement(np.array([1.0, 2.0]), "keV")

    restored = roundtrip(value, tmp_path)

    assert type(restored) is PrivateMeasurement
    np.testing.assert_array_equal(restored._value, value._value)
    assert restored._unit == "keV"
    assert restored.value.units == pint.get_application_registry().keV
    np.testing.assert_array_equal(restored.value.magnitude, value.value.magnitude)


def test_equinox_private_arrays_and_nested_metadata_retain_exact_values(tmp_path):
    values = np.array([1.0, 2.0])
    errors = values + 1e-12
    value = PrivateArrays(values, errors, {"original": values.copy(), "changed": errors.copy()})

    restored = roundtrip(value, tmp_path)

    np.testing.assert_array_equal(restored._value, values)
    np.testing.assert_array_equal(restored._error, errors)
    np.testing.assert_array_equal(restored._metadata["original"], values)
    np.testing.assert_array_equal(restored._metadata["changed"], errors)


def test_equinox_loader_uses_old_file_defaults_and_recomputes_non_init_fields(tmp_path):
    # Build a valid file independently of the writer to exercise the loader alone.
    with h5py.File(tmp_path / "module.h5", "w") as handle:
        entry = handle.create_group("model")
        entry.attrs.update(type=H5StoreTypes.Equinox.name, module_class=f"{__name__}.Affine")
        for name, value in {"weight": np.array([1, 2]), "bias": 3.5, "label": "saved"}.items():
            ExportableClassMixin._dump_h5_entry(handle, f"/model/{name}", value, {})
        _, restored = ExportableClassMixin._load_h5_entry(handle, "/model")
    assert type(restored) is Affine
    np.testing.assert_array_equal(restored.weight, [1, 2])
    assert restored.bias == 3.5
    assert restored.label == "saved"
    assert restored._private == "default"
    assert restored.derived == 7


def test_equinox_loader_uses_default_factory_for_missing_private_field(tmp_path):
    with h5py.File(tmp_path / "module.h5", "w") as handle:
        entry = handle.create_group("model")
        entry.attrs.update(
            type=H5StoreTypes.Equinox.name,
            module_class=f"{__name__}.PrivateCollection",
        )
        _, restored = ExportableClassMixin._load_h5_entry(handle, "/model")

    assert restored._items == []


def test_equinox_loader_rejects_missing_required_private_field(tmp_path):
    with h5py.File(tmp_path / "module.h5", "w") as handle:
        entry = handle.create_group("model")
        entry.attrs.update(
            type=H5StoreTypes.Equinox.name,
            module_class=f"{__name__}.PrivateMeasurement",
        )
        with pytest.raises(H5StoreException, match="Missing required Equinox field '_value'"):
            ExportableClassMixin._load_h5_entry(handle, "/model")


@pytest.mark.parametrize("class_path", (None, "builtins.dict", f"{__name__}.MissingModule"))
def test_equinox_loader_rejects_missing_or_invalid_module_class(class_path, tmp_path):
    with h5py.File(tmp_path / "invalid.h5", "w") as handle:
        entry = handle.create_group("model")
        entry.attrs["type"] = H5StoreTypes.Equinox.name
        if class_path is not None:
            entry.attrs["module_class"] = class_path
        with pytest.raises(H5StoreException, match="not a valid Equinox Module"):
            ExportableClassMixin._load_h5_entry(handle, "/model")
