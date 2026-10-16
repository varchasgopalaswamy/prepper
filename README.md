# Prepper

Store Python objects, populated cached properties, and method caches in HDF5.
Requires Python 3.14 (`>=3.14,<3.15`). Python 3.15 support is deferred until
compatible h5py wheels are available; see [h5py#2972](https://github.com/h5py/h5py/issues/2972).

## Comparison settings

Set `PREPPER_COMPARE_RTOL` before importing Prepper to choose the relative tolerance
used for numeric equality and serialization deduplication:

```bash
export PREPPER_COMPARE_RTOL=1e-6
```

The default is `1e-5`. Set it to `0` for exact comparisons. Values must be finite
and nonnegative.
