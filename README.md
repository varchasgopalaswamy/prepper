# Prepper

Store Python objects, populated cached properties, and method caches in HDF5.
Requires Python 3.14 or newer.

## Comparison settings

Set `PREPPER_COMPARE_RTOL` before importing Prepper to choose the relative tolerance
used for numeric equality and serialization deduplication:

```bash
export PREPPER_COMPARE_RTOL=1e-6
```

The default is `1e-5`. Set it to `0` for exact comparisons. Values must be finite
and nonnegative.
