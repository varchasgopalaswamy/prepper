from __future__ import annotations

__protected__ = ["utils", "io_handlers", "enums", "utils"]


# <AUTOGEN_INIT>
import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "NATIVE_DTYPES",
    "ExportableClassMixin",
    "H5StoreException",
    "NotASaveableClass",
    "SimpleSaveableClass",
    "SimpleSaveableClass2",
    "break_key",
    "cached_property",
    "caching",
    "ensure_required_attributes",
    "enums",
    "exceptions",
    "exportable",
    "io_handlers",
    "local_cache",
    "make_cache_name",
    "req_class_attrs",
    "req_dataset_attrs",
    "req_none_attrs",
    "roundtrip",
    "saveable_class",
    "test_IO",
    "test_cached_property",
    "test_decorators",
    "test_local_cache",
    "test_saveable_class",
    "test_with_float_list",
    "test_with_floats",
    "test_with_heterogenous_list",
    "test_with_int_list",
    "test_with_ints",
    "test_with_str_list",
    "tests",
    "utils",
]
# </AUTOGEN_INIT>

import h5py

# Make h5py write groups in order
h5py.get_config().track_order = True
