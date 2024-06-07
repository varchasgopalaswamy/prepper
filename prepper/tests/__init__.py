import lazy_loader


__getattr__, __dir__, __all__ = lazy_loader.attach_stub(__name__, __file__)

__all__ = [
    "NATIVE_DTYPES",
    "NotASaveableClass",
    "SimpleSaveableClass",
    "SimpleSaveableClass2",
    "ensure_required_attributes",
    "req_class_attrs",
    "req_dataset_attrs",
    "req_none_attrs",
    "roundtrip",
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
]
