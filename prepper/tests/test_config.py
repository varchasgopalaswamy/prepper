import math

import pytest

from prepper.config import Settings, configure, get_settings


@pytest.fixture(autouse=True)
def restore_settings():
    previous = get_settings().model_dump()
    yield
    configure(**previous)


def test_default_settings(monkeypatch):
    monkeypatch.delenv("PREPPER_COMPARE_RTOL", raising=False)
    assert Settings().compare_rtol == 1e-5


def test_environment_and_explicit_override(monkeypatch):
    monkeypatch.setenv("PREPPER_COMPARE_RTOL", "0.002")
    assert Settings().compare_rtol == 0.002
    assert Settings(compare_rtol=0).compare_rtol == 0
    monkeypatch.setenv("PREPPER_COMPARE_RTOL", "invalid")
    assert Settings(compare_rtol=0.25).compare_rtol == 0.25
    with pytest.raises(ValueError, match="compare_rtol"):
        Settings()


@pytest.mark.parametrize("value", (-1, math.inf, -math.inf, math.nan, "invalid"))
def test_invalid_configuration_is_atomic(value):
    configure(compare_rtol=0.02)
    shared = get_settings()
    with pytest.raises(ValueError, match="compare_rtol"):
        configure(compare_rtol=value)
    assert get_settings() is shared
    assert shared.compare_rtol == 0.02


def test_unknown_setting_does_not_partially_update():
    configure(compare_rtol=0.02)
    with pytest.raises(ValueError, match="Unknown settings: unknown"):
        configure(compare_rtol=0.2, unknown=True)
    assert get_settings().compare_rtol == 0.02


def test_runtime_configuration_preserves_identity_and_ignores_later_environment(monkeypatch):
    shared = get_settings()
    configure(compare_rtol=0.03)
    monkeypatch.setenv("PREPPER_COMPARE_RTOL", "invalid")
    configure()
    assert get_settings() is shared
    assert shared.compare_rtol == 0.03
    configure(compare_rtol=0)
    assert shared.compare_rtol == 0


def test_direct_assignment_is_validated():
    shared = get_settings()
    shared.compare_rtol = 0.001
    with pytest.raises(ValueError, match="compare_rtol"):
        shared.compare_rtol = -1
    assert shared.compare_rtol == 0.001
