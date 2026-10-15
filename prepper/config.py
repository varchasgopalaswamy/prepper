"""Validated runtime configuration initialized from PREPPER_* variables."""

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "configure", "get_settings"]


class Settings(BaseSettings):
    """
    Configure comparisons from keyword arguments or PREPPER_* environment variables.

    Explicit arguments take precedence over environment values. ``compare_rtol``
    is a finite, nonnegative relative tolerance; zero requests exact comparisons.
    There is no absolute tolerance that grants extra slack to small magnitudes.
    """

    model_config = SettingsConfigDict(
        env_prefix="PREPPER_", extra="forbid", validate_assignment=True
    )

    compare_rtol: float = Field(default=1e-5, ge=0, allow_inf_nan=False)


settings = Settings()


def get_settings() -> Settings:
    """Return the shared settings used by subsequent library operations."""
    return settings


def configure(**changes: Any) -> None:
    """
    Validate all changes before updating the shared runtime settings.

    Later environment changes do not affect the shared settings. Unknown names
    and invalid values raise ``ValueError`` without changing existing settings.
    """
    unknown = changes.keys() - Settings.model_fields.keys()
    if unknown:
        msg = f"Unknown settings: {', '.join(sorted(unknown))}"
        raise ValueError(msg)
    candidate = Settings(**(settings.model_dump() | changes))
    for name in changes:
        setattr(settings, name, getattr(candidate, name))
