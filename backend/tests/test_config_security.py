"""Configuration tests that never expose or require a real database password."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def test_password_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_empty_password_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POSTGRES_PASSWORD", "")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secret_is_hidden_in_repr_and_url_is_encoded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "unit-test-secret@with space"
    monkeypatch.setenv("POSTGRES_PASSWORD", secret)

    settings = Settings(_env_file=None)

    assert secret not in repr(settings)
    assert settings.postgres_password.get_secret_value() == secret
    assert "unit-test-secret%40with%20space" in settings.database_url
