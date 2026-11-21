"""Unit tests for the transaction helper used by derived-variant creation."""

from __future__ import annotations

from contextlib import contextmanager

import pytest

from app.core import database


class FakeConnection:
    def __init__(self) -> None:
        self.autocommit = True
        self.commit_calls = 0
        self.rollback_calls = 0

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


@pytest.fixture
def fake_connection(monkeypatch: pytest.MonkeyPatch):
    connection = FakeConnection()

    @contextmanager
    def fake_connection_cm():
        yield connection

    monkeypatch.setattr(database, "connection", fake_connection_cm)
    return connection


def test_transaction_commits_and_restores_autocommit(fake_connection: FakeConnection) -> None:
    with database.transaction() as conn:
        assert conn is fake_connection
        assert conn.autocommit is False

    assert fake_connection.commit_calls == 1
    assert fake_connection.rollback_calls == 0
    assert fake_connection.autocommit is True


def test_transaction_rolls_back_and_restores_autocommit(
    fake_connection: FakeConnection,
) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with database.transaction() as conn:
            assert conn.autocommit is False
            raise RuntimeError("boom")

    assert fake_connection.commit_calls == 0
    assert fake_connection.rollback_calls == 1
    assert fake_connection.autocommit is True
