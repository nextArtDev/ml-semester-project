"""Tests for the environment metadata capture."""

from __future__ import annotations

from app.core.environment import (
    capture_environment,
    database_versions,
    reset_environment_cache,
)


class TestEnvironmentCapture:
    def test_capture_returns_expected_keys(self) -> None:
        reset_environment_cache()
        env = capture_environment()
        for key in (
            "python",
            "os",
            "system",
            "machine",
            "cpu_cores",
            "numpy",
            "pandas",
            "scikit_learn",
            "postgresql",
            "pgvector",
        ):
            assert key in env

    def test_capture_is_cached(self) -> None:
        reset_environment_cache()
        first = capture_environment()
        second = capture_environment()
        assert first is second

    def test_reset_clears_cache(self) -> None:
        reset_environment_cache()
        first = capture_environment()
        reset_environment_cache()
        second = capture_environment()
        assert first is not second

    def test_python_version_is_a_string(self) -> None:
        reset_environment_cache()
        assert isinstance(capture_environment()["python"], str)

    def test_cpu_cores_is_positive(self) -> None:
        reset_environment_cache()
        cores = capture_environment()["cpu_cores"]
        assert cores is None or cores > 0

    def test_memory_is_positive_if_present(self) -> None:
        reset_environment_cache()
        memory = capture_environment()["memory_bytes"]
        assert memory is None or memory > 0

    def test_database_versions_returns_dict(self) -> None:
        versions = database_versions()
        assert isinstance(versions, dict)
        assert "postgresql" in versions
