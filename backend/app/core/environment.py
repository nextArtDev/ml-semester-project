"""Capture of the execution environment for reproducible experiments.

Every experiment records the software versions and the hardware it ran on,
so that a latency measurement can be interpreted in context and a result can
be reproduced on comparable hardware.

The capture is deliberately best-effort: a missing value is recorded as
``None`` rather than raising, because the environment must never be able to
abort an experiment.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from typing import Any

from app.core.database import fetch_one


def _run(command: list[str]) -> str | None:
    """Run a command and return its trimmed stdout, or None on failure."""
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    output = completed.stdout.strip()
    return output or None


def _cpu_model() -> str | None:
    if platform.system() == "Windows":
        value = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_Processor).Name",
            ]
        )
        return value
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _total_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        value = _run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
            ]
        )
        if value:
            try:
                return int(value.strip())
            except ValueError:
                return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages) * int(page_size)
    except (ValueError, OSError, AttributeError):
        return None


def _logical_cores() -> int | None:
    return os.cpu_count()


def database_versions() -> dict[str, Any]:
    """The PostgreSQL and pgvector versions of the connected server."""
    versions: dict[str, Any] = {
        "postgresql": None,
        "pgvector": None,
        "server_os": None,
    }
    try:
        row = fetch_one(
            "SELECT version() AS server_version, "
            "current_setting('server_version') AS server_version_num, "
            "(SELECT extversion FROM pg_extension WHERE extname = 'vector') AS pgvector"
        )
    except Exception:
        return versions

    if row:
        versions["postgresql"] = row.get("server_version_num")
        versions["postgresql_full"] = row.get("server_version")
        versions["pgvector"] = row.get("pgvector")
    return versions


@lru_cache(maxsize=1)
def capture_environment() -> dict[str, Any]:
    """Capture the software and hardware environment.

    The database query is executed once and cached, because the versions
    cannot change during the lifetime of the process.
    """
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_cores": _logical_cores(),
        "memory_bytes": _total_memory_bytes(),
        "numpy": _package_version("numpy"),
        "pandas": _package_version("pandas"),
        "scikit_learn": _package_version("sklearn"),
        "psycopg": _package_version("psycopg"),
        "fastapi": _package_version("fastapi"),
    }
    environment.update(database_versions())
    environment["database_host"] = os.environ.get("POSTGRES_HOST")
    return environment


def _package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def reset_environment_cache() -> None:
    """Clear the cached environment. Used by the tests."""
    capture_environment.cache_clear()
