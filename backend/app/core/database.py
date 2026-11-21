"""Database connection pool and low-level helpers."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core.config import settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    """Return the lazily created global connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=settings.database_url,
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            kwargs={"row_factory": dict_row, "autocommit": True},
            open=True,
        )
        _pool.wait(timeout=15)
        logger.info("PostgreSQL connection pool ready at %s", settings.postgres_db)
    return _pool


@contextmanager
def connection() -> Iterator[psycopg.Connection]:
    """Yield a pooled connection as a context manager."""
    with get_pool().connection() as conn:
        yield conn


@contextmanager
def transaction() -> Iterator[psycopg.Connection]:
    """Yield one pooled connection inside a real PostgreSQL transaction.

    The pool is configured with ``autocommit=True``. This helper temporarily
    switches the borrowed connection to autocommit off, so every cursor created
    from that same connection participates in one transaction. PostgreSQL DDL
    and COPY are transactional; commit happens only after the body succeeds.

    The original autocommit setting is restored in ``finally`` so the pooled
    connection is returned in its expected state.
    """
    with connection() as conn:
        previous_autocommit = conn.autocommit
        conn.autocommit = False
        try:
            yield conn
            conn.commit()
        except Exception:
            try:
                conn.rollback()
            except Exception:
                logger.exception("Transaction rollback failed")
            raise
        finally:
            conn.autocommit = previous_autocommit


def fetch_all(sql: str, params: tuple | dict | None = None) -> list[dict]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def fetch_one(sql: str, params: tuple | dict | None = None) -> dict | None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


def execute(sql: str, params: tuple | dict | None = None) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
