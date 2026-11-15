"""Schema bootstrap helpers."""

from __future__ import annotations

import logging

import psycopg

from app.core.config import settings
from app.core.database import execute
from app.db.schema import migration_statements

logger = logging.getLogger(__name__)


def ensure_database_exists() -> None:
    """Create the application database if it is missing."""
    with psycopg.connect(settings.admin_database_url, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (settings.postgres_db,))
            if cur.fetchone() is None:
                logger.info("Creating database %s", settings.postgres_db)
                cur.execute(f'CREATE DATABASE "{settings.postgres_db}"')


def init_schema() -> None:
    """Apply the schema statements. Safe to call repeatedly."""
    for stmt in migration_statements():
        execute(stmt)
    logger.info("Database schema is up to date")
