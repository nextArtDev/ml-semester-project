"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the KNN classification platform backend."""

    model_config = SettingsConfigDict(
        env_file=(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @field_validator("postgres_password")
    @classmethod
    def _password_must_not_be_empty(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value():
            raise ValueError("POSTGRES_PASSWORD must not be empty")
        return value

    # --- Database -------------------------------------------------------
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: SecretStr = Field(..., description="Database password supplied through POSTGRES_PASSWORD")
    postgres_db: str = "knn_knn_platform"

    db_pool_min_size: int = 2
    db_pool_max_size: int = 10
    db_command_timeout_s: int = 120

    # --- API -------------------------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_prefix: str = "/api"
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    # --- Dataset cache --------------------------------------------------
    data_dir: Path = BACKEND_ROOT / "data"
    covertype_url: str = (
        "https://archive.ics.uci.edu/static/public/31/covertype.zip"
    )

    # --- Experiment defaults --------------------------------------------
    default_k: int = 5
    default_train_fraction: float = 0.7
    default_test_size: int = 5_000
    max_test_size: int = 50_000

    @property
    def database_url(self) -> str:
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def admin_database_url(self) -> str:
        """Connection to the maintenance DB, used to create the app database."""
        password = quote(self.postgres_password.get_secret_value(), safe="")
        return (
            f"postgresql://{self.postgres_user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/postgres"
        )

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
