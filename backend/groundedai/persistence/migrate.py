"""Programmatic access to Alembic migrations."""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import Engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(database_url: str, embedding_dim: int) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR).replace("%", "%%"))
    config.attributes["database_url"] = database_url
    config.attributes["embedding_dim"] = embedding_dim
    return config


def upgrade_database(database_url: str, embedding_dim: int, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url, embedding_dim), revision)


def downgrade_database(database_url: str, embedding_dim: int, revision: str = "base") -> None:
    command.downgrade(alembic_config(database_url, embedding_dim), revision)


def head_revision() -> str | None:
    script = ScriptDirectory.from_config(alembic_config("sqlite://", 0))
    return script.get_current_head()


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()
