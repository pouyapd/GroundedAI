"""Alembic environment.

The database URL comes from (in order): ``config.attributes["database_url"]``
(set by :func:`groundedai.persistence.migrate.upgrade_database`), the
``sqlalchemy.url`` option, or the application settings.
"""

from __future__ import annotations

from alembic import context
from sqlalchemy.engine import Connection

from groundedai.persistence.db import create_db_engine
from groundedai.persistence.models import Base

config = context.config
target_metadata = Base.metadata


def _database_url() -> str:
    url = config.attributes.get("database_url") or config.get_main_option("sqlalchemy.url")
    if url:
        return str(url)
    from groundedai.core.config import get_settings

    return get_settings().resolved_database_url


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        render_as_batch=connection.dialect.name == "sqlite",
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_offline() -> None:
    context.configure(url=_database_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connection = config.attributes.get("connection")
    if connection is not None:
        _run(connection)
        return
    engine = create_db_engine(_database_url())
    try:
        with engine.connect() as new_connection:
            _run(new_connection)
    finally:
        engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
