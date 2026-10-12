"""Database engine and session factory."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker


def is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def create_db_engine(url: str) -> Engine:
    if is_sqlite(url):
        _ensure_sqlite_parent_dir(url)
        engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 30})

        @event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection: Any, _record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()

        return engine
    return create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)


def _ensure_sqlite_parent_dir(url: str) -> None:
    prefix = "sqlite:///"
    if url.startswith(prefix) and ":memory:" not in url:
        Path(url[len(prefix) :]).parent.mkdir(parents=True, exist_ok=True)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """A transactional unit of work: commit on success, roll back on error."""
    session = factory()
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def check_connection(engine: Engine) -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))
