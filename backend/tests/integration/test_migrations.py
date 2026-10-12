from pathlib import Path

from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import inspect, text

from groundedai.persistence.db import create_db_engine
from groundedai.persistence.migrate import (
    current_revision,
    downgrade_database,
    head_revision,
    upgrade_database,
)
from groundedai.persistence.models import Base

SEARCH_OBJECTS = ("chunks_fts", "tsv", "ix_chunks_tsv", "ix_chunks_embedding_hnsw")


def _include(
    obj: object, name: str | None, type_: str, reflected: bool, compare_to: object
) -> bool:
    return not (name or "").startswith(SEARCH_OBJECTS)


def test_upgrade_creates_schema_and_matches_models(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade_database(url, 384)
    engine = create_db_engine(url)
    try:
        assert current_revision(engine) == head_revision() == "0001"
        tables = set(inspect(engine).get_table_names())
        assert {"documents", "chunks", "chunks_fts", "query_events", "evaluation_runs"} <= tables
        with engine.connect() as connection:
            context = MigrationContext.configure(
                connection, opts={"include_object": _include, "compare_type": False}
            )
            assert compare_metadata(context, Base.metadata) == []
    finally:
        engine.dispose()


def test_downgrade_removes_everything(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade_database(url, 384)
    downgrade_database(url, 384)
    engine = create_db_engine(url)
    try:
        assert set(inspect(engine).get_table_names()) == {"alembic_version"}
        with engine.connect() as connection:
            triggers = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type = 'trigger'")
            ).all()
        assert triggers == []
    finally:
        engine.dispose()


def test_upgrade_is_idempotent(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade_database(url, 384)
    upgrade_database(url, 384)
    engine = create_db_engine(url)
    try:
        assert current_revision(engine) == "0001"
    finally:
        engine.dispose()
