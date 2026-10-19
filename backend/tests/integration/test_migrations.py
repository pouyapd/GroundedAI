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
        assert current_revision(engine) == head_revision() == "0002"
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
        assert current_revision(engine) == "0002"
    finally:
        engine.dispose()


def test_0002_adds_page_breaks_to_existing_chunks_and_reverts(tmp_path: Path) -> None:
    url = f"sqlite:///{(tmp_path / 'm.db').as_posix()}"
    upgrade_database(url, 384, "0001")
    engine = create_db_engine(url)
    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    "INSERT INTO documents (id, title, filename, media_type, size_bytes, sha256, "
                    "status, chunk_count, attempts, created_at, updated_at) VALUES ('doc_1', 't', "
                    "'t.pdf', 'application/pdf', 1, 'x', 'ready', 1, 1, '2026-10-18', '2026-10-18')"
                )
            )
            connection.execute(
                text(
                    "INSERT INTO chunks (id, document_id, chunk_index, text, char_start, char_end, "
                    "page_start, page_end, text_sha256) "
                    "VALUES ('doc_1:0000', 'doc_1', 0, 'old text', 0, 8, 1, 2, 'h')"
                )
            )
        upgrade_database(url, 384)
        with engine.connect() as connection:
            row = connection.execute(text("SELECT text, page_breaks FROM chunks")).one()
        assert tuple(row) == ("old text", None)  # old chunks keep working without breaks
        downgrade_database(url, 384, "0001")
        columns = {column["name"] for column in inspect(engine).get_columns("chunks")}
        assert "page_breaks" not in columns
    finally:
        engine.dispose()
