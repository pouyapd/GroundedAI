"""Portable column types."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import DateTime, Dialect, LargeBinary
from sqlalchemy.types import TypeDecorator, TypeEngine


class UTCDateTime(TypeDecorator[datetime]):
    """Timezone-aware UTC timestamps on every backend (SQLite drops tzinfo)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect: Dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class EmbeddingVector(TypeDecorator[np.ndarray]):
    """A float32 embedding.

    PostgreSQL stores it in a pgvector ``vector`` column (dimension fixed by the
    migration). SQLite stores the raw float32 bytes; similarity is computed in
    NumPy by :class:`~groundedai.persistence.search.SqliteSearchBackend`.
    """

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector())
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32)
        if dialect.name == "postgresql":
            return array
        return array.tobytes()

    def process_result_value(self, value: Any, dialect: Dialect) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, bytes | bytearray | memoryview):
            return np.frombuffer(bytes(value), dtype=np.float32)
        return np.asarray(value, dtype=np.float32)
