"""Data-access helpers. All queries use the ORM or bound parameters."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from groundedai.persistence.models import (
    ChunkRow,
    DocumentRow,
    DocumentStatus,
    EvaluationRunRow,
    QueryEventRow,
)


class DocumentRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, document_id: str) -> DocumentRow | None:
        return self.session.get(DocumentRow, document_id)

    def get_by_sha256(self, sha256: str) -> DocumentRow | None:
        return self.session.scalar(select(DocumentRow).where(DocumentRow.sha256 == sha256))

    def page(
        self, *, offset: int, limit: int, status: str | None = None
    ) -> tuple[Sequence[DocumentRow], int]:
        query = select(DocumentRow)
        count_query = select(func.count()).select_from(DocumentRow)
        if status:
            query = query.where(DocumentRow.status == status)
            count_query = count_query.where(DocumentRow.status == status)
        rows = self.session.scalars(
            query.order_by(DocumentRow.created_at.desc(), DocumentRow.id)
            .offset(offset)
            .limit(limit)
        ).all()
        total = self.session.scalar(count_query) or 0
        return rows, total

    def existing_ids(self, document_ids: Sequence[str]) -> set[str]:
        if not document_ids:
            return set()
        return set(
            self.session.scalars(select(DocumentRow.id).where(DocumentRow.id.in_(document_ids)))
        )

    def ids_with_status(self, statuses: Sequence[str]) -> list[str]:
        return list(
            self.session.scalars(select(DocumentRow.id).where(DocumentRow.status.in_(statuses)))
        )

    def add(self, row: DocumentRow) -> None:
        self.session.add(row)

    def delete(self, row: DocumentRow) -> None:
        self.session.delete(row)

    def status_counts(self) -> dict[str, int]:
        rows = self.session.execute(
            select(DocumentRow.status, func.count()).group_by(DocumentRow.status)
        ).all()
        counts = {status.value: 0 for status in DocumentStatus}
        counts.update({str(status): int(count) for status, count in rows})
        return counts

    def embedding_models_in_use(self) -> dict[str, int]:
        rows = self.session.execute(
            select(DocumentRow.embedding_model, func.count())
            .where(DocumentRow.status == DocumentStatus.READY)
            .group_by(DocumentRow.embedding_model)
        ).all()
        return {str(model): int(count) for model, count in rows}


class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def add_many(self, rows: Sequence[ChunkRow]) -> None:
        self.session.add_all(rows)

    def delete_for_document(self, document_id: str) -> int:
        result = self.session.execute(delete(ChunkRow).where(ChunkRow.document_id == document_id))
        return int(getattr(result, "rowcount", 0) or 0)

    def get_many(self, chunk_ids: Sequence[str]) -> dict[str, tuple[ChunkRow, str]]:
        """Chunks by id together with their document title."""
        if not chunk_ids:
            return {}
        rows = self.session.execute(
            select(ChunkRow, DocumentRow.title)
            .join(DocumentRow, DocumentRow.id == ChunkRow.document_id)
            .where(ChunkRow.id.in_(chunk_ids))
        ).all()
        return {chunk.id: (chunk, title) for chunk, title in rows}

    def list_for_document(
        self, document_id: str, *, offset: int, limit: int
    ) -> tuple[Sequence[ChunkRow], int]:
        rows = self.session.scalars(
            select(ChunkRow)
            .where(ChunkRow.document_id == document_id)
            .order_by(ChunkRow.chunk_index)
            .offset(offset)
            .limit(limit)
        ).all()
        total = (
            self.session.scalar(
                select(func.count())
                .select_from(ChunkRow)
                .where(ChunkRow.document_id == document_id)
            )
            or 0
        )
        return rows, total

    def count(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(ChunkRow)) or 0)


class QueryEventRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, row: QueryEventRow) -> None:
        self.session.add(row)

    def since(self, cutoff: datetime) -> Sequence[QueryEventRow]:
        return self.session.scalars(
            select(QueryEventRow)
            .where(QueryEventRow.created_at >= cutoff)
            .order_by(QueryEventRow.created_at)
        ).all()

    def total(self) -> int:
        return int(self.session.scalar(select(func.count()).select_from(QueryEventRow)) or 0)


class EvaluationRunRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, row: EvaluationRunRow) -> None:
        self.session.add(row)

    def get(self, run_id: str) -> EvaluationRunRow | None:
        return self.session.get(EvaluationRunRow, run_id)

    def page(self, *, offset: int, limit: int) -> tuple[Sequence[EvaluationRunRow], int]:
        rows = self.session.scalars(
            select(EvaluationRunRow)
            .order_by(EvaluationRunRow.created_at.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        total = self.session.scalar(select(func.count()).select_from(EvaluationRunRow)) or 0
        return rows, total

    def running_count(self) -> int:
        return int(
            self.session.scalar(
                select(func.count())
                .select_from(EvaluationRunRow)
                .where(EvaluationRunRow.status.in_(["queued", "running"]))
            )
            or 0
        )

    def update(self, run_id: str, **fields: Any) -> None:
        row = self.get(run_id)
        if row is None:
            return
        for key, value in fields.items():
            setattr(row, key, value)
