"""Find documents indexed with older settings and re-index them in bulk.

A document is *outdated* when it is ready and its recorded index settings (chunking record,
text clean-up, embedding model) differ from the current ones; see
:meth:`IngestionService.index_differences`. The bulk job re-indexes the outdated documents
one at a time with :meth:`IngestionService.reindex`, so each stays searchable with its
previous chunks until its new ones are written, and a failure leaves its previous index in
place.

Recovery needs no saved job state: whether a document is outdated is read from the database.
After a restart (or a failure) the documents that were re-indexed are current and are not
touched again; starting the job again re-indexes only the rest. Only one job runs per process;
asking for a second one while one runs returns the running job.
"""

from __future__ import annotations

import logging
import threading
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from sqlalchemy.orm import Session, sessionmaker

from groundedai.persistence.db import session_scope
from groundedai.persistence.models import utcnow
from groundedai.persistence.repositories import DocumentRepository
from groundedai.services.ingestion_service import IngestionService, ReindexOutcome

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DifferenceCount:
    setting: str
    indexed: str
    current: str
    documents: int


@dataclass(frozen=True, slots=True)
class IndexSummary:
    ready: int
    outdated: int
    outdated_without_source: int
    differences: list[DifferenceCount]
    # outdated documents whose original file is stored, oldest first, with their titles
    reindexable: dict[str, str]

    @property
    def current(self) -> int:
        return self.ready - self.outdated


@dataclass
class ReindexJob:
    titles: dict[str, str]  # document id -> title, in the order they are re-indexed
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    state: Literal["running", "finished"] = "running"
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    reindexed: int = 0
    already_current: int = 0
    skipped: list[ReindexOutcome] = field(default_factory=list)
    failed: list[ReindexOutcome] = field(default_factory=list)
    processing_id: str | None = None
    processing_title: str | None = None

    @property
    def document_ids(self) -> list[str]:
        return list(self.titles)

    @property
    def total(self) -> int:
        return len(self.titles)

    @property
    def done(self) -> int:
        return self.reindexed + self.already_current + len(self.skipped) + len(self.failed)


class ReindexService:
    def __init__(self, session_factory: sessionmaker[Session], ingestion: IngestionService):
        self.session_factory = session_factory
        self.ingestion = ingestion
        self._lock = threading.Lock()
        self._job: ReindexJob | None = None

    @property
    def job(self) -> ReindexJob | None:
        """The running job, or the last one this process ran."""
        return self._job

    def summary(self) -> IndexSummary:
        with session_scope(self.session_factory) as session:
            rows = DocumentRepository(session).ready()
            counts: Counter[tuple[str, str, str]] = Counter()
            outdated = without_source = 0
            reindexable: dict[str, str] = {}
            for row in rows:
                differences = self.ingestion.index_differences(row)
                if not differences:
                    continue
                outdated += 1
                for difference in differences:
                    counts[(difference.setting, difference.indexed, difference.current)] += 1
                if self.ingestion.source_stored(row):
                    reindexable[row.id] = row.title
                else:
                    without_source += 1
        return IndexSummary(
            ready=len(rows),
            outdated=outdated,
            outdated_without_source=without_source,
            differences=[
                DifferenceCount(setting, indexed, current, documents)
                for (setting, indexed, current), documents in counts.most_common()
            ],
            reindexable=reindexable,
        )

    def start(self) -> tuple[ReindexJob | None, bool]:
        """A new job over the outdated documents, unless one is running (then that one).

        Returns the job and whether it is new. With nothing to re-index, no job is made.
        """
        with self._lock:
            if self._job is not None and self._job.state == "running":
                return self._job, False
            titles = self.summary().reindexable
            if not titles:
                return self._job, False
            self._job = ReindexJob(titles=titles)
            logger.info("reindex_started", extra={"job": self._job.id, "documents": len(titles)})
            return self._job, True

    def run(
        self, job: ReindexJob, on_outcome: Callable[[int, ReindexOutcome], None] | None = None
    ) -> ReindexJob:
        """Re-index the job's documents in order; each outcome is recorded as it happens
        (and passed to ``on_outcome`` with its position, counted from 1)."""
        try:
            for position, (document_id, title) in enumerate(job.titles.items(), 1):
                job.processing_id = document_id
                job.processing_title = title
                try:
                    outcome = self.ingestion.reindex(document_id)
                except Exception:  # e.g. the database was unreachable; go on with the rest
                    logger.exception("reindex_error", extra={"document_id": document_id})
                    outcome = ReindexOutcome(
                        document_id,
                        title,
                        "failed",
                        "Unexpected error while re-indexing. The previous index is still used.",
                        code="reindex_error",
                    )
                if outcome.result == "reindexed":
                    job.reindexed += 1
                elif outcome.result == "current":
                    job.already_current += 1
                elif outcome.result == "skipped":
                    job.skipped.append(outcome)
                else:
                    job.failed.append(outcome)
                if on_outcome is not None:
                    on_outcome(position, outcome)
        finally:
            job.processing_id = None
            job.processing_title = None
            job.state = "finished"
            job.finished_at = utcnow()
            logger.info(
                "reindex_finished",
                extra={
                    "job": job.id,
                    "reindexed": job.reindexed,
                    "failed": len(job.failed),
                    "skipped": len(job.skipped),
                },
            )
        return job
