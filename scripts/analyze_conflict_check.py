"""How often does the figure-conflict check fire, and on which questions?

    python scripts/analyze_conflict_check.py [--mode hybrid] [--top-k 5]

Runs retrieval for every evaluation question with the settings of the answer evaluation
(400/60-character chunks, the configured embedding model), assembles the same context the
model would receive, and applies ``generation.conflicts.find_figure_conflicts``. Prints each
flagged question and a confusion table against the dataset's "conflicting" category. No model
is called.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from groundedai.core.config import get_settings  # noqa: E402
from groundedai.evaluation.dataset import load_dataset  # noqa: E402
from groundedai.evaluation.index import build_index  # noqa: E402
from groundedai.generation.conflicts import find_figure_conflicts  # noqa: E402
from groundedai.generation.context import assemble_context  # noqa: E402
from groundedai.ingestion.chunking import ChunkingConfig  # noqa: E402
from groundedai.persistence.db import session_scope  # noqa: E402
from groundedai.persistence.search import SqliteSearchBackend  # noqa: E402
from groundedai.retrieval.embeddings import build_embedding_provider  # noqa: E402
from groundedai.retrieval.engine import RetrievalEngine  # noqa: E402
from groundedai.retrieval.types import RetrievalRequest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--mode", default="hybrid", choices=["lexical", "dense", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"chunk_size": 400, "chunk_overlap": 60})
    dataset = load_dataset(settings.dataset_dir)
    embedding = build_embedding_provider(settings)
    flagged_conflict = flagged_other = missed = 0
    with tempfile.TemporaryDirectory(
        prefix="groundedai-conflicts-", ignore_cleanup_errors=True
    ) as tmp:
        corpus = build_index(dataset, embedding, ChunkingConfig(400, 60), Path(tmp), "conflicts")
        engine = RetrievalEngine(
            SqliteSearchBackend(), lambda: corpus.embedding_provider, lambda: None
        )
        try:
            for question in dataset.questions:
                request = RetrievalRequest(
                    query=question.question,
                    mode=args.mode,  # type: ignore[arg-type]
                    top_k=args.top_k,
                    candidate_k=settings.candidate_k,
                    fusion=settings.fusion,
                    alpha=settings.hybrid_alpha,
                    rrf_k=settings.rrf_k,
                )
                with session_scope(corpus.session_factory) as session:
                    chunks = engine.retrieve(session, request).chunks
                context = assemble_context(chunks, settings.max_context_chars)
                found = find_figure_conflicts(question.question, context.sources)
                is_conflict = question.category == "conflicting"
                if found:
                    print(f"{question.id} [{question.category}] {found[0].warning()}")
                if found and is_conflict:
                    flagged_conflict += 1
                elif found:
                    flagged_other += 1
                elif is_conflict:
                    missed += 1
                    print(f"{question.id} [conflicting] NOT flagged")
        finally:
            corpus.close()
    total_conflict = flagged_conflict + missed
    print(
        f"\n{args.mode}, top {args.top_k}: "
        f"conflict questions flagged {flagged_conflict}/{total_conflict}; "
        f"other questions flagged {flagged_other}/{len(dataset.questions) - total_conflict}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
