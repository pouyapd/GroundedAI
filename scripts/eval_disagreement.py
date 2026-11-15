"""Score the disagreement check on questions with a known outcome, through the real QA pipeline.

    python scripts/eval_disagreement.py --dataset evaluation/datasets/groundedai-versions-v1 \
        --split dev --out report.json [--set conflict_check=disagreement-v3]
    python scripts/eval_disagreement.py --dataset evaluation/datasets/groundedai-nist-v1 \
        --out report.json            # its flag_conflict questions are the positives

Every question is asked with the extractive generator against its scope, and the disagreements
in the answer are compared with the expected outcome. A question tagged
``expect-conflict:versions`` (or with ``expected_behavior`` ``flag_conflict``) expects a
reported disagreement whose two statements come from two different documents of its scope and
whose quotes contain both expected values (``answer_must_include``, alternatives with ``|``).
Every other question expects no reported disagreement. A report that pairs the wrong documents
or misses a value is not counted as correct.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def expected_outcome(question: Any) -> str:
    for tag in question.tags:
        if tag.startswith("expect-conflict:"):
            return "versions" if tag.endswith(":versions") else "none"
    return "versions" if question.expected_behavior == "flag_conflict" else "none"


def split_of(question: Any) -> str:
    for tag in question.tags:
        if tag.startswith("split:"):
            return tag.split(":", 1)[1]
    return "test"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="all", help="dev, validation, test or all")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from groundedai.core.config import Settings
    from groundedai.evaluation.dataset import load_dataset
    from groundedai.evaluation.environment import code_version, environment_info
    from groundedai.evaluation.index import build_index
    from groundedai.generation.providers import build_generation_provider
    from groundedai.ingestion.chunking import ChunkingConfig
    from groundedai.observability.metrics import MetricsRegistry
    from groundedai.persistence.search import SqliteSearchBackend
    from groundedai.retrieval.embeddings import build_embedding_provider
    from groundedai.retrieval.engine import RetrievalEngine
    from groundedai.schemas.query import QueryRequest
    from groundedai.services.qa_service import QAService

    overrides = dict(item.split("=", 1) for item in args.set)
    workdir = Path(tempfile.mkdtemp(prefix="eval-disagreement-"))
    settings = Settings(
        _env_file=None,
        data_dir=workdir,
        rate_limit_enabled=False,
        generation_provider="extractive-demo",
        **{k: value(v) for k, v in overrides.items()},
    )
    dataset = load_dataset(args.dataset)
    corpus = build_index(
        dataset,
        build_embedding_provider(settings),
        ChunkingConfig(settings.chunk_size, settings.chunk_overlap, settings.text_cleanup),
        workdir,
        dataset.manifest.name,
    )
    provider = build_generation_provider(settings)
    engine = RetrievalEngine(SqliteSearchBackend(), lambda: corpus.embedding_provider, lambda: None)
    qa = QAService(
        settings,
        corpus.session_factory,
        engine,
        lambda: provider,
        MetricsRegistry(),
        record_events=False,
    )
    key_of = {document_id: key for key, document_id in corpus.doc_ids.items()}
    rows = []
    for question in dataset.questions:
        if args.split != "all" and split_of(question) != args.split:
            continue
        expected = expected_outcome(question)
        scope = [corpus.doc_ids[key] for key in question.scope] or None
        response = qa.answer(QueryRequest(question=question.question, document_ids=scope))
        reports = []
        for conflict in response.conflicts:
            docs = [key_of.get(s.document_id, s.document_id) for s in conflict.statements]
            quotes = " ".join(s.quote for s in conflict.statements).lower()
            values_found = [
                any(alt.lower() in quotes for alt in fact.split("|"))
                for fact in question.answer_must_include
            ]
            reports.append(
                {
                    "kind": conflict.kind,
                    "documents": docs,
                    "pages": [s.page for s in conflict.statements],
                    "values": [s.values for s in conflict.statements],
                    "quotes": [s.quote for s in conflict.statements],
                    "two_documents_in_scope": len(set(docs)) == 2
                    and set(docs) <= set(question.scope),
                    "expected_values_found": all(values_found) if values_found else False,
                }
            )
        correct = [r for r in reports if r["two_documents_in_scope"] and r["expected_values_found"]]
        if expected == "versions":
            outcome = "TP" if correct else ("wrong report" if reports else "FN")
        else:
            outcome = "FP" if reports else "TN"
        rows.append(
            {
                "id": question.id,
                "split": split_of(question),
                "tags": question.tags,
                "question": question.question,
                "expected": expected,
                "outcome": outcome,
                "reports": reports,
                "answer_status": response.status,
            }
        )
    qa.close()
    positives = [r for r in rows if r["expected"] == "versions"]
    negatives = [r for r in rows if r["expected"] == "none"]
    tp = sum(r["outcome"] == "TP" for r in rows)
    fp = sum(r["outcome"] == "FP" for r in rows)
    wrong = sum(r["outcome"] == "wrong report" for r in rows)
    reported = tp + fp + wrong
    summary = {
        "questions": len(rows),
        "positives": len(positives),
        "negatives": len(negatives),
        "true_positives": tp,
        "false_negatives": sum(r["outcome"] == "FN" for r in rows),
        "wrong_reports_on_positives": wrong,
        "false_positives": fp,
        "precision": round(tp / reported, 4) if reported else None,
        "recall": round(tp / len(positives), 4) if positives else None,
        "false_positive_rate": round(fp / len(negatives), 4) if negatives else None,
    }
    report = {
        "kind": "disagreement-eval",
        "dataset": {"name": dataset.manifest.name, "fingerprint": dataset.fingerprint},
        "split": args.split,
        "settings_overrides": overrides,
        "conflict_check": settings.conflict_check,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": environment_info(),
        "code": code_version(),
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary))
    for row in rows:
        kinds = [(r["kind"], r["documents"]) for r in row["reports"]]
        print(f"  {row['id']:5s} {row['expected']:8s} {row['outcome']:12s} {kinds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
