"""Export the static demo data (frontend/public/demo) from a real local run.

The static demo (GitHub Pages) has no backend. This script indexes the bundled
synthetic corpus in a temporary database, asks a fixed list of dataset questions
through the real pipeline, and writes the responses, document list, status and an
operations snapshot as JSON. The browser replays them verbatim and labels them as
recordings. Published benchmark reports are copied unchanged (minus per-query rows).

    python scripts/export_demo_data.py
    # with real embeddings (needs the [ml] extra and a cached model):
    GROUNDEDAI_EMBEDDING_PROVIDER=sentence-transformers python scripts/export_demo_data.py
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from groundedai.core.config import get_settings  # noqa: E402
from groundedai.evaluation.dataset import EvalQuestion, load_dataset  # noqa: E402
from groundedai.evaluation.metrics import key_fact_recall  # noqa: E402
from groundedai.evaluation.reporting import list_reports, report_prompt_version  # noqa: E402
from groundedai.persistence.db import session_scope  # noqa: E402
from groundedai.persistence.repositories import DocumentRepository  # noqa: E402
from groundedai.schemas.documents import DocumentOut  # noqa: E402
from groundedai.schemas.query import QueryRequest  # noqa: E402
from groundedai.services.container import build_services  # noqa: E402

# A mix of successes and known failure modes, so the demo is representative.
# Working examples first, then the known failure modes (wrong abstention, conflict not flagged,
# and a confident answer to an out-of-scope question that still has a valid citation).
DEMO_QUESTIONS = ["q013", "q054", "q016", "q031", "q039", "q046", "q052", "q009", "q025", "q055"]
# Every question is recorded with every retrieval mode and passage count offered in the UI, so
# those controls replay a real recording made with exactly that setting.
DEMO_MODES = ["hybrid", "lexical", "dense"]
DEMO_TOP_K = [3, 5, 8, 10]
DEFAULTS = {"retrieval_mode": "hybrid", "top_k": 5}
OUT_DIR = REPO_ROOT / "frontend" / "public" / "demo"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=1, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def expectation(question: EvalQuestion) -> dict[str, Any]:
    """The dataset's expected result, shown next to the recording so failures are visible."""
    return {
        "question_id": question.id,
        "category": question.category,
        "behavior": question.expected_behavior,
        "reference_answer": question.reference_answer,
        "notes": question.notes,
    }


def fact_recall(question: EvalQuestion, answer: str | None) -> float | None:
    recall = key_fact_recall(answer, question.answer_must_include) if answer else None
    return round(recall, 4) if recall is not None else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--chunk-size", type=int, default=400)
    parser.add_argument("--chunk-overlap", type=int, default=60)
    args = parser.parse_args()

    base = get_settings()
    dataset = load_dataset(base.dataset_dir)
    questions = {question.id: question for question in dataset.questions}
    exported_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    with tempfile.TemporaryDirectory(prefix="groundedai-demo-", ignore_cleanup_errors=True) as tmp:
        settings = base.model_copy(
            update={
                "data_dir": Path(tmp),
                "database_url": None,
                "chunk_size": args.chunk_size,
                "chunk_overlap": args.chunk_overlap,
                "rate_limit_enabled": False,
            }
        )
        services = build_services(settings)
        try:
            for document in dataset.manifest.documents:
                path = dataset.document_path(document.key)
                row = services.ingestion.ingest_bytes(
                    path.name, None, path.read_bytes(), title=document.title
                )
                print(f"indexed {document.key}: {row.status}, {row.chunk_count} chunks")

            items = []
            for question_id in DEMO_QUESTIONS:
                question = questions[question_id]
                recordings = []
                for mode in DEMO_MODES:
                    for top_k in DEMO_TOP_K:
                        response = services.qa.answer(
                            QueryRequest(
                                question=question.question,
                                retrieval_mode=mode,  # type: ignore[arg-type]
                                top_k=top_k,
                                include_diagnostics=True,
                            )
                        )
                        answer = response.answer if response.status == "answered" else None
                        recordings.append(
                            {
                                "retrieval_mode": mode,
                                "top_k": top_k,
                                "key_fact_recall": fact_recall(question, answer),
                                "response": response.model_dump(mode="json"),
                            }
                        )
                statuses = {
                    f"{r['retrieval_mode']}/{r['top_k']}": r["response"]["status"]
                    for r in recordings
                }
                default = statuses["hybrid/5"]
                changed = {key: value for key, value in statuses.items() if value != default}
                print(f"{question_id}: hybrid/5 {default}; differs: {changed or 'none'}")
                items.append(
                    {
                        "question": question.question,
                        "expected": expectation(question),
                        "recordings": recordings,
                    }
                )

            with session_scope(services.session_factory) as session:
                rows, total = DocumentRepository(session).page(offset=0, limit=100)
                documents = [
                    DocumentOut.model_validate(row).model_dump(mode="json") for row in rows
                ]
            status = services.status.system_status().model_dump(mode="json")
            metrics = services.status.operational_metrics().model_dump(mode="json")
            embedding = services.embedding.get().model_id
            generation = services.generation.get()
        finally:
            services.close()

    kind = (
        f"the language model {generation.model}"
        if generation.is_llm
        else f"the extractive demo generator {generation.model}, which is not an LLM,"
    )
    note = (
        f"Answers were recorded from GroundedAI running locally with {kind} and "
        f"{embedding} embeddings, then replayed verbatim."
    )
    write_json(
        OUT_DIR / "answers.json",
        {
            "recorded_at": exported_at,
            "note": note,
            "modes": DEMO_MODES,
            "top_k_values": DEMO_TOP_K,
            "defaults": DEFAULTS,
            "items": items,
        },
    )
    write_json(
        OUT_DIR / "documents.json", {"items": documents, "total": total, "offset": 0, "limit": 100}
    )
    write_json(OUT_DIR / "status.json", status)
    write_json(OUT_DIR / "metrics.json", metrics)

    index = []
    for report in list_reports(base.published_reports_dir):
        trimmed = {key: value for key, value in report.items() if key != "per_query"}
        write_json(OUT_DIR / "reports" / f"{report['report_id']}.json", trimmed)
        dataset_info = report.get("dataset") or {}
        index.append(
            {
                "id": report["report_id"],
                "kind": report["kind"],
                "title": report.get("title", report["report_id"]),
                "generated_at": report.get("generated_at"),
                "dataset": dataset_info.get("name"),
                "n_configurations": len(report.get("configurations", []))
                if report["kind"] == "retrieval"
                else 1,
                "prompt_version": report_prompt_version(report),
            }
        )
    write_json(OUT_DIR / "reports" / "index.json", {"items": index})
    write_json(
        OUT_DIR / "manifest.json",
        {
            "exported_at": exported_at,
            "note": note
            + " Benchmark reports are real measurements from the published report files.",
        },
    )
    print(f"Wrote static demo data to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
