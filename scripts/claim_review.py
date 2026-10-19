"""Human review of the claim check on real model answers: collect, label blind, score.

    python scripts/claim_review.py collect --out evaluation/reviews/claims-review-v1
        [--dataset-dir evaluation/datasets/groundedai-claims-review-v1] [--provider ollama]
    # open OUT/review.html in a browser, label every statement, download labels-<name>.json
    python scripts/claim_review.py score --run OUT --labels labels-a.json [labels-b.json]

``collect`` asks every question of the dataset once with the configured model (settings from
``GROUNDEDAI_*`` variables, as for ``groundedai eval generation``), and checks each answer with
both claim-check versions. It writes:

* ``answers.json``: questions, answers and the passages the model received; no check results;
* ``checks.json``: what claims-v1 and claims-v2 said about every statement (keep it closed
  while labelling);
* ``review.html``: a self-contained page that shows each answer, its statements and the
  passages, and asks for a label per statement (supported / partly supported / not supported /
  unclear) and the passages that support it. It never shows the check results.

``score`` compares the human labels with each version: a statement the check accepts (found
by a cited or uncited passage) but people mark "not supported" is a false acceptance; one the
check does not accept but people mark "supported" is a false rejection. With two label files it
also reports how often the two people agree. Nothing is scored without human labels.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.core.config import Settings  # noqa: E402
from groundedai.evaluation.dataset import load_dataset  # noqa: E402
from groundedai.evaluation.environment import code_version, environment_info  # noqa: E402
from groundedai.evaluation.index import build_index  # noqa: E402
from groundedai.generation.claims import (  # noqa: E402
    CLAIM_CHECK_VERSIONS,
    MATCHED,
    EvidenceSource,
    check_claims,
)
from groundedai.generation.providers import build_generation_provider  # noqa: E402
from groundedai.ingestion.chunking import ChunkingConfig  # noqa: E402
from groundedai.observability.metrics import MetricsRegistry  # noqa: E402
from groundedai.persistence.search import SqliteSearchBackend  # noqa: E402
from groundedai.retrieval.embeddings import build_embedding_provider  # noqa: E402
from groundedai.retrieval.engine import RetrievalEngine  # noqa: E402
from groundedai.schemas.query import QueryRequest  # noqa: E402
from groundedai.services.qa_service import QAService  # noqa: E402

DEFAULT_DATASET = ROOT / "evaluation" / "datasets" / "groundedai-claims-review-v1"
LABELS = ("supported", "partly_supported", "not_supported", "unclear")
_BRACKETS_RE = re.compile(r"\[[^\[\]\n]*\]")


def claim_key(text: str) -> str:
    """The statement without brackets, spacing or case, to match it across versions."""
    return re.sub(r"\W+", " ", _BRACKETS_RE.sub(" ", text)).strip().casefold()


# ---------------------------------------------------------------------- collect
def collect(dataset_dir: Path, out: Path) -> None:
    dataset = load_dataset(dataset_dir)
    settings = Settings(claim_check="claims-v2", max_concurrent_generations=1)
    provider = build_generation_provider(settings)
    out.mkdir(parents=True, exist_ok=True)
    questions, checks = [], []
    with tempfile.TemporaryDirectory(prefix="claim-review-", ignore_cleanup_errors=True) as tmp:
        corpus = build_index(
            dataset,
            build_embedding_provider(settings),
            ChunkingConfig(settings.chunk_size, settings.chunk_overlap),
            Path(tmp),
            "review",
        )
        corpus_model = corpus.embedding_provider.model_id
        engine = RetrievalEngine(
            SqliteSearchBackend(), lambda: corpus.embedding_provider, lambda: None
        )
        qa = QAService(
            settings,
            corpus.session_factory,
            engine,
            lambda: provider,
            MetricsRegistry(),
            record_events=False,
            keep_model_output=True,
        )
        try:
            for number, question in enumerate(dataset.questions, start=1):
                print(f"[{number}/{len(dataset.questions)}] {question.id}", flush=True)
                response = qa.answer(QueryRequest(question=question.question))
                sources = [
                    EvidenceSource(s.label, s.document_title, s.text) for s in response.sources
                ]
                raw = response.model_output or ""
                reports = {}
                if response.status == "answered":
                    reports = {
                        version: check_claims(raw, sources, question.question, version)
                        for version in CLAIM_CHECK_VERSIONS
                    }
                    # The recomputed v2 result must be what the application returned.
                    returned = [claim.status for claim in response.claims]
                    if [c.status for c in reports["claims-v2"].claims] != returned:
                        raise SystemExit(f"{question.id}: recomputed claims-v2 differs")
                statements: dict[str, str] = {}
                for report in reports.values():
                    for claim in report.claims:
                        statements.setdefault(claim_key(claim.text), claim.text)
                questions.append(
                    {
                        "id": question.id,
                        "question": question.question,
                        "status": response.status,
                        "answer": response.answer,
                        "model_output": raw,
                        "statements": [
                            {"key": key, "text": text} for key, text in statements.items()
                        ],
                        "sources": [
                            {
                                "label": s.label,
                                "title": s.document_title,
                                "pages": [s.page_start, s.page_end],
                                "text": s.text,
                            }
                            for s in response.sources
                        ],
                    }
                )
                checks.append(
                    {
                        "id": question.id,
                        "expected_behavior": question.expected_behavior,
                        "status": response.status,
                        "conflicts": [c.kind for c in response.conflicts],
                        "warnings": response.warnings,
                        "claims": {
                            version: {
                                claim_key(claim.text): {
                                    "status": claim.status,
                                    "flags": claim.flags,
                                    "cited": claim.cited,
                                    "evidence": [quote.label for quote in claim.evidence],
                                }
                                for claim in report.claims
                            }
                            for version, report in reports.items()
                        },
                    }
                )
        finally:
            qa.close()
            corpus.close()
    meta = {
        "dataset": dataset.manifest.name,
        "dataset_fingerprint": dataset.fingerprint,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "provider": {"name": provider.name, "model": provider.model},
        "settings": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "embedding": corpus_model,
            "retrieval_mode": settings.retrieval_mode,
            "top_k": settings.top_k,
            "prompt_version": settings.prompt_version,
            "conflict_check": settings.conflict_check,
        },
        "environment": environment_info(),
    }
    _write(out / "answers.json", {**meta, "questions": questions})
    _write(out / "checks.json", {**meta, "questions": checks})
    (out / "review.html").write_text(review_page(meta, questions), encoding="utf-8", newline="\n")
    print(f"Wrote {out / 'answers.json'}, {out / 'checks.json'} and {out / 'review.html'}")


def _write(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", "utf-8", newline="\n")


def review_page(meta: dict[str, Any], questions: list[dict[str, Any]]) -> str:
    data = json.dumps({"dataset": meta["dataset"], "questions": questions}, ensure_ascii=False)
    data = data.replace("</", "<\\/")
    return _PAGE.replace("__DATA__", data).replace("__TITLE__", html.escape(meta["dataset"]))


_PAGE = (Path(__file__).with_name("claim_review_page.html")).read_text(encoding="utf-8")


# ------------------------------------------------------------------------ score
def score(run: Path, label_files: list[Path]) -> dict[str, Any]:
    checks = json.loads((run / "checks.json").read_text(encoding="utf-8"))
    people = [json.loads(path.read_text(encoding="utf-8")) for path in label_files]
    for person in people:
        if person.get("dataset") != checks["dataset"]:
            raise SystemExit("labels are for another dataset")
    result: dict[str, Any] = {"dataset": checks["dataset"], "annotators": len(people)}
    first = people[0]["labels"] if people else {}
    for version in CLAIM_CHECK_VERSIONS:
        counts: Counter[str] = Counter()
        attributed = attributable = 0
        for question in checks["questions"]:
            for key, claim in question["claims"].get(version, {}).items():
                human = first.get(f"{question['id']}|{key}", {})
                label = human.get("label")
                if label not in LABELS:
                    counts["unlabelled"] += 1
                    continue
                accepted = claim["status"] in MATCHED
                counts[f"{label}, {'accepted' if accepted else 'not accepted'}"] += 1
                if label == "supported" and accepted and human.get("sources"):
                    attributable += 1
                    attributed += bool(set(claim["evidence"]) & set(human["sources"]))
        supported = counts["supported, accepted"] + counts["supported, not accepted"]
        unsupported = counts["not_supported, accepted"] + counts["not_supported, not accepted"]
        result[version] = {
            "counts": dict(sorted(counts.items())),
            "false_acceptance": _rate(counts["not_supported, accepted"], unsupported),
            "false_rejection": _rate(counts["supported, not accepted"], supported),
            "evidence_in_human_passages": _rate(attributed, attributable),
        }
    if len(people) >= 2:
        a, b = people[0]["labels"], people[1]["labels"]
        both = [k for k in a if a[k].get("label") in LABELS and b.get(k, {}).get("label") in LABELS]
        same = sum(1 for k in both if a[k]["label"] == b[k]["label"])
        result["agreement_between_two_annotators"] = _rate(same, len(both))
    conflicts = [q for q in checks["questions"] if q["expected_behavior"] == "flag_conflict"]
    reported = [q for q in checks["questions"] if q["conflicts"]]
    result["conflicts_dataset_labels"] = {
        "note": "expected_behavior written by the dataset author, not by the annotators",
        "conflict_questions_with_a_report": _rate(
            sum(bool(q["conflicts"]) for q in conflicts), len(conflicts)
        ),
        "reports_on_other_questions": len([q for q in reported if q not in conflicts]),
    }
    return result


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": round(k / n, 3) if n else None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("collect", help="ask every question once and write the files")
    run.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    run.add_argument("--out", type=Path, required=True)
    rate = commands.add_parser("score", help="compare human labels with both check versions")
    rate.add_argument("--run", type=Path, required=True)
    rate.add_argument("--labels", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "collect":
        collect(args.dataset_dir, args.out)
    else:
        print(json.dumps(score(args.run, args.labels), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
