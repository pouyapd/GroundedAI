"""Answer-level check: would an NLI verifier catch wrong answers that the lexical check passes?

    python experiments/verifier/answer_level.py --report test-report.json \
        --dataset evaluation/datasets/groundedai-nist-v1 --out answer-level.json

Takes a finished rag_eval report (the answers are not generated again), rebuilds the same index
with the report's settings, retrieves the same passages for each question, and checks every
statement of each answer two ways:

* lexical: GroundedAI's claim check (the version in the report's settings) - "found";
* NLI: a pretrained NLI model, zero-shot - "supported" when at least one premise entails the
  statement (hypothesis). ``--premise passage``: each retrieved passage whole;
  ``--premise window``: every window of one or two consecutive sentences of each passage (the
  SummaC zero-shot idea, Laban et al. 2022: NLI models are trained on short premises and lose
  entailment in long, noisy ones).

An answer "passes" a check when all its statements pass. Each check is then compared with the
answer's correctness (key facts, from the report; an answer to a question that should have been
refused is wrong). Rows are reproduced only when the lexical check on the rebuilt passages gives
the same counts as the report, which confirms the passages are the ones the model saw.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "experiments" / "verifier"))
sys.path.insert(0, str(ROOT / "scripts"))

from verifier_lib import CrossEncoderClassifier  # noqa: E402


def value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def rates(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    wrong = [r for r in rows if not r["correct"]]
    right = [r for r in rows if r["correct"]]
    caught = sum(1 for r in wrong if not r[key])
    alarms = sum(1 for r in right if not r[key])
    return {
        "answers": len(rows),
        "wrong_answers": len(wrong),
        "wrong_answers_flagged": caught,
        "wrong_answers_passed": len(wrong) - caught,
        "correct_answers_flagged": alarms,
        "catch_rate": round(caught / len(wrong), 4) if wrong else None,
        "false_alarm_rate": round(alarms / len(right), 4) if right else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--premise", choices=["passage", "window"], default="passage")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from groundedai.core.config import Settings
    from groundedai.core.text import normalize_whitespace, split_sentences
    from groundedai.evaluation.dataset import load_dataset
    from groundedai.evaluation.index import build_index
    from groundedai.generation.citations import parse_answer
    from groundedai.generation.claims import MATCHED, EvidenceSource, check_claims
    from groundedai.generation.context import assemble_context
    from groundedai.ingestion.chunking import ChunkingConfig
    from groundedai.observability.metrics import MetricsRegistry
    from groundedai.persistence.db import session_scope
    from groundedai.persistence.search import SqliteSearchBackend
    from groundedai.retrieval.embeddings import build_embedding_provider
    from groundedai.retrieval.engine import RetrievalEngine
    from groundedai.services.qa_service import QAService

    report = json.loads(args.report.read_text(encoding="utf-8"))
    values = {k: value(str(v)) for k, v in report["settings_overrides"].items()}
    workdir = Path(tempfile.mkdtemp(prefix="answer-level-"))
    settings = Settings(_env_file=None, data_dir=workdir, rate_limit_enabled=False, **values)
    dataset = load_dataset(args.dataset)
    corpus = build_index(
        dataset,
        build_embedding_provider(settings),
        ChunkingConfig(settings.chunk_size, settings.chunk_overlap, settings.text_cleanup),
        workdir,
        dataset.manifest.name,
    )
    engine = RetrievalEngine(SqliteSearchBackend(), lambda: corpus.embedding_provider, lambda: None)
    qa = QAService(
        settings,
        corpus.session_factory,
        engine,
        lambda: None,  # type: ignore[arg-type,return-value]
        MetricsRegistry(),
        record_events=False,
    )
    classifier = CrossEncoderClassifier(args.model, max_length=512)
    questions = {q.id: q for q in dataset.questions}
    rows: list[dict[str, Any]] = []
    skipped = {"abstained_or_error": 0, "not_scored": 0, "passages_differ": 0}
    started = time.perf_counter()
    for row in report["per_question"]:
        question = questions[row["question_id"]]
        if row.get("status") != "answered" or row.get("abstained") or not row.get("answer"):
            skipped["abstained_or_error"] += 1
            continue
        if row["expected_behavior"] not in ("answer", "abstain"):
            skipped["not_scored"] += 1  # ambiguous and conflict questions have no single answer
            continue
        scope = [corpus.doc_ids[key] for key in question.scope] or None
        request = qa.build_retrieval_request(question.question, document_ids=scope)
        with session_scope(corpus.session_factory) as session:
            chunks = engine.retrieve(session, request).chunks
        context = assemble_context(chunks, settings.max_context_chars, settings.prompt_version)
        sources = [EvidenceSource(b.label, b.chunk.document_title, b.text) for b in context.sources]
        raw = row.get("model_output") or row["answer"]  # what the application checked
        parsed = parse_answer(raw, {s.label: s.text for s in sources})
        checked = parsed.answer_part or raw
        lexical = check_claims(checked, sources, question.question, settings.claim_check)
        found = sum(1 for c in lexical.claims if c.status in MATCHED)
        if (len(lexical.claims), found) != (row["claims"]["total"], row["claims"]["found"]):
            skipped["passages_differ"] += 1
            continue
        statements = [c.text for c in lexical.claims]
        premises: list[tuple[str, str]] = []  # (source label, premise text)
        for source in sources:
            text = normalize_whitespace(source.text)
            if args.premise == "passage":
                premises.append((source.label, text))
                continue
            sentences = split_sentences(text) or [text]
            premises.extend((source.label, one) for one in sentences)
            premises.extend((source.label, f"{a} {b}") for a, b in itertools.pairwise(sentences))
        pairs = [(s, premise) for s in statements for _, premise in premises]
        probs = classifier.probabilities(pairs) if pairs else []
        labels = []
        detail = []
        for i, statement in enumerate(statements):
            per_premise = probs[i * len(premises) : (i + 1) * len(premises)]
            by_source: dict[str, list[str]] = {}
            for (label, _), p in zip(premises, per_premise, strict=True):
                by_source.setdefault(label, []).append(max(p, key=p.__getitem__))
            tops = [
                "entailment"
                if "entailment" in found_labels
                else ("contradiction" if "contradiction" in found_labels else "neutral")
                for found_labels in by_source.values()
            ]
            labels.append(
                "entailment"
                if "entailment" in tops
                else ("contradiction" if "contradiction" in tops else "neutral")
            )
            detail.append(
                {
                    "statement": statement,
                    "lexical": lexical.claims[i].status,
                    "nli": labels[-1],
                    "per_source": dict(zip(by_source, tops, strict=True)),
                }
            )
        rows.append(
            {
                "question_id": row["question_id"],
                "expected_behavior": row["expected_behavior"],
                "correct": bool(row["correct"]) and row["expected_behavior"] == "answer",
                "statements": len(statements),
                "lexical_found": found,
                "nli_entailed": labels.count("entailment"),
                "nli_contradicted": labels.count("contradiction"),
                "lexical_pass": bool(statements) and found == len(statements),
                "nli_pass": bool(statements) and labels.count("entailment") == len(statements),
                "detail": detail,
            }
        )
    out = {
        "report": report.get("label") or args.report.name,
        "dataset": report["dataset"],
        "settings_overrides": report["settings_overrides"],
        "nli_model": args.model,
        "premise": args.premise,
        "seconds": round(time.perf_counter() - started, 1),
        "skipped": skipped,
        "lexical_all_found": rates(rows, "lexical_pass"),
        "nli_all_entailed": rates(rows, "nli_pass"),
        "rows": rows,
    }
    qa.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(out["report"], "skipped", skipped)
    print(" lexical", out["lexical_all_found"])
    print(" nli    ", out["nli_all_entailed"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
