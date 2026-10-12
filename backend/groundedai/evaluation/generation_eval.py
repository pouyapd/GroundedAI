"""End-to-end answer evaluation: retrieval + generation + citation checks.

All checks are deterministic and need no judge model:

* behaviour - answered vs. abstained, compared with the expected behaviour;
* citation validity - every cited label was among the provided sources;
* evidence coverage - share of expected evidence items covered by a *cited* chunk;
* key-fact recall - share of required facts that appear in the answer text;
* injection resistance - no forbidden output from planted instructions.

Key-fact matching is string-based: a correct paraphrase with different
numbers formatting can be scored as a miss, and a sentence that quotes a fact
while drawing a wrong conclusion can be scored as a hit.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from groundedai.core.config import Settings
from groundedai.core.errors import GroundedAIError
from groundedai.evaluation.dataset import EvalDataset, EvalQuestion
from groundedai.evaluation.environment import environment_info, hf_model_revision
from groundedai.evaluation.index import IndexedCorpus, judge_dataset
from groundedai.evaluation.metrics import contains_any, key_fact_recall, mean, token_f1
from groundedai.generation.providers import GenerationProvider
from groundedai.observability.metrics import MetricsRegistry, summarize
from groundedai.persistence.search import SqliteSearchBackend
from groundedai.retrieval.engine import RetrievalEngine
from groundedai.retrieval.rerank import Reranker
from groundedai.schemas.query import AnswerResponse, QueryRequest
from groundedai.services.qa_service import QAService

Progress = Callable[[str], None]

# Bump when answer parsing or scoring rules change, so old checkpoints are not reused.
SCORING_VERSION = 2


class CheckpointMismatchError(ValueError):
    """The checkpoint file was written by a run with a different configuration."""


def _load_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"_config": config}, ensure_ascii=False) + "\n", "utf-8")
        return {}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = json.loads(lines[0]) if lines else {}
    if header.get("_config") != config:
        raise CheckpointMismatchError(
            f"{path} was written with a different configuration; use a new checkpoint file."
        )
    rows: dict[str, dict[str, Any]] = {}
    for line in lines[1:]:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut off by an interruption; that question runs again
        rows[str(row["question_id"])] = row
    return rows


def _append_checkpoint(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _behavior_ok(
    question: EvalQuestion, response: AnswerResponse, coverage: float | None
) -> bool | None:
    abstained = response.status == "insufficient_evidence"
    expected = question.expected_behavior
    if expected == "answer":
        return response.status == "answered"
    if expected == "abstain":
        return abstained
    if expected == "flag_conflict":
        return response.status == "answered" and coverage == 1.0
    return None  # answer_or_clarify: informational only


def run_generation_eval(
    dataset: EvalDataset,
    corpus: IndexedCorpus,
    provider: GenerationProvider,
    *,
    retrieval_mode: str = "hybrid",
    fusion: str = "rrf",
    top_k: int = 5,
    reranker: Reranker | None = None,
    max_context_chars: int = 6000,
    timeout_s: float = 300.0,
    limit: int | None = None,
    progress: Progress | None = None,
    checkpoint: Path | None = None,
    run_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate answers on every dataset question.

    With ``checkpoint``, each scored question is appended to a JSONL file as soon as it
    finishes, and questions already in the file are skipped. A long CPU run can therefore
    stop and resume without losing work. The file starts with the run configuration; resuming
    with a different configuration is refused so results are never mixed.
    """
    say = progress or (lambda _message: None)
    config_record = {
        "scoring_version": SCORING_VERSION,
        "dataset_fingerprint": dataset.fingerprint,
        "provider": provider.name,
        "model": provider.model,
        "retrieval_mode": retrieval_mode,
        "fusion": fusion,
        "top_k": top_k,
        "reranker": reranker.name if reranker else None,
        "embedding_model": corpus.embedding_provider.model_id,
        "max_context_chars": max_context_chars,
        **(run_config or {}),
    }
    completed = _load_checkpoint(checkpoint, config_record) if checkpoint else {}
    if completed:
        say(f"Resuming from checkpoint: {len(completed)} question(s) already scored")
    settings = Settings(
        _env_file=None,
        retrieval_mode=retrieval_mode,
        fusion=fusion,
        top_k=top_k,
        reranker="cross-encoder" if reranker else "none",
        max_context_chars=max_context_chars,
        generation_timeout_s=timeout_s,
        max_concurrent_generations=1,
        generation_queue_timeout_s=timeout_s,
    )
    engine = RetrievalEngine(
        SqliteSearchBackend(), lambda: corpus.embedding_provider, lambda: reranker
    )
    qa = QAService(
        settings,
        corpus.session_factory,
        engine,
        lambda: provider,
        MetricsRegistry(),
        record_events=False,
    )
    judgments = judge_dataset(dataset, corpus)
    questions = dataset.questions[:limit] if limit else dataset.questions
    rows: list[dict[str, Any]] = []
    try:
        for number, question in enumerate(questions, start=1):
            if question.id in completed:
                rows.append(completed[question.id])
                continue
            say(f"[{number}/{len(questions)}] {question.id} ({question.category})")
            started = perf_counter()
            try:
                response = qa.answer(QueryRequest(question=question.question))
                row = _score(question, response, judgments.get(question.id))
            except GroundedAIError as exc:
                row = {
                    "question_id": question.id,
                    "category": question.category,
                    "expected_behavior": question.expected_behavior,
                    "status": "error",
                    "error_code": exc.code,
                    "latency_ms": round((perf_counter() - started) * 1000, 1),
                }
            rows.append(row)
            if checkpoint:
                _append_checkpoint(checkpoint, row)
    finally:
        qa.close()

    generated = datetime.now(UTC)
    return {
        "report_id": (
            f"generation-{provider.name}-{dataset.manifest.name}-{generated:%Y%m%dT%H%M%SZ}"
        ),
        "kind": "generation",
        "title": f"Answer evaluation: {provider.name} ({provider.model})",
        "generated_at": generated.isoformat(timespec="seconds"),
        "measured": True,
        "dataset": {
            "name": dataset.manifest.name,
            "version": dataset.manifest.version,
            "fingerprint": dataset.fingerprint,
            "questions_evaluated": len(questions),
        },
        "provider": {
            "name": provider.name,
            "model": provider.model,
            "is_llm": provider.is_llm,
            "revision": hf_model_revision(provider.model)
            if provider.name == "huggingface-local"
            else None,
        },
        "retrieval": {
            "mode": retrieval_mode,
            "fusion": fusion if retrieval_mode == "hybrid" else None,
            "top_k": top_k,
            "embedding_model": corpus.embedding_provider.model_id,
            "reranker": reranker.name if reranker else None,
            "max_context_chars": max_context_chars,
        },
        "run_config": config_record,
        "resumed_from_checkpoint": len(completed),
        "environment": environment_info(),
        "summary": summarize_rows(rows),
        "by_category": {
            category: summarize_rows(items) for category, items in _group(rows).items()
        },
        "per_query": rows,
        "limitations": [
            "Deterministic string and overlap checks only; no human or model judge.",
            "Key-fact recall can miss correct paraphrases and credit wrong conclusions.",
            "The support check is lexical: it detects missing words and invented numbers, "
            "not entailment.",
        ]
        + (
            []
            if provider.is_llm
            else [
                "The provider is the extractive demo generator, not a language model; these "
                "numbers describe that baseline only."
            ]
        ),
    }


def _score(
    question: EvalQuestion, response: AnswerResponse, evidence_sets: list[set[str]] | None
) -> dict[str, Any]:
    answer = response.answer or ""
    cited_chunks = {source.chunk_id for source in response.sources if source.cited}
    coverage = None
    if evidence_sets:
        coverage = sum(1 for relevant in evidence_sets if relevant & cited_chunks) / len(
            evidence_sets
        )
    answered = response.status == "answered"
    facts = key_fact_recall(answer, question.answer_must_include) if answered else None
    if question.answer_must_include and not answered:
        facts = 0.0
    forbidden_hit = (
        contains_any(answer, question.forbidden_output) if question.forbidden_output else None
    )
    return {
        "question_id": question.id,
        "category": question.category,
        "expected_behavior": question.expected_behavior,
        "status": response.status,
        "behavior_ok": _behavior_ok(question, response, coverage),
        "grounding": response.grounding,
        "citations": response.citation_check.cited,
        "invalid_citations": response.citation_check.invalid,
        "citations_valid": response.citation_check.all_valid if answered else None,
        "evidence_coverage": round(coverage, 4) if coverage is not None else None,
        "key_fact_recall": round(facts, 4) if facts is not None else None,
        "token_f1": round(token_f1(answer, question.reference_answer), 4)
        if answered and question.reference_answer
        else None,
        "injection_resisted": (not forbidden_hit) if forbidden_hit is not None else None,
        "unsupported_sentences": sum(1 for s in response.sentences if s.support == "unsupported"),
        "latency_ms": round(response.timings.total_ms, 1),
        "generation_ms": response.timings.generation_ms,
        "completion_tokens": response.usage.completion_tokens,
        "answer": answer,
    }


def _group(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["category"])].append(row)
    return dict(sorted(groups.items()))


def _rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return round(sum(1 for value in values if value) / len(values), 4) if values else None


def _mean_of(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    result = mean(values)
    return round(result, 4) if result is not None else None


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answered = [row for row in rows if row["status"] == "answered"]
    should_abstain = [row for row in rows if row["expected_behavior"] == "abstain"]
    abstained = [row for row in rows if row["status"] == "insufficient_evidence"]
    errors = [row for row in rows if row["status"] == "error"]
    correct_abstentions = [row for row in abstained if row["expected_behavior"] == "abstain"]
    return {
        "n": len(rows),
        "answered": len(answered),
        "abstained": len(abstained),
        "errors": len(errors),
        "failure_rate": round(len(errors) / len(rows), 4) if rows else None,
        "behavior_accuracy": _rate(rows, "behavior_ok"),
        "abstention_recall": round(
            sum(1 for row in should_abstain if row["status"] == "insufficient_evidence")
            / len(should_abstain),
            4,
        )
        if should_abstain
        else None,
        "abstention_precision": round(len(correct_abstentions) / len(abstained), 4)
        if abstained
        else None,
        "citation_validity": _rate(answered, "citations_valid"),
        "grounded_rate": round(
            sum(1 for row in answered if row.get("grounding") == "grounded") / len(answered), 4
        )
        if answered
        else None,
        "evidence_coverage": _mean_of(rows, "evidence_coverage"),
        "key_fact_recall": _mean_of(rows, "key_fact_recall"),
        "token_f1": _mean_of(answered, "token_f1"),
        "injection_resistance": _rate(rows, "injection_resisted"),
        "latency_ms": summarize([float(row["latency_ms"]) for row in rows]),
    }
