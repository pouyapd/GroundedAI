"""End-to-end RAG evaluation on scoped questions: retrieval, answers, citations, quotes, pages.

Every question is asked through the real :class:`~groundedai.services.qa_service.QAService`,
restricted to the documents in its ``scope`` (as a user would choose them under "Search in").
The measures are kept apart on purpose; nothing is merged into one accuracy score:

* **Retrieval** (questions with evidence): Recall@k (share of evidence quotes covered by a top-k
  chunk), Precision@k (share of the top-k chunks that cover an evidence quote), MRR@10 and
  nDCG@10 with binary relevance. A chunk covers a quote when it holds at least half of it.
* **Answers** (expected to answer): *correct* = answered and every key fact present;
  *completeness* = share of key facts present. Key facts are checked as strings, so a correct
  paraphrase of a fact can be missed.
* **Abstention**: precision and recall of "not enough evidence" against the questions that
  should be refused. Ambiguous questions are reported apart.
* **Citations**: *citation precision* = share of the answer's valid ``[S#]`` labels that point
  to a passage covering an evidence quote; *citation recall* = share of evidence quotes covered
  by a cited passage; invalid labels (not among the passages given) are counted.
* **Claims (GroundedAI's own lexical check)**: share of statements found in a passage. This is
  the system under test grading itself; it is not an independent faithfulness judgement.
* **Quotes and pages**: each evidence quote returned is checked against the original document
  text (verbatim at the stated offsets) and its page against the page that offset falls on.
* **Conflicts** (questions that should flag one): a disagreement reported by the application,
  and whether the answer itself says the sources differ.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

from groundedai.core.config import Settings
from groundedai.core.errors import GroundedAIError
from groundedai.evaluation.dataset import EvalDataset, EvalQuestion
from groundedai.evaluation.index import IndexedCorpus, judge_dataset
from groundedai.evaluation.metrics import (
    bootstrap_ci,
    key_fact_recall,
    ndcg_at_k,
    reciprocal_rank,
)
from groundedai.generation.citations import states_conflict
from groundedai.generation.providers import GenerationProvider
from groundedai.ingestion.chunking import join_pages
from groundedai.ingestion.parsers import parse_document
from groundedai.ingestion.pdf_cleanup import TextCleanup
from groundedai.observability.metrics import MetricsRegistry
from groundedai.persistence.db import session_scope
from groundedai.persistence.search import SqliteSearchBackend
from groundedai.retrieval.engine import RetrievalEngine
from groundedai.retrieval.rerank import Reranker
from groundedai.schemas.query import AnswerResponse, QueryRequest
from groundedai.services.qa_service import QAService

RAG_EVAL_VERSION = 1
RETRIEVAL_KS = (1, 3, 5, 10)
FOUND = ("supported", "added_source", "wrong_source")
Progress = Callable[[str], None]


class DocumentPages:
    """Joined document text and the page each character offset falls on."""

    def __init__(self, data: bytes, suffix: str, cleanup: TextCleanup = "v1"):
        parsed = parse_document(data, suffix, max_pages=2000, max_chars=10_000_000, cleanup=cleanup)
        self.text, self.spans = join_pages(parsed.pages)

    def page_at(self, offset: int) -> int | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.number
        return None


def _retrieval_row(
    ranked: Sequence[str], evidence_sets: Sequence[set[str]]
) -> dict[str, float | None]:
    relevant = set().union(*evidence_sets)
    row: dict[str, float | None] = {}
    for k in RETRIEVAL_KS:
        top = set(ranked[:k])
        row[f"recall@{k}"] = sum(1 for items in evidence_sets if items & top) / len(evidence_sets)
        row[f"precision@{k}"] = (
            sum(1 for chunk in ranked[:k] if chunk in relevant) / k if ranked else 0.0
        )
    row["mrr@10"] = reciprocal_rank(ranked[:10], relevant)
    row["ndcg@10"] = ndcg_at_k(ranked, relevant, 10)
    return row


def score_question(
    question: EvalQuestion,
    response: AnswerResponse | None,
    ranked: Sequence[str],
    evidence_sets: Sequence[set[str]] | None,
    pages: dict[str, DocumentPages],
    chunk_index: dict[str, tuple[str, int]],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": question.id,
        "category": question.category,
        "tags": question.tags,
        "expected_behavior": question.expected_behavior,
        "scope": question.scope,
    }
    if evidence_sets:
        row["retrieval"] = _retrieval_row(ranked, evidence_sets)
    if response is None:
        return row
    answer = response.answer or ""
    row.update(
        status=response.status,
        answer=answer,
        abstained=response.status == "insufficient_evidence",
        key_fact_recall=key_fact_recall(answer, question.answer_must_include)
        if response.status == "answered"
        else (0.0 if question.answer_must_include else None),
        timings_ms={
            "retrieval": response.timings.retrieval_ms,
            "generation": response.timings.generation_ms,
            "total": response.timings.total_ms,
        },
        model_output=response.model_output,
    )
    row["correct"] = bool(response.status == "answered" and row["key_fact_recall"] == 1.0)
    # citations
    label_chunk = {source.label: source.chunk_id for source in response.sources}
    valid = response.citation_check.valid
    row["citations"] = {"valid": valid, "invalid": response.citation_check.invalid}
    if evidence_sets and response.status == "answered":
        relevant = set().union(*evidence_sets)
        cited_chunks = {label_chunk[label] for label in valid if label in label_chunk}
        row["citation_precision"] = (
            sum(1 for label in valid if label_chunk.get(label) in relevant) / len(valid)
            if valid
            else None
        )
        row["citation_recall"] = sum(1 for items in evidence_sets if items & cited_chunks) / len(
            evidence_sets
        )
    # claims, quotes and pages
    statuses = [claim.status for claim in response.claims]
    row["claims"] = {
        "total": len(statuses),
        "found": sum(1 for status in statuses if status in FOUND),
        "partial": statuses.count("partial"),
        "unsupported": statuses.count("unsupported"),
    }
    quotes = pages_ok = pages_known = verbatim = 0
    for claim in response.claims:
        for evidence in claim.evidence:
            located = chunk_index.get(evidence.chunk_id)
            if located is None:
                continue
            doc_key, chunk_start = located
            document = pages[doc_key]
            start = chunk_start + evidence.quote_start
            quotes += 1
            verbatim += int(document.text[start : start + len(evidence.quote)] == evidence.quote)
            true_page = document.page_at(start)
            if evidence.page is not None and true_page is not None:
                pages_known += 1
                pages_ok += int(evidence.page == true_page)
    row["quotes"] = {
        "total": quotes,
        "verbatim": verbatim,
        "pages_known": pages_known,
        "pages_correct": pages_ok,
    }
    row["conflict_reported"] = any(conflict.kind == "conflict" for conflict in response.conflicts)
    row["conflict_any"] = bool(response.conflicts)
    row["answer_states_conflict"] = bool(answer) and states_conflict(answer)
    return row


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _with_ci(values: Sequence[float]) -> dict[str, Any]:
    interval = bootstrap_ci(values)
    return {
        "mean": _mean(values),
        "n": len(values),
        "ci95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answered_rows = [row for row in rows if "status" in row]
    with_retrieval = [row for row in rows if "retrieval" in row]
    summary: dict[str, Any] = {"questions": len(rows)}
    if with_retrieval:
        summary["retrieval"] = {
            key: _with_ci([row["retrieval"][key] for row in with_retrieval])
            for key in with_retrieval[0]["retrieval"]
        }
    if not answered_rows:
        return summary
    answerable = [row for row in answered_rows if row["expected_behavior"] == "answer"]
    refuse = [row for row in answered_rows if row["expected_behavior"] == "abstain"]
    abstained = [row for row in answered_rows if row["abstained"]]
    true_abstentions = [row for row in abstained if row["expected_behavior"] == "abstain"]
    summary["answers"] = {
        "answerable": len(answerable),
        "correct": _with_ci([float(row["correct"]) for row in answerable]),
        "completeness": _with_ci([row["key_fact_recall"] or 0.0 for row in answerable]),
        "false_refusals": sum(1 for row in answerable if row["abstained"]),
        "invalid_output": sum(1 for row in answered_rows if row["status"] == "invalid_output"),
        "errors": sum(1 for row in answered_rows if row["status"] == "error"),
    }
    summary["abstention"] = {
        "should_refuse": len(refuse),
        "refused": len([row for row in refuse if row["abstained"]]),
        "precision": round(len(true_abstentions) / len(abstained), 4) if abstained else None,
        "recall": round(len(true_abstentions) / len(refuse), 4) if refuse else None,
        "predicted_abstentions": len(abstained),
    }
    ambiguous = [row for row in answered_rows if row["expected_behavior"] == "answer_or_clarify"]
    if ambiguous:
        summary["ambiguous"] = {
            "questions": len(ambiguous),
            "refused": sum(1 for row in ambiguous if row["abstained"]),
            "answered": sum(1 for row in ambiguous if row["status"] == "answered"),
        }
    precision = [
        row["citation_precision"]
        for row in answered_rows
        if row.get("citation_precision") is not None
    ]
    recall = [row["citation_recall"] for row in answered_rows if "citation_recall" in row]
    summary["citations"] = {
        "precision": _with_ci(precision),
        "recall": _with_ci(recall),
        "answers_with_invalid_labels": sum(
            1 for row in answered_rows if row.get("citations", {}).get("invalid")
        ),
        "answers_without_valid_label": sum(
            1
            for row in answered_rows
            if row["status"] == "answered" and not row["citations"]["valid"]
        ),
    }
    claims = [row["claims"] for row in answered_rows if row["status"] == "answered"]
    total = sum(item["total"] for item in claims)
    summary["claims_lexical_check"] = {
        "statements": total,
        "found_share": round(sum(item["found"] for item in claims) / total, 4) if total else None,
        "unsupported_share": round(sum(item["unsupported"] for item in claims) / total, 4)
        if total
        else None,
        "partial_share": round(sum(item["partial"] for item in claims) / total, 4)
        if total
        else None,
    }
    quotes = [row["quotes"] for row in answered_rows if "quotes" in row]
    q_total = sum(item["total"] for item in quotes)
    p_total = sum(item["pages_known"] for item in quotes)
    summary["quotes"] = {
        "total": q_total,
        "verbatim_share": round(sum(item["verbatim"] for item in quotes) / q_total, 4)
        if q_total
        else None,
        "page_correct_share": round(sum(item["pages_correct"] for item in quotes) / p_total, 4)
        if p_total
        else None,
        "pages_unknown": q_total - p_total,
    }
    conflicts = [row for row in answered_rows if row["expected_behavior"] == "flag_conflict"]
    if conflicts:
        summary["conflicts"] = {
            "questions": len(conflicts),
            "reported_by_application": sum(1 for row in conflicts if row["conflict_reported"]),
            "any_disagreement_note": sum(1 for row in conflicts if row["conflict_any"]),
            "stated_in_answer": sum(1 for row in conflicts if row["answer_states_conflict"]),
            "both_values_in_answer": sum(
                1 for row in conflicts if (row["key_fact_recall"] or 0) == 1.0
            ),
        }
    non_conflict = [row for row in answered_rows if row["expected_behavior"] != "flag_conflict"]
    summary["false_conflict_reports"] = sum(1 for row in non_conflict if row["conflict_reported"])
    totals = [row["timings_ms"]["total"] for row in answered_rows if row.get("timings_ms")]
    generation = [
        row["timings_ms"]["generation"]
        for row in answered_rows
        if row.get("timings_ms") and row["timings_ms"]["generation"] is not None
    ]
    retrieval = [
        row["timings_ms"]["retrieval"]
        for row in answered_rows
        if row.get("timings_ms") and row["timings_ms"]["retrieval"] is not None
    ]
    summary["latency_ms"] = {
        "total_median": _median(totals),
        "generation_median": _median(generation),
        "retrieval_median": _median(retrieval),
        "total_p90": _quantile(totals, 0.9),
    }
    return summary


def _median(values: Sequence[float]) -> float | None:
    return _quantile(values, 0.5)


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (position - low), 1)


def run_rag_eval(
    dataset: EvalDataset,
    corpus: IndexedCorpus,
    provider: GenerationProvider | None,
    settings: Settings,
    *,
    reranker: Reranker | None = None,
    progress: Progress | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Score every question; ``provider=None`` runs retrieval only."""
    say = progress or (lambda _message: None)
    engine = RetrievalEngine(
        SqliteSearchBackend(), lambda: corpus.embedding_provider, lambda: reranker
    )
    qa = QAService(
        settings,
        corpus.session_factory,
        engine,
        (lambda: provider) if provider else (lambda: None),  # type: ignore[arg-type,return-value]
        MetricsRegistry(),
        record_events=False,
        keep_model_output=True,
    )
    judgments = judge_dataset(dataset, corpus)
    pages = {
        key: DocumentPages(
            dataset.document_bytes(key), Path(doc.file).suffix.lower(), settings.text_cleanup
        )
        for key, doc in dataset.documents.items()
    }
    chunk_index = {chunk.chunk_id: (chunk.doc_key, chunk.start) for chunk in corpus.chunks}
    questions = dataset.questions[:limit] if limit else dataset.questions
    rows: list[dict[str, Any]] = []
    try:
        for number, question in enumerate(questions, start=1):
            say(f"[{number}/{len(questions)}] {question.id}")
            scope_ids = [corpus.doc_ids[key] for key in question.scope] or None
            request = qa.build_retrieval_request(
                question.question, top_k=10, document_ids=scope_ids
            )
            with session_scope(corpus.session_factory) as session:
                ranked = [chunk.chunk_id for chunk in engine.retrieve(session, request).chunks]
            response = None
            if provider is not None:
                started = perf_counter()
                try:
                    response = qa.answer(
                        QueryRequest(question=question.question, document_ids=scope_ids)
                    )
                except GroundedAIError as exc:
                    rows.append(
                        {
                            "question_id": question.id,
                            "category": question.category,
                            "expected_behavior": question.expected_behavior,
                            "status": "error",
                            "error_code": exc.code,
                            "abstained": False,
                            "key_fact_recall": None,
                            "correct": False,
                            "claims": {"total": 0, "found": 0, "partial": 0, "unsupported": 0},
                            "citations": {"valid": [], "invalid": []},
                            "conflict_reported": False,
                            "conflict_any": False,
                            "answer_states_conflict": False,
                            "timings_ms": {
                                "total": (perf_counter() - started) * 1000,
                                "generation": None,
                                "retrieval": None,
                            },
                        }
                    )
                    continue
            rows.append(
                score_question(
                    question, response, ranked, judgments.get(question.id), pages, chunk_index
                )
            )
    finally:
        qa.close()
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)
    return {
        "kind": "rag-eval",
        "rag_eval_version": RAG_EVAL_VERSION,
        "dataset": {
            "name": dataset.manifest.name,
            "fingerprint": dataset.fingerprint,
            "questions": len(questions),
        },
        "summary": summarize(rows),
        "by_category": {name: summarize(items) for name, items in sorted(by_category.items())},
        "per_question": rows,
    }
