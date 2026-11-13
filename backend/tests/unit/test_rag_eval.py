import json
from pathlib import Path

import pytest

from groundedai.evaluation.dataset import DatasetError, load_dataset
from groundedai.evaluation.rag_eval import _retrieval_row, summarize


def test_retrieval_row_definitions() -> None:
    ranked = ["a", "x", "b", "y", "z"]
    row = _retrieval_row(ranked, [{"a"}, {"b", "c"}, {"d"}])
    assert row["recall@1"] == pytest.approx(1 / 3)
    assert row["recall@3"] == pytest.approx(2 / 3)
    assert row["precision@3"] == pytest.approx(2 / 3)
    assert row["mrr@10"] == 1.0
    assert 0 < row["ndcg@10"] < 1


def _row(behaviour: str, abstained: bool, correct: bool = False) -> dict[str, object]:
    return {
        "category": "factual",
        "expected_behavior": behaviour,
        "status": "insufficient_evidence" if abstained else "answered",
        "abstained": abstained,
        "correct": correct,
        "key_fact_recall": 1.0 if correct else 0.0,
        "citations": {"valid": [], "invalid": []},
        "claims": {"total": 0, "found": 0, "partial": 0, "unsupported": 0},
        "quotes": {"total": 0, "verbatim": 0, "pages_known": 0, "pages_correct": 0},
        "conflict_reported": False,
        "conflict_any": False,
        "answer_states_conflict": False,
        "timings_ms": {"total": 10.0, "generation": 5.0, "retrieval": 1.0},
    }


def test_abstention_precision_and_recall_are_kept_apart_from_correctness() -> None:
    rows = [
        _row("answer", abstained=False, correct=True),
        _row("answer", abstained=True),  # a false refusal
        _row("abstain", abstained=True),
        _row("abstain", abstained=False),  # answered a question it should refuse
    ]
    summary = summarize(rows)
    assert summary["answers"]["correct"]["mean"] == 0.5
    assert summary["answers"]["false_refusals"] == 1
    assert summary["abstention"]["precision"] == 0.5
    assert summary["abstention"]["recall"] == 0.5


def _dataset(tmp_path: Path, question: dict[str, object]) -> Path:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "a.txt").write_text("Alpha text with a fact.", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256((tmp_path / "corpus" / "a.txt").read_bytes()).hexdigest()
    manifest = {
        "name": "t",
        "version": "1",
        "license": "MIT",
        "description": "d",
        "created": "x",
        "documents": [{"key": "a", "file": "corpus/a.txt", "title": "A", "sha256": digest}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "questions.jsonl").write_text(json.dumps(question) + "\n", encoding="utf-8")
    return tmp_path


def test_scope_must_name_known_documents_and_hold_the_evidence(tmp_path: Path) -> None:
    question = {
        "id": "q1",
        "category": "factual",
        "question": "Which fact?",
        "expected_behavior": "answer",
        "evidence": [{"doc": "a", "quote": "Alpha text with a fact."}],
        "scope": ["b"],
    }
    with pytest.raises(DatasetError, match="unknown document key b in scope"):
        load_dataset(_dataset(tmp_path, question))


def test_missing_corpus_file_names_the_fetch_command(tmp_path: Path) -> None:
    root = _dataset(
        tmp_path,
        {
            "id": "q1",
            "category": "unanswerable",
            "question": "Which fact?",
            "expected_behavior": "abstain",
        },
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    manifest["fetch"] = "python scripts/fetch_nist_corpus.py"
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (root / "corpus" / "a.txt").unlink()
    with pytest.raises(DatasetError, match="fetch_nist_corpus"):
        load_dataset(root)
