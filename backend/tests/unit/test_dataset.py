import json
import shutil
from pathlib import Path

import pytest

from groundedai.evaluation.dataset import DatasetError, load_dataset
from tests.conftest import DATASET_DIR


def test_bundled_dataset_loads_with_hash_verification() -> None:
    dataset = load_dataset(DATASET_DIR)
    assert dataset.manifest.name == "groundedai-synthetic-v1"
    assert len(dataset.manifest.documents) == 14
    assert len(dataset.questions) == 56
    categories = {question.category for question in dataset.questions}
    assert categories == {
        "factual",
        "multi_chunk",
        "synthesis",
        "unanswerable",
        "out_of_scope",
        "ambiguous",
        "conflicting",
        "injection",
    }
    assert len(dataset.fingerprint) == 16


def test_answerable_questions_have_evidence() -> None:
    dataset = load_dataset(DATASET_DIR)
    for question in dataset.questions:
        if question.expected_behavior in ("answer", "flag_conflict"):
            assert question.evidence, question.id


def _copy(tmp_path: Path) -> Path:
    target = tmp_path / "dataset"
    shutil.copytree(DATASET_DIR, target)
    return target


def test_tampered_document_fails_hash_check(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    path = root / "corpus" / "tessera-api-limits.txt"
    path.write_text(path.read_text(encoding="utf-8") + "extra", encoding="utf-8")
    with pytest.raises(DatasetError, match="hash mismatch"):
        load_dataset(root)


def test_duplicate_question_ids_are_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    questions = root / "questions.jsonl"
    first_line = questions.read_text(encoding="utf-8").splitlines()[0]
    questions.write_text(first_line + "\n" + first_line + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate question id"):
        load_dataset(root)


def test_unknown_evidence_document_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    record = {
        "id": "x1",
        "category": "factual",
        "question": "q?",
        "expected_behavior": "answer",
        "evidence": [{"doc": "missing-doc", "quote": "a quote that is long enough"}],
    }
    (root / "questions.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="unknown document key"):
        load_dataset(root)


def test_not_a_dataset_directory(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_dataset(tmp_path)
