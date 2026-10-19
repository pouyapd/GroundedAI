"""scripts/claim_review.py: scoring human labels, and the blind review files (no model needed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.conftest import REPO_ROOT

REVIEW_DIR = REPO_ROOT / "evaluation" / "reviews" / "claims-review-v1"
DATASET = "groundedai-claims-review-v1"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "claim_review_cli", REPO_ROOT / "scripts" / "claim_review.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _claim(status: str, evidence: list[str]) -> dict[str, Any]:
    return {"status": status, "flags": [], "cited": [], "evidence": evidence}


def _run(tmp_path: Path) -> Path:
    """Three statements: both versions accept s1 and s2 and reject s3."""
    claims = {
        "s1": _claim("supported", ["S1"]),
        "s2": _claim("added_source", ["S2"]),
        "s3": _claim("unsupported", []),
    }
    checks = {
        "dataset": DATASET,
        "questions": [
            {
                "id": "q1",
                "expected_behavior": "answer",
                "status": "answered",
                "conflicts": [],
                "warnings": [],
                "claims": {"claims-v1": claims, "claims-v2": claims},
            }
        ],
    }
    run = tmp_path / "run"
    run.mkdir()
    (run / "checks.json").write_text(json.dumps(checks), encoding="utf-8")
    return run


def _labels(tmp_path: Path, name: str, labels: dict[str, tuple[str, list[str]]]) -> Path:
    path = tmp_path / f"labels-{name}.json"
    body = {
        f"q1|{key}": {"label": label, "sources": sources}
        for key, (label, sources) in labels.items()
    }
    path.write_text(json.dumps({"dataset": DATASET, "annotator": name, "labels": body}), "utf-8")
    return path


def test_two_annotators_are_scored_on_the_statements_they_agree_on(tmp_path: Path) -> None:
    run = _run(tmp_path)
    a = _labels(
        tmp_path,
        "a",
        {"s1": ("supported", ["S1"]), "s2": ("supported", ["S2"]), "s3": ("supported", ["S3"])},
    )
    b = _labels(
        tmp_path,
        "b",
        {"s1": ("supported", ["S1"]), "s2": ("not_supported", []), "s3": ("supported", [])},
    )
    result = _script().score(run, [a, b])
    v1 = result["claims-v1"]
    # s2 is disputed: it must not count as a false acceptance or as a correct one.
    assert v1["counts"] == {
        "disputed": 1,
        "supported, accepted": 1,
        "supported, not accepted": 1,
    }
    assert v1["false_acceptance"] == {"k": 0, "n": 0, "rate": None}
    assert v1["false_rejection"] == {"k": 1, "n": 2, "rate": 0.5}
    assert v1["evidence_in_human_passages"] == {"k": 1, "n": 1, "rate": 1.0}
    assert result["agreement_between_two_annotators"] == {"k": 2, "n": 3, "rate": 0.667}
    assert result["disagreements"] == ["q1|s2"]
    assert "agree" in result["label_basis"]


def test_an_adjudication_file_settles_only_the_disputed_statements(tmp_path: Path) -> None:
    run = _run(tmp_path)
    a = _labels(tmp_path, "a", {"s1": ("supported", ["S1"]), "s2": ("supported", ["S2"])})
    b = _labels(tmp_path, "b", {"s1": ("supported", ["S1"]), "s2": ("not_supported", [])})
    # The adjudicator's label for s1 is ignored: s1 was not disputed.
    decided = _labels(
        tmp_path, "adjudicated", {"s1": ("not_supported", []), "s2": ("not_supported", [])}
    )
    result = _script().score(run, [a, b], adjudication=decided)
    v1 = result["claims-v1"]
    assert v1["counts"]["supported, accepted"] == 1
    assert v1["false_acceptance"] == {"k": 1, "n": 1, "rate": 1.0}
    assert result["disagreements"] == []
    assert result["adjudicated"] == 1


def test_one_annotator_is_reported_as_such(tmp_path: Path) -> None:
    run = _run(tmp_path)
    a = _labels(tmp_path, "a", {"s3": ("not_supported", [])})
    result = _script().score(run, [a])
    assert result["claims-v2"]["counts"] == {"not_supported, not accepted": 1, "unlabelled": 2}
    assert "one annotator" in result["label_basis"]
    assert "agreement_between_two_annotators" not in result


def test_labels_for_another_dataset_are_refused(tmp_path: Path) -> None:
    run = _run(tmp_path)
    other = tmp_path / "labels-x.json"
    other.write_text(json.dumps({"dataset": "something-else", "labels": {}}), "utf-8")
    with pytest.raises(SystemExit):
        _script().score(run, [other])


def test_the_files_a_labeller_opens_do_not_show_check_results_or_expectations() -> None:
    page = (REVIEW_DIR / "review.html").read_text(encoding="utf-8")
    answers = (REVIEW_DIR / "answers.json").read_text(encoding="utf-8")
    for text in (page, answers):
        for hidden in ("claims-v1", "claims-v2", "expected_behavior", "answer_must_include"):
            assert hidden not in text
        for status in ("wrong_source", "added_source", '"unsupported"', '"partial"'):
            assert status not in text
    # Labels kept in the browser from an earlier session must be visible and removable,
    # so a second labeller on the same browser does not start from the first one's labels.
    assert 'id="restored"' in page
    assert 'id="clear"' in page


def test_the_behaviour_table_reproduces_the_recorded_development_run() -> None:
    spec = importlib.util.spec_from_file_location(
        "answer_behaviours_cli", REPO_ROOT / "scripts" / "answer_behaviours.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = (
        REPO_ROOT
        / "evaluation/reports/dev/generation-ollama-groundedai-dev-v1-20261019T091342Z.json"
    )
    table = module.behaviours(json.loads(report.read_text(encoding="utf-8")))
    assert table["1. correct answers (every key fact)"] == "16/19"
    assert table["2. false refusals"] == "2/19"
    assert table["3. correct refusals"] == "5/6"
    assert table["10. conflict reports on other questions"] == "0/23"
