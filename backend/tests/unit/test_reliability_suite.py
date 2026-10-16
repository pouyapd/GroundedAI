"""The groundedai-reliability-v1 tooling, on a small hand-written suite (not the real cases)."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from groundedai.evaluation import reliability as suite
from tests.conftest import REPO_ROOT

SPEC = REPO_ROOT / "evaluation" / "datasets" / suite.SUITE_NAME / "SPEC.md"

DEV_DOC = """Riverside Waste Service - Collection Guide
Household bins are collected every Tuesday from 07:00, and the crew starts in the north of the
town. Garden waste is collected every two weeks between March and November. Residents should put
the green bin out by the kerb the evening before.

A replacement bin costs €35 and is delivered within 10 working days of the order. Bulky items
can be collected for a fee of €25 per item, up to five items per visit, and must be left by the
front gate.

Missed collections must be reported within 48 hours through the online form or by phone.
Residents may not place hazardous waste such as paint or batteries in any bin; take it to the
depot instead.
"""
TEST_DOC = """Gullhaven Ferries - Passenger Guide
The first ferry leaves Gullhaven at 06:15 and the last one returns at 22:40 every day of the
week. Boarding closes ten minutes before departure, and late passengers wait for the next
crossing.

An adult single ticket costs €12, and children under 12 travel for €6. Bicycles travel free of
charge, but each passenger may bring only one bicycle on board at a time.

Tickets bought on board cost €2 more than tickets bought online. Refunds are paid within 14 days
if a crossing is cancelled because of the weather, and passengers are told by text message.
"""


def _passages(document_id: str, body: str, prefix: str) -> list[dict[str, str]]:
    parts = [part.strip() for part in body.strip().split("\n\n")]
    return [
        {"passage_id": f"P-{prefix}-{index:03d}", "document_id": document_id, "text": part}
        for index, part in enumerate(parts, start=1)
    ]


def _abc(split: str, prefix: str, domain: str, pool: list[dict[str, str]]) -> dict[str, list[Any]]:
    refs = [{"label": f"S{i}", "passage_id": p["passage_id"]} for i, p in enumerate(pool, start=1)]
    s = "D" if split == "dev" else "T"
    if split == "dev":
        fee, fee_text, refusal_q = (
            "€35",
            "A replacement bin costs €35",
            "Who leads the waste service?",
        )
        q_cost, q_absent = (
            "How much is a replacement bin?",
            "What do residents receive each spring?",
        )
        bad = "Residents receive a free compost bin every spring. [S2]"
        b_ok, b_q = (
            "Bulky items cost €25 per item to collect. [S2]",
            "What does bulky item collection cost?",
        )
        b_bad_q = "When are missed collections reported?"
        b_bad = "Missed collections must be reported within 48 hours. [S4]"
    else:
        fee, fee_text, refusal_q = (
            "€12",
            "An adult single ticket costs €12",
            "Who captains the morning ferry?",
        )
        q_cost, q_absent = "What does an adult single ticket cost?", "Is there a lounge on board?"
        bad = "Every ferry has a heated passenger lounge. [S2]"
        b_ok, b_q = "Children under 12 travel for €6. [S2]", "What do children pay?"
        b_bad_q = "How quickly are weather refunds paid?"
        b_bad = "Refunds are paid within 14 days after a weather cancellation. [S4]"
    del fee
    return {
        "A": [
            {
                "id": f"A-{s}-001",
                "category": "A",
                "split": split,
                "subtype": "verbatim",
                "domain": domain,
                "question": q_cost,
                "passages": refs,
                "statement": f"{fee_text}. [S2]",
                "gold": {"supported": True, "supporting_labels": ["S2"]},
                "rationale": "R-A1 stated in S2.",
            },
            {
                "id": f"A-{s}-002",
                "category": "A",
                "split": split,
                "subtype": "absent",
                "domain": domain,
                "question": q_absent,
                "passages": refs,
                "statement": bad,
                "gold": {"supported": False, "supporting_labels": []},
                "rationale": "R-A2 not stated anywhere.",
            },
        ],
        "B": [
            {
                "id": f"B-{s}-001",
                "category": "B",
                "split": split,
                "subtype": "correct_single",
                "domain": domain,
                "question": b_q,
                "passages": refs,
                "statement": b_ok,
                "cited_labels": ["S2"],
                "gold": {"citation_correct": True, "supporting_labels": ["S2"]},
                "rationale": "R-B1 cites S2.",
            },
            {
                "id": f"B-{s}-002",
                "category": "B",
                "split": split,
                "subtype": "nonexistent_label",
                "domain": domain,
                "question": b_bad_q,
                "passages": refs,
                "statement": b_bad,
                "cited_labels": ["S4"],
                "gold": {"citation_correct": False, "supporting_labels": ["S3"]},
                "rationale": "R-B2 no S4.",
            },
        ],
        "C": [
            {
                "id": f"C-{s}-001",
                "category": "C",
                "split": split,
                "subtype": "correct_value",
                "domain": domain,
                "question": f"{q_cost.rstrip('?')} in total?",
                "passages": refs,
                "answer": f"{fee_text}. [S2]",
                "gold": {
                    "answer_kind": "correct_answer",
                    "answerable": True,
                    "question_type": "amount",
                },
                "rationale": "R-C1 gives the price.",
            },
            {
                "id": f"C-{s}-002",
                "category": "C",
                "split": split,
                "subtype": "correct_refusal",
                "domain": domain,
                "question": refusal_q,
                "passages": refs,
                "answer": "Insufficient evidence: the documents do not name this person.",
                "gold": {
                    "answer_kind": "correct_refusal",
                    "answerable": False,
                    "question_type": "person_or_role",
                },
                "rationale": "R-C2 nobody is named.",
            },
        ],
    }


def _d(split: str, domain: str) -> list[Any]:
    s = "D" if split == "dev" else "T"
    a, b = f"D-{s}-001-a", f"D-{s}-001-b"
    c, d = f"D-{s}-002-a", f"D-{s}-002-b"
    return [
        {
            "id": f"D-{s}-001",
            "category": "D",
            "split": split,
            "subtype": "plain_conflict",
            "domain": domain,
            "question": f"How much does a standard parking permit cost per month ({split})?",
            "passages": [
                {
                    "label": "S1",
                    "document_id": a,
                    "title": "Permit rules",
                    "text": "A standard parking permit costs €40 per month for every resident.",
                },
                {
                    "label": "S2",
                    "document_id": b,
                    "title": "Permit FAQ",
                    "text": "The standard parking permit costs €55 per month for every resident.",
                },
            ],
            "gold": {"disagreement": True, "kind": "conflict", "pair": ["S1", "S2"]},
            "rationale": "R-D1 40 vs 55.",
        },
        {
            "id": f"D-{s}-002",
            "category": "D",
            "split": split,
            "subtype": "agreement",
            "domain": domain,
            "question": f"How much is the key deposit for a storage locker ({split})?",
            "passages": [
                {
                    "label": "S1",
                    "document_id": c,
                    "title": "Locker rules",
                    "text": "The key deposit for a storage locker is €20, paid in cash.",
                },
                {
                    "label": "S2",
                    "document_id": d,
                    "title": "Locker FAQ",
                    "text": "A storage locker needs a key deposit of 20 euros, paid in cash.",
                },
            ],
            "gold": {"disagreement": False, "kind": "none", "pair": None},
            "rationale": "R-D2 same value.",
        },
    ]


FIXTURE_COUNTS = {
    "A": {"verbatim": (1, 1), "absent": (1, 1)},
    "B": {"correct_single": (1, 1), "nonexistent_label": (1, 1)},
    "C": {"correct_value": (1, 1), "correct_refusal": (1, 1)},
    "D": {"plain_conflict": (1, 1), "agreement": (1, 1)},
}


def _write_jsonl(path: Path, rows: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), "utf-8")


def make_suite(root: Path) -> Path:
    for split, document_id, body, prefix, domain in (
        ("dev", "dev-waste-1", DEV_DOC, "D", "municipal waste collection"),
        ("test", "test-ferry-1", TEST_DOC, "T", "ferry operator"),
    ):
        base = root / "documents" / split
        base.mkdir(parents=True, exist_ok=True)
        (base / f"{document_id}.md").write_text(body, encoding="utf-8")
        _title, _, rest = body.partition("\n")
        pool = _passages(document_id, rest, prefix)
        _write_jsonl(base / "passages.jsonl", pool)
        for category, rows in _abc(split, prefix, domain, pool).items():
            _write_jsonl(root / split / f"{category}.jsonl", rows)
        _write_jsonl(root / split / "D.jsonl", _d(split, domain))
    manifest = {"name": suite.SUITE_NAME, "status": "draft", **suite.hash_record(root)}
    (root / "manifest.json").write_text(suite.as_json(manifest), encoding="utf-8")
    return root


@pytest.fixture
def fixture_suite(tmp_path: Path) -> Path:
    return make_suite(tmp_path / "suite")


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "reliability_suite_cli", REPO_ROOT / "scripts" / "reliability_suite.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------- counts
def test_counts_match_the_specification_and_add_up() -> None:
    rows = re.findall(r"^\| ([ABCD]) \| (\w+) \| (\d+) \| (\d+) \|$", SPEC.read_text("utf-8"), re.M)
    from_spec: dict[str, dict[str, tuple[int, int]]] = {}
    for category, subtype, dev, test in rows:
        from_spec.setdefault(category, {})[subtype] = (int(dev), int(test))
    assert from_spec == suite.COUNTS
    dev = sum(d for subtypes in suite.COUNTS.values() for d, _t in subtypes.values())
    test = sum(t for subtypes in suite.COUNTS.values() for _d, t in subtypes.values())
    assert (dev, test, dev + test) == (198, 462, 660)
    totals = {c: tuple(map(sum, zip(*s.values(), strict=True))) for c, s in suite.COUNTS.items()}
    assert totals == {"A": (54, 126), "B": (36, 84), "C": (48, 112), "D": (60, 140)}
    for subtypes in suite.COUNTS.values():
        for dev_n, test_n in subtypes.values():
            assert dev_n * 7 == test_n * 3  # every subtype split exactly 30 / 70
    assert set(suite.A_SUPPORTED) == set(suite.COUNTS["A"])
    assert set(suite.B_CORRECT) == set(suite.COUNTS["B"])
    assert set(suite.C_KIND) == set(suite.COUNTS["C"])
    assert set(suite.D_KIND) == set(suite.COUNTS["D"])


# --------------------------------------------------------------- validation
def test_a_well_formed_suite_validates(fixture_suite: Path) -> None:
    issues, counts = suite.validate_suite(fixture_suite, FIXTURE_COUNTS)
    assert [issue for issue in issues if issue.severity == "error"] == []
    assert counts["test"]["D"] == {"plain_conflict": 1, "agreement": 1}


def _rewrite(path: Path, case_id: str, change: dict[str, Any]) -> None:
    rows = [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]
    for row in rows:
        if row["id"] == case_id:
            row.update(change)
    _write_jsonl(path, rows)


def test_the_validator_reports_problems_without_case_text(fixture_suite: Path) -> None:
    root = fixture_suite
    _rewrite(
        root / "test" / "A.jsonl",
        "A-T-001",
        {"gold": {"supported": False, "supporting_labels": []}},
    )
    _rewrite(root / "test" / "B.jsonl", "B-T-002", {"cited_labels": ["S3"]})
    _rewrite(
        root / "test" / "C.jsonl", "C-T-002", {"question": "What does an adult single ticket cost?"}
    )
    d_rows = [
        json.loads(line) for line in (root / "test" / "D.jsonl").read_text("utf-8").splitlines()
    ]
    d_rows[0]["passages"][1]["document_id"] = d_rows[0]["passages"][0]["document_id"]
    _write_jsonl(root / "test" / "D.jsonl", d_rows)
    pool = root / "documents" / "test" / "passages.jsonl"
    pool.write_text(pool.read_text("utf-8").replace("06:15", "06:20"), encoding="utf-8")
    issues, _ = suite.validate_suite(root, FIXTURE_COUNTS)
    codes = {(issue.code, issue.where) for issue in issues if issue.severity == "error"}
    assert ("gold_subtype_mismatch", "A-T-001") in codes
    assert ("cited_labels", "B-T-002") in codes
    assert ("duplicate_question", "A-T-001") in codes or ("duplicate_question", "C-T-002") in codes
    assert ("documents", "D-T-001") in codes
    assert any(code == "passage_not_verbatim" for code, _ in codes)
    assert any(code == "hash_mismatch" for code, _ in codes)
    printed = "\n".join(str(issue) for issue in issues)
    for secret in ("adult single ticket", "06:20", "R-A1", "Permit rules"):
        assert secret not in printed


# ------------------------------------------------------------------ scoring
def test_version_one_predictions_on_hand_written_cases(fixture_suite: Path) -> None:
    cases = suite.load_cases(fixture_suite, "test")
    predictions, outcomes = suite.run_checks(cases, suite.CHECKS["v1"])
    assert predictions["A-T-001"] == {"flagged": False}
    assert predictions["A-T-002"] == {"flagged": True}
    assert predictions["B-T-001"] == {"judged_correct": True}
    assert predictions["B-T-002"] == {"judged_correct": False}  # [S4] is not a passage
    assert predictions["C-T-002"]["refusal"] is True
    assert predictions["C-T-001"] == {"flagged": False, "refusal": False}
    assert outcomes["D-T-001"]["detected"] is True
    assert outcomes["D-T-002"]["reported"] is False
    assert all(outcome["correct"] for outcome in outcomes.values())


def test_measures_intervals_and_paired_test() -> None:
    low, high = suite.wilson(85, 100) or (0.0, 0.0)
    assert (round(low, 3), round(high, 3)) == (0.767, 0.907)
    assert suite.wilson(0, 0) is None
    assert suite.mcnemar_exact(0, 0) == 1.0
    assert suite.mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert suite.mcnemar_exact(3, 3) == 1.0
    high_ok = suite.measure("A.unsupported_recall", 57, 63)
    assert (high_ok["met"], high_ok["established"]) == (True, True)
    just = suite.measure("A.unsupported_recall", 56, 63)
    assert (just["met"], just["established"]) == (True, False)
    low_rate = suite.measure("D.false_alarm_rate", 6, 63)
    assert (low_rate["met"], low_rate["stretch_met"], low_rate["established"]) == (True, True, True)
    outcomes = [
        {
            "category": "A",
            "subtype": "paraphrase",
            "positive": False,
            "flagged": True,
            "correct": False,
        },
        {
            "category": "A",
            "subtype": "paraphrase",
            "positive": False,
            "flagged": False,
            "correct": True,
        },
        {"category": "A", "subtype": "absent", "positive": True, "flagged": True, "correct": True},
    ]
    summary = suite.summarize(outcomes)["A"]
    assert summary["paraphrase_acceptance"]["k"] == 1
    assert summary["unsupported_precision"]["value"] == 0.5
    before = {"x": {"category": "A", "correct": False}, "y": {"category": "A", "correct": True}}
    after = {"x": {"category": "A", "correct": True}, "y": {"category": "A", "correct": True}}
    assert suite.paired(before, after)["A"] == {"cases": 2, "fixed": 1, "broke": 0, "p_value": 1.0}


# -------------------------------------------------------------------- audit
def test_the_blind_audit_page_shows_no_label_and_the_comparison_counts(
    fixture_suite: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _script()
    monkeypatch.setattr(cli, "AUDIT_SAMPLE", {"A": 2, "B": 2, "C": 2, "D": 2})
    out = tmp_path / "audit"
    assert cli.main(["audit-prepare", "--root", str(fixture_suite), "--out", str(out)]) == 0
    page = (out / "blind_audit.html").read_text("utf-8")
    for leak in (
        "R-A1",
        "R-D2",
        "A-T-001",
        "D-T-002",
        "verbatim",
        "nonexistent_label",
        "rationale",
    ):
        assert leak not in page
    sample = json.loads((out / "audit_sample.json").read_text("utf-8"))
    cases = {case.id: case for case in suite.load_cases(fixture_suite, "test")}
    answers = {n: {"label": suite.gold_label(cases[i])} for n, i in sample["numbers"].items()}
    first = next(iter(answers))
    first_case = cases[sample["numbers"][first]]
    answers[first] = {
        "label": next(
            value
            for value, _text in suite.AUDIT_LABELS[first_case.category]
            if value != suite.gold_label(first_case)
        )
    }
    (tmp_path / "answers.json").write_text(json.dumps(answers), "utf-8")
    capsys.readouterr()
    assert (
        cli.main(
            [
                "audit-compare",
                "--root",
                str(fixture_suite),
                "--sample",
                str(out / "audit_sample.json"),
                "--answers",
                str(tmp_path / "answers.json"),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert "blind agreement: 7 of 8" in capsys.readouterr().out
    result = json.loads((out / "audit_result.json").read_text("utf-8"))
    assert [item["number"] for item in result["disagreements"]] == [first]
    assert result["disagreements"][0]["reason"] == "different label"
    assert result["answers"]["missing"] == []
    assert "Proposed label" in (out / "reveal.html").read_text("utf-8")
    (tmp_path / "decisions.json").write_text(json.dumps({first: {"label": "replace"}}), "utf-8")
    assert (
        cli.main(
            [
                "audit-decide",
                "--result",
                str(out / "audit_result.json"),
                "--decisions",
                str(tmp_path / "decisions.json"),
            ]
        )
        == 0
    )
    applied = json.loads((out / "audit_decisions_applied.json").read_text("utf-8"))
    assert applied["replace"] == [sample["numbers"][first]]


def test_saved_answers_are_matched_to_the_sample_before_comparing(tmp_path: Path) -> None:
    cli = _script()
    categories = {"1": "A", "2": "B", "3": "C", "4": "D"}
    path = tmp_path / "answers.json"

    def answers(text: str) -> tuple[dict[str, str], dict[str, Any]]:
        path.write_text(text, "utf-8")
        result: tuple[dict[str, str], dict[str, Any]] = cli.check_answers(path, categories)
        return result

    labels, summary = answers(
        '{"1": {"label": "supported", "note": "private note"}, "2": {"label": null, "note": ""},'
        ' "3": {"label": "conflict", "note": ""}, "4": {"label": "  ", "note": ""}}'
    )
    assert labels == {"1": "supported"}
    assert (summary["entries"], summary["answered"]) == (4, 1)
    assert summary["missing"] == ["2", "4"]
    assert summary["invalid"] == ["3"]  # "conflict" is a group D label, not a group C one
    assert "private note" not in json.dumps(summary)
    labels, summary = answers('{"1": "not_supported", "2": {"label": "wrong", "note": ""}}')
    assert labels == {"1": "not_supported", "2": "wrong"}
    assert summary["missing"] == ["3", "4"]
    for incompatible in (
        '{"1": {"label": "supported"}, "1": {"label": "not_supported"}}',
        '{"1": {"label": "supported"}, "51": {"label": "supported"}}',
        '[{"label": "supported"}]',
        "not json",
    ):
        with pytest.raises(cli.AnswersError):
            answers(incompatible)


def test_the_comparison_refuses_answers_that_do_not_fit_the_sample(
    fixture_suite: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _script()
    monkeypatch.setattr(cli, "AUDIT_SAMPLE", {"A": 2, "B": 2, "C": 2, "D": 2})
    out = tmp_path / "audit"
    assert cli.main(["audit-prepare", "--root", str(fixture_suite), "--out", str(out)]) == 0
    compare = [
        "audit-compare",
        "--root",
        str(fixture_suite),
        "--sample",
        str(out / "audit_sample.json"),
        "--answers",
        str(tmp_path / "answers.json"),
        "--out",
        str(out),
    ]
    (tmp_path / "answers.json").write_text('{"9": {"label": "supported"}}', "utf-8")
    assert cli.main(compare) == 2
    assert not (out / "audit_result.json").exists()
    sample = json.loads((out / "audit_sample.json").read_text("utf-8"))
    cases = {case.id: case for case in suite.load_cases(fixture_suite, "test")}
    answers = {n: {"label": suite.gold_label(cases[i])} for n, i in sample["numbers"].items()}
    answers["1"] = {"label": None}
    # In group D "none" is a real label ("No disagreement"); a missing choice must not read as it.
    d_number = next(n for n, i in sample["numbers"].items() if cases[i].category == "D")
    answers[d_number] = {"label": None}
    (tmp_path / "answers.json").write_text(json.dumps(answers), "utf-8")
    assert cli.main(compare) == 0
    result = json.loads((out / "audit_result.json").read_text("utf-8"))
    assert (result["agreed"], result["total"]) == (6, 8)
    assert result["answers"]["missing"] == sorted(["1", d_number], key=int)
    assert {item["reason"] for item in result["disagreements"]} == {"no valid blind label"}
    reveal = (out / "reveal.html").read_text("utf-8")
    d_section = reveal.split(f'data-n="{d_number}"')[1].split("</section>")[0]
    assert "<b>Your blind label:</b> (no valid label chosen)" in d_section
    assert "<b>Your blind label:</b> No disagreement" not in reveal
    # A sample drawn from different test content is refused.
    changed = fixture_suite / "test" / "A.jsonl"
    changed.write_text(changed.read_text("utf-8").replace("R-A1", "R-A1 edited"), "utf-8")
    assert cli.main(compare) == 2


def test_a_later_round_scores_the_random_sample_apart_from_included_cases(
    fixture_suite: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _script()
    monkeypatch.setattr(cli, "AUDIT_SAMPLE", {"A": 1, "B": 1, "C": 1, "D": 0})
    (tmp_path / "exclude.json").write_text('["A-T-001"]', "utf-8")
    (tmp_path / "include.json").write_text('["D-T-001"]', "utf-8")
    out = tmp_path / "round2"
    prepare = ["audit-prepare", "--root", str(fixture_suite), "--out", str(out), "--seed", "7"]
    options = ["--round", "2", "--exclude", str(tmp_path / "exclude.json")]
    assert cli.main([*prepare, *options, "--include", str(tmp_path / "include.json")]) == 0
    page = (out / "blind_audit.html").read_text("utf-8")
    assert "round 2" in page
    assert "blind_answers_round2.json" in page
    assert "D-T-001" not in page
    sample = json.loads((out / "audit_sample.json").read_text("utf-8"))
    assert "A-T-001" not in sample["numbers"].values()
    assert [sample["numbers"][n] for n in sample["included"]] == ["D-T-001"]
    cases = {case.id: case for case in suite.load_cases(fixture_suite, "test")}
    answers = {n: {"label": suite.gold_label(cases[i])} for n, i in sample["numbers"].items()}
    answers[sample["included"][0]] = {"label": "none"}  # the included case's proposed: conflict
    (tmp_path / "answers.json").write_text(json.dumps(answers), "utf-8")
    capsys.readouterr()
    compare = ["audit-compare", "--root", str(fixture_suite), "--out", str(out)]
    answers_file = str(tmp_path / "answers.json")
    files = ["--sample", str(out / "audit_sample.json"), "--answers", answers_file]
    assert cli.main([*compare, *files]) == 0
    printed = capsys.readouterr().out
    assert "blind agreement, random sample: 3 of 3" in printed
    assert "each of which must agree: 0 of 1" in printed
    result = json.loads((out / "audit_result.json").read_text("utf-8"))
    assert (result["agreed"], result["total"]) == (3, 3)
    assert (result["included"]["agreed"], result["included"]["total"]) == (0, 1)
    assert result["disagreements"] == [
        {
            "number": sample["included"][0],
            "id": "D-T-001",
            "reason": "different label",
            "included": True,
        }
    ]
    (tmp_path / "include.json").write_text('["Z-T-001"]', "utf-8")
    assert cli.main([*prepare, "--include", str(tmp_path / "include.json")]) == 2


def test_replacements_keep_the_subtype_and_are_logged(
    fixture_suite: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli = _script()
    root = fixture_suite
    rows = {
        row["id"]: row
        for path in (root / "test" / "C.jsonl", root / "test" / "D.jsonl")
        for row in (json.loads(line) for line in path.read_text("utf-8").splitlines())
    }
    new_c = dict(rows["C-T-001"], question="What is the price of one adult single ticket?")
    bad_d = dict(rows["D-T-002"], subtype="plain_conflict")
    replace = ["replace-cases", "--root", str(root), "--reason", "round 1: not confirmed"]
    cases_file = tmp_path / "replacements.jsonl"
    before = {p: p.read_bytes() for p in (root / "test").glob("*.jsonl")}
    cases_file.write_text(json.dumps(new_c) + "\n" + json.dumps(bad_d) + "\n", "utf-8")
    assert cli.main([*replace, "--cases", str(cases_file)]) == 2
    assert {p: p.read_bytes() for p in before} == before  # a refused run changes nothing
    assert not (root / "REVISIONS.jsonl").exists()
    unknown = dict(rows["C-T-002"], id="C-T-099")
    cases_file.write_text(json.dumps(unknown) + "\n", "utf-8")
    assert cli.main([*replace, "--cases", str(cases_file)]) == 2
    cases_file.write_text(json.dumps(new_c) + "\n", "utf-8")
    capsys.readouterr()
    assert cli.main([*replace, "--cases", str(cases_file)]) == 0
    assert "adult single ticket" not in capsys.readouterr().out
    c_file, log = root / "test" / "C.jsonl", root / "REVISIONS.jsonl"
    replaced = [json.loads(line) for line in c_file.read_text("utf-8").splitlines()]
    assert [row["question"] for row in replaced if row["id"] == "C-T-001"] == [new_c["question"]]
    assert [row["id"] for row in replaced] == ["C-T-001", "C-T-002"]
    (entry,) = [json.loads(line) for line in log.read_text("utf-8").splitlines()]
    assert (entry["id"], entry["subtype"], entry["reason"]) == (
        "C-T-001",
        "correct_value",
        "round 1: not confirmed",
    )
    assert entry["old_sha256"] != entry["new_sha256"]
    issues, _ = suite.validate_suite(root, FIXTURE_COUNTS)
    assert {issue.code for issue in issues if issue.severity == "error"} == {"hash_mismatch"}
    assert cli.main(["rehash", "--root", str(root)]) == 0
    issues, _ = suite.validate_suite(root, FIXTURE_COUNTS)
    assert [issue for issue in issues if issue.severity == "error"] == []


def test_the_test_split_is_scored_only_when_frozen_and_once(
    fixture_suite: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli = _script()
    out = tmp_path / "reports"
    args = [
        "score",
        "--split",
        "test",
        "--checks",
        "v1",
        "--root",
        str(fixture_suite),
        "--out",
        str(out),
    ]
    assert cli.main(args) == 2  # not frozen
    manifest = json.loads((fixture_suite / "manifest.json").read_text("utf-8"))
    manifest["status"] = "frozen"
    (fixture_suite / "manifest.json").write_text(suite.as_json(manifest), "utf-8")
    monkeypatch.setattr(suite, "FROZEN_TEST_SHA256", manifest["test_sha256"])
    capsys.readouterr()
    assert cli.main(args) == 0
    printed = capsys.readouterr().out
    for secret in ("adult single ticket", "R-A1", "Permit rules", "A-T-001"):
        assert secret not in printed
    assert cli.main(args) == 2  # once per version of the checks
    assert cli.main([*args, "--rerun-reason", "test of the override"]) == 0
    log = (out / "test-runs.jsonl").read_text("utf-8").splitlines()
    assert len(log) == 2
    assert json.loads(log[1])["rerun_reason"] == "test of the override"
    (sealed,) = sorted((out / "sealed").glob("*.json"))[:1]
    outcomes = json.loads(sealed.read_text("utf-8"))["outcomes"]
    assert all(set(value) == {"category", "correct"} for value in outcomes.values())
    # A changed held-out file is refused, whatever the reason given.
    changed = fixture_suite / "test" / "A.jsonl"
    changed.write_text(changed.read_text("utf-8").replace("R-A1", "R-A1 edited"), "utf-8")
    assert cli.main([*args, "--rerun-reason", "after an edit"]) == 2
