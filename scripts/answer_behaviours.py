"""Split answer-evaluation reports into separate behaviours, side by side (no model needed).

    python scripts/answer_behaviours.py REPORT.json [REPORT.json ...]

Reads the per-question rows of reports written by ``groundedai eval generation`` (scoring
version 5) or ``scripts/revalidate_recorded_answers.py`` and prints one Markdown table with a
column per report. Every value is a count ``k/n`` from the report's own fields; nothing is
re-checked. "Correct" means the answer contains every key fact of the question; a correct
answer can still contain other, unsupported statements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MATCHED = ("claims_supported", "claims_wrong_source", "claims_added_source")


def _n(row: dict[str, Any], key: str) -> int:
    return int(row.get(key) or 0)


def _frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def behaviours(report: dict[str, Any]) -> dict[str, str]:
    rows = report["per_query"]
    answerable = [r for r in rows if r["expected_behavior"] in ("answer", "flag_conflict")]
    abstain = [r for r in rows if r["expected_behavior"] == "abstain"]
    answered = [r for r in rows if r["status"] == "answered"]
    correct = [
        r for r in answerable if r["status"] == "answered" and (r["key_fact_recall"] or 0) >= 1
    ]
    statements = sum(_n(r, "claims_total") for r in answered)
    not_found = [_n(r, "claims_unsupported") + _n(r, "claims_partial") for r in answered]
    attribution = [
        r["key_fact_attribution"] for r in answered if r["key_fact_attribution"] is not None
    ]
    conflict = [r for r in rows if r["expected_behavior"] == "flag_conflict"]
    other = [r for r in rows if r["expected_behavior"] != "flag_conflict"]
    return {
        "claim check": str(report["run_config"].get("claim_check")),
        "1. correct answers (every key fact)": _frac(len(correct), len(answerable)),
        "2. false refusals": _frac(
            sum(r["status"] == "insufficient_evidence" for r in answerable), len(answerable)
        ),
        "3. correct refusals": _frac(
            sum(r["status"] == "insufficient_evidence" for r in abstain), len(abstain)
        ),
        "   unanswerable questions answered": _frac(
            sum(r["status"] == "answered" for r in abstain), len(abstain)
        ),
        "4. statements not found or partly found": _frac(sum(not_found), statements),
        "   answers with such a statement": _frac(sum(k > 0 for k in not_found), len(answered)),
        "5. correct answers not fully verified": _frac(
            sum(r["verdict"] != "verified" for r in correct), len(correct)
        ),
        "6. statements with an exact quote": _frac(
            sum(sum(_n(r, key) for key in MATCHED) for r in answered), statements
        ),
        "7. key facts quoted from the expected document (mean)": (
            f"{sum(attribution) / len(attribution):.2f}" if attribution else "-"
        ),
        "   statements quoted from a passage the model did not cite": _frac(
            sum(_n(r, "claims_wrong_source") + _n(r, "claims_added_source") for r in answered),
            statements,
        ),
        "8. answers citing a label that does not exist": _frac(
            sum(bool(r["invalid_citations"]) for r in answered), len(answered)
        ),
        "   answers with malformed citations": _frac(
            sum(_n(r, "malformed_citations") > 0 for r in answered), len(answered)
        ),
        "   replies without answer text (invalid output)": _frac(
            sum(r["status"] == "invalid_output" for r in rows), len(rows)
        ),
        "9. conflict questions with a conflict report": _frac(
            sum(bool(r["conflicts"]) for r in conflict), len(conflict)
        ),
        "   conflict stated in the model's answer": _frac(
            sum(bool(r["conflict_stated"]) for r in conflict), len(conflict)
        ),
        "10. conflict reports on other questions": _frac(
            sum(bool(r["false_conflict"]) for r in other), len(other)
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    columns = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    tables = [behaviours(report) for report in columns]
    print("| Behaviour | " + " | ".join(r["report_id"].rsplit("-", 1)[-1] for r in columns) + " |")
    print("|---|" + "---|" * len(columns))
    for key in tables[0]:
        print(f"| {key} | " + " | ".join(table[key] for table in tables) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
