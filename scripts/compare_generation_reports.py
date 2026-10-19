"""Compare answer-evaluation reports on the same questions, with paired bootstrap intervals.

    python scripts/compare_generation_reports.py BASELINE.json OTHER.json [MORE.json ...] \
        [--markdown out.md] [--json out.json]

Every report is re-summarised with the current metric definitions (scoring version 3) from
its stored per-question rows, so older reports can be compared with newer ones:

* an answer whose first sentence is a refusal ("The sources do not provide ... [S5]") is
  re-scored as a refusal (``citations.opens_with_refusal``), using the stored text; the
  number of re-scored rows is printed;
* rows from reports before scoring version 3 lack some fields: malformed citations and
  conflict statements are recomputed from the stored answer text; sentence-level support
  counts were not stored, so evidence support is reported as missing for those reports.

Paired differences (other minus baseline) are computed over the questions both reports
scored, with a 95% percentile bootstrap interval (2000 resamples, seed 13). A difference is
called "distinguishable" only when the interval excludes zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from groundedai.evaluation.generation_eval import summarize_rows  # noqa: E402
from groundedai.evaluation.metrics import paired_bootstrap_difference  # noqa: E402
from groundedai.generation.citations import (  # noqa: E402
    malformed_citations,
    opens_with_refusal,
    states_conflict,
)

SUMMARY_FIELDS = [
    ("n", "Questions"),
    ("answered", "Answered"),
    ("abstained", "Abstained"),
    ("invalid_outputs", "Invalid output (markers only or empty)"),
    ("errors", "Errors"),
    ("timeouts", "Timeouts"),
    ("behavior_accuracy", "Correct behaviour (scored questions)"),
    ("answer_rate_answerable", "Answer rate, answerable questions"),
    ("incorrect_refusal_rate", "Incorrect refusal rate, answerable questions"),
    ("abstention_recall", "Correct abstention rate (should-refuse questions)"),
    ("abstention_precision", "Abstention precision"),
    ("conflict_detection_rate", "Conflict questions: both sides cited"),
    ("conflict_stated_rate", "Conflict questions: disagreement stated"),
    ("key_fact_recall", "Key-fact recall"),
    ("citation_format_compliance", "Citation-format compliance (answers)"),
    ("citation_validity", "Citation validity (answers)"),
    ("evidence_support_rate", "Evidence support (cited sentences)"),
    ("evidence_coverage", "Evidence coverage"),
    ("malformed_citation_rate", "Answers with malformed citations"),
    ("invalid_output_rate", "Invalid output rate"),
    ("injection_resistance", "Injection resistance"),
]

PAIRED = [
    ("behavior_ok", "Correct behaviour"),
    ("answered_when_answerable", "Answered (answerable questions)"),
    ("refused_when_answerable", "Refused (answerable questions)"),
    ("key_fact_recall", "Key-fact recall"),
]


def load(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if report.get("kind") != "generation" or "per_query" not in report:
        raise SystemExit(f"{path} is not an answer-evaluation report with per-question rows")
    version = int((report.get("run_config") or {}).get("scoring_version", 0))
    rows = [enrich(row, version) for row in report["per_query"]]
    report["_rows"] = rows
    report["_summary"] = summarize_rows(rows)
    report["_scoring_version"] = version
    return report


def enrich(row: dict[str, Any], version: int) -> dict[str, Any]:
    """Fill scoring-version-3 fields for older rows and apply the current refusal rule."""
    row = dict(row)
    answer = row.get("answer") or ""
    if row.get("status") == "answered" and opens_with_refusal(row.get("model_output") or answer):
        # Scored before the rule "a first sentence that is a refusal makes the reply a
        # refusal, even with a citation" existed (citations.opens_with_refusal).
        row["status"] = "insufficient_evidence"
        row["rescored_as_refusal"] = True
        expected = row.get("expected_behavior")
        if expected in ("answer", "abstain", "flag_conflict"):
            row["behavior_ok"] = expected == "abstain"
        if row.get("key_fact_recall") is not None:
            row["key_fact_recall"] = 0.0
        if row.get("evidence_coverage") is not None:
            row["evidence_coverage"] = 0.0
        row["token_f1"] = None
    answered = row.get("status") == "answered"
    if version < 3:
        row["malformed_citations"] = len(malformed_citations(answer)) if answered else 0
        row["conflict_stated"] = states_conflict(answer) if answered else False
        row.setdefault("cited_sentences", None)
        row.setdefault("supported_sentences", None)
    elif not answered:
        row["malformed_citations"] = 0
        row["conflict_stated"] = False
    answerable = row.get("expected_behavior") in ("answer", "flag_conflict")
    row["answered_when_answerable"] = (row.get("status") == "answered") if answerable else None
    row["refused_when_answerable"] = (
        (row.get("status") == "insufficient_evidence") if answerable else None
    )
    return row


def label(report: dict[str, Any]) -> str:
    provider = report.get("provider") or {}
    config = report.get("run_config") or {}
    prompt = report.get("prompt_version") or config.get("prompt_version", "answer-v1")
    fingerprint = config.get("prompt_fingerprint")
    return f"{provider.get('model', '?')} · {prompt}" + (f" ({fingerprint})" if fingerprint else "")


def paired(base: dict[str, Any], other: dict[str, Any], key: str) -> dict[str, Any] | None:
    a = {row["question_id"]: row.get(key) for row in base["_rows"]}
    b = {row["question_id"]: row.get(key) for row in other["_rows"]}
    common = [qid for qid in a if qid in b and a[qid] is not None and b[qid] is not None]
    if len(common) < 2:
        return None
    result = paired_bootstrap_difference(
        [float(b[qid]) for qid in common], [float(a[qid]) for qid in common], seed=13
    )
    return {"n": len(common), **result}


def fmt(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("reports", nargs="+", type=Path, help="baseline first")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    reports = [load(path) for path in args.reports]
    base = reports[0]

    lines = ["| Metric | " + " | ".join(label(r) for r in reports) + " |"]
    lines.append("|---|" + "---|" * len(reports))
    for key, name in SUMMARY_FIELDS:
        lines.append(
            f"| {name} | " + " | ".join(fmt(r["_summary"].get(key)) for r in reports) + " |"
        )
    latency = " | ".join(
        f"{(r['_summary']['latency_ms'] or {}).get('p50', 0) / 1000:.1f} s / "
        f"{(r['_summary']['latency_ms'] or {}).get('p95', 0) / 1000:.1f} s"
        for r in reports
    )
    lines.append(f"| Latency p50 / p95 | {latency} |")
    rescored = " | ".join(
        str(sum(1 for row in r["_rows"] if row.get("rescored_as_refusal"))) for r in reports
    )
    lines.append(f"| Answers re-scored as refusals (current parser) | {rescored} |")

    comparisons = []
    for other in reports[1:]:
        lines += ["", f"Paired differences, {label(other)} minus {label(base)}:", ""]
        lines += [
            "| Measure | n | Difference | 95% interval | Distinguishable |",
            "|---|---|---|---|---|",
        ]
        for key, name in PAIRED:
            result = paired(base, other, key)
            comparisons.append(
                {"baseline": label(base), "other": label(other), "measure": key, **(result or {})}
            )
            if result is None:
                lines.append(f"| {name} | – | – | – | – |")
                continue
            lines.append(
                f"| {name} | {result['n']} | {result['mean_diff']:+.3f} | "
                f"[{result['ci95_low']:+.3f}, {result['ci95_high']:+.3f}] | "
                f"{'yes' if result['distinguishable'] else 'no'} |"
            )

    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8", newline="\n")
    if args.json:
        payload = {
            "reports": [
                {
                    "label": label(r),
                    "report_id": r.get("report_id"),
                    "scoring_version": r["_scoring_version"],
                    "summary": r["_summary"],
                }
                for r in reports
            ],
            "paired": comparisons,
        }
        args.json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
