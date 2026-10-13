"""Score the disagreement checks on the held-out cases in groundedai-conflicts-v1.

    python scripts/evaluate_conflict_cases.py [--out evaluation/reports/conflicts]

Each case is a question and 2-3 passages from different documents, given to the checks exactly
as retrieved passages would be. No retrieval and no language model are involved. The case file
must match the SHA-256 recorded in the dataset manifest; the script refuses to run otherwise.

Both checks are scored: ``disagreement-v1`` (current) and ``figures-v1`` (published on
2026-10-13, which has no notion of dates). A case counts as

* **found** when a disagreement between the two expected passages is reported,
* **classified** when, in addition, its kind is right (``conflict`` or ``different_dates``),
* a **false alarm** when anything is reported for a case that has no disagreement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from groundedai.evaluation.environment import code_version  # noqa: E402
from groundedai.generation.conflicts import find_figure_conflicts  # noqa: E402
from groundedai.generation.context import assemble_context  # noqa: E402
from groundedai.generation.disagreement import find_disagreements  # noqa: E402
from groundedai.retrieval.types import ChunkScores, RetrievedChunk  # noqa: E402

DATASET = REPO_ROOT / "evaluation" / "datasets" / "groundedai-conflicts-v1"
EXPECTED_KIND = {"conflict": "conflict", "temporal": "different_dates"}


def _load_cases() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    data = (DATASET / manifest["cases_file"]).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["cases_sha256"]:
        raise SystemExit(f"cases.jsonl was changed (sha256 {digest}); refusing to score it")
    cases = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    return manifest, cases


def _context(case: dict[str, Any]) -> Any:
    chunks = [
        RetrievedChunk(
            f"{case['id']}-{index}",
            passage["doc"],
            passage["title"],
            0,
            passage["text"],
            0,
            len(passage["text"]),
            None,
            None,
            ChunkScores(),
        )
        for index, passage in enumerate(case["passages"])
    ]
    return assemble_context(chunks, max_chars=6000)


def _score_case(case: dict[str, Any]) -> dict[str, Any]:
    context = _context(case)
    expected_pair = sorted(case.get("expected_pair", []))
    current = find_disagreements(case["question"], context.sources)
    published = find_figure_conflicts(case["question"], context.sources)
    current_found = [{"kind": item.kind, "labels": sorted(item.labels)} for item in current]
    published_found = [sorted((item.label_a, item.label_b)) for item in published]
    on_pair = [item for item in current_found if item["labels"] == expected_pair]
    result: dict[str, Any] = {
        "id": case["id"],
        "class": case["class"],
        "question": case["question"],
        "expected_pair": expected_pair or None,
        "disagreement_v1": current_found,
        "figures_v1": published_found,
    }
    if case["class"] in EXPECTED_KIND:
        result["found"] = bool(on_pair)
        result["classified"] = any(item["kind"] == EXPECTED_KIND[case["class"]] for item in on_pair)
        result["figures_v1_found"] = expected_pair in published_found
    else:
        result["false_alarm"] = bool(current_found)
        result["figures_v1_false_alarm"] = bool(published_found)
    return result


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    disagreeing = [r for r in results if r["class"] in EXPECTED_KIND]
    others = [r for r in results if r["class"] not in EXPECTED_KIND]
    by_class: dict[str, dict[str, int]] = {}
    for name in ("conflict", "temporal", "context", "agreement", "incomplete"):
        rows = [r for r in results if r["class"] == name]
        if name in EXPECTED_KIND:
            by_class[name] = {
                "cases": len(rows),
                "found": sum(1 for r in rows if r["found"]),
                "classified": sum(1 for r in rows if r["classified"]),
                "figures_v1_found": sum(1 for r in rows if r["figures_v1_found"]),
            }
        else:
            by_class[name] = {
                "cases": len(rows),
                "false_alarms": sum(1 for r in rows if r["false_alarm"]),
                "figures_v1_false_alarms": sum(1 for r in rows if r["figures_v1_false_alarm"]),
            }
    return {
        "cases": len(results),
        "disagreement_cases": len(disagreeing),
        "found": sum(1 for r in disagreeing if r["found"]),
        "classified": sum(1 for r in disagreeing if r["classified"]),
        "no_disagreement_cases": len(others),
        "false_alarms": sum(1 for r in others if r["false_alarm"]),
        "figures_v1_found": sum(1 for r in disagreeing if r["figures_v1_found"]),
        "figures_v1_false_alarms": sum(1 for r in others if r["figures_v1_false_alarm"]),
        "by_class": by_class,
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        f"# {report['title']}",
        "",
        f"Generated {report['generated_at']} · code `{report['code_version']}` · cases "
        f"`{report['dataset']['cases_sha256'][:12]}`",
        "",
        "| Check | Disagreements found (of "
        f"{summary['disagreement_cases']}) | Kind also right | False alarms (of "
        f"{summary['no_disagreement_cases']}) |",
        "|---|---|---|---|",
        f"| disagreement-v1 (current) | {summary['found']} | {summary['classified']} | "
        f"{summary['false_alarms']} |",
        f"| figures-v1 (published 2026-10-13) | {summary['figures_v1_found']} | – | "
        f"{summary['figures_v1_false_alarms']} |",
        "",
        "| Class | Cases | disagreement-v1 | figures-v1 |",
        "|---|---|---|---|",
    ]
    for name, item in summary["by_class"].items():
        if "found" in item:
            current = f"found {item['found']}, kind right {item['classified']}"
            published = f"found {item['figures_v1_found']}"
        else:
            current = f"false alarms {item['false_alarms']}"
            published = f"false alarms {item['figures_v1_false_alarms']}"
        lines.append(f"| {name} | {item['cases']} | {current} | {published} |")
    lines += [
        "",
        "## Cases",
        "",
        "| Case | Class | disagreement-v1 | figures-v1 |",
        "|---|---|---|---|",
    ]
    for row in report["cases"]:
        current = ", ".join(f"{d['kind']} {'/'.join(d['labels'])}" for d in row["disagreement_v1"])
        published = ", ".join("/".join(pair) for pair in row["figures_v1"])
        lines.append(f"| {row['id']} | {row['class']} | {current or '–'} | {published or '–'} |")
    lines += ["", "## Limitations", ""] + [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def _describe(version: dict[str, Any] | None) -> str:
    if not version:
        return "unknown"
    dirty = " (uncommitted changes)" if version.get("uncommitted_changes") else ""
    return f"{version.get('commit')}{dirty}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "evaluation" / "reports" / "conflicts"
    )
    args = parser.parse_args()
    manifest, cases = _load_cases()
    results = [_score_case(case) for case in cases]
    generated = datetime.now(UTC)
    version = code_version()
    report = {
        "report_id": f"conflict-cases-{generated:%Y%m%dT%H%M%SZ}",
        "kind": "conflict_cases",
        "title": "Disagreement check on the held-out cases (groundedai-conflicts-v1)",
        "generated_at": generated.isoformat(timespec="seconds"),
        "code_version": _describe(version),
        "dataset": {
            "name": manifest["name"],
            "version": manifest["version"],
            "cases_sha256": manifest["cases_sha256"],
            "classes": dict(Counter(case["class"] for case in cases)),
        },
        "summary": _summary(results),
        "cases": results,
        "limitations": [
            "30 cases written by the same author as the check; small counts, no intervals.",
            "The cases test the check on its own, with the passages given; retrieval is not "
            "involved.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / report["report_id"]
    base.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n"
    )
    base.with_suffix(".md").write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(_markdown(report))
    print("Written:", base.with_suffix(".json"), base.with_suffix(".md"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
