"""Compare two rag_eval reports on the same questions, with paired bootstrap intervals.

    python scripts/compare_rag_reports.py OLD.json NEW.json [--out comparison.json]

For each measure, the per-question difference (new - old) is averaged with a 95% percentile
bootstrap interval over questions (5,000 resamples, seed 13). Measures: answer correctness and
false refusals on the questions that should be answered, refusals on the questions that should
be refused, and retrieval recall@5 / MRR@10 where both reports have them. Questions are matched
by id; a question missing from either report is left out and counted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.evaluation.metrics import paired_bootstrap_difference  # noqa: E402

SAMPLES = 5000
SEED = 13


def _rows(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return report, {row["question_id"]: row for row in report["per_question"]}


def _paired(old: list[float], new: list[float]) -> dict[str, Any]:
    if not old:  # no question of this kind (for example in a limited run)
        return {"n": 0, "old_mean": None, "new_mean": None, "gained": 0, "lost": 0}
    result: dict[str, Any] = {
        "n": len(old),
        "old_mean": round(sum(old) / len(old), 4),
        "new_mean": round(sum(new) / len(new), 4),
        "gained": sum(1 for a, b in zip(old, new, strict=True) if b > a),
        "lost": sum(1 for a, b in zip(old, new, strict=True) if b < a),
    }
    if len(old) >= 2:
        diff = paired_bootstrap_difference(new, old, samples=SAMPLES, seed=SEED)
        result.update({k: round(v, 4) if isinstance(v, float) else v for k, v in diff.items()})
    return result


def compare(old_path: Path, new_path: Path) -> dict[str, Any]:
    old_report, old = _rows(old_path)
    new_report, new = _rows(new_path)
    shared = [qid for qid in old if qid in new]
    answer = [q for q in shared if old[q]["expected_behavior"] == "answer"]
    abstain = [q for q in shared if old[q]["expected_behavior"] == "abstain"]
    measures: dict[str, Any] = {
        "correct (should answer)": _paired(
            [float(bool(old[q]["correct"])) for q in answer],
            [float(bool(new[q]["correct"])) for q in answer],
        ),
        "false refusal (should answer)": _paired(
            [float(bool(old[q]["abstained"])) for q in answer],
            [float(bool(new[q]["abstained"])) for q in answer],
        ),
        "refused (should refuse)": _paired(
            [float(bool(old[q]["abstained"])) for q in abstain],
            [float(bool(new[q]["abstained"])) for q in abstain],
        ),
    }
    with_retrieval = [q for q in shared if old[q].get("retrieval") and new[q].get("retrieval")]
    for key in ("recall@5", "mrr@10"):
        rows = [
            q for q in with_retrieval if key in old[q]["retrieval"] and key in new[q]["retrieval"]
        ]
        if len(rows) >= 2:
            measures[key] = _paired(
                [float(old[q]["retrieval"][key]) for q in rows],
                [float(new[q]["retrieval"][key]) for q in rows],
            )
    return {
        "old": {"file": old_path.name, "label": old_report.get("label")},
        "new": {"file": new_path.name, "label": new_report.get("label")},
        "bootstrap": {"samples": SAMPLES, "seed": SEED, "interval": "95% percentile, paired"},
        "questions_compared": len(shared),
        "left_out": sorted(set(old) ^ set(new)),
        "measures": measures,
        "changed_answers": {
            "gained": [q for q in answer if not old[q]["correct"] and new[q]["correct"]],
            "lost": [q for q in answer if old[q]["correct"] and not new[q]["correct"]],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = compare(args.old, args.new)
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8", newline="\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
