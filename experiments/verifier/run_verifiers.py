"""Compare claim verifiers on the hand-written test set and the rule-made sets; write a report.

    python experiments/verifier/run_verifiers.py --out report.json \
        --method rule:claims-v1 --method rule:claims-v2 \
        --method nli:cross-encoder/nli-deberta-v3-xsmall \
        --method ft:experiments/finetune/runs/x/epoch-2

Sets: ``test`` = evaluation/datasets/groundedai-claims-nist-v1 (hand-written, test PDFs),
``validation`` and ``synthetic`` = rule-made pairs from experiments/verifier/data (validation
PDFs; the fictional synthetic documents, out of domain). Predictions are kept per claim.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

from verifier_lib import ROOT, evaluate, finetuned_verifier, load, nli_verifier, rule_verifier

SETS = {
    "test": ROOT / "evaluation" / "datasets" / "groundedai-claims-nist-v1" / "claims.jsonl",
    "validation": ROOT / "experiments" / "verifier" / "data" / "validation.jsonl",
    "synthetic": ROOT / "experiments" / "verifier" / "data" / "synthetic.jsonl",
}


def build(method: str):
    kind, _, name = method.partition(":")
    if kind == "rule":
        return rule_verifier(name)
    if kind == "nli":
        return nli_verifier(name)
    if kind == "ft":
        return finetuned_verifier(name)
    raise SystemExit(f"unknown method {method}")


def model_size_mb(method: str) -> float | None:
    kind, _, name = method.partition(":")
    if kind == "rule":
        return 0.0
    path = Path(name)
    if not path.is_dir():  # a hub id: look in the local Hugging Face cache
        cache = (
            Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + name.replace("/", "--"))
        )
        snapshots = sorted((cache / "snapshots").glob("*"))
        if not snapshots:
            return None
        path = snapshots[-1]
    files = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model.bin"))
    return round(sum(f.stat().st_size for f in files) / 2**20, 1) if files else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", action="append", required=True)
    parser.add_argument("--sets", default="test,validation,synthetic")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report: dict = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "machine": platform.processor(),
        "results": {},
        "predictions": {},
    }
    for method in args.method:
        started = time.perf_counter()
        verifier = build(method)
        load_seconds = round(time.perf_counter() - started, 2)
        report["results"][method] = {"load_seconds": load_seconds, "size_mb": model_size_mb(method)}
        for name in args.sets.split(","):
            rows = load(SETS[name])
            scores, predicted = evaluate(verifier, rows)
            report["results"][method][name] = scores
            report["predictions"].setdefault(name, {})[method] = predicted
            print(
                f"{method:60s} {name:10s} acc {scores['accuracy']:.3f} macro-F1 "
                f"{scores['macro_f1']:.3f} FPR(supported) "
                f"{scores['supported_vs_rest']['false_positive_rate']}",
                flush=True,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)  # saved after every set
            args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
