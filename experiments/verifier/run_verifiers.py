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
    # hand-written, fictional documents (groundedai-versions-v1): dev and validation halves
    "versions-dev": ROOT / "evaluation" / "datasets" / "groundedai-versions-v1" / "claims.jsonl",
    "versions-validation": ROOT
    / "evaluation"
    / "datasets"
    / "groundedai-versions-v1"
    / "claims.jsonl",
}
SPLIT_FILTER = {"versions-dev": "dev", "versions-validation": "validation"}


def build(method: str):
    kind, _, name = method.partition(":")
    if kind == "rule":
        return rule_verifier(name)
    if kind == "nli":
        return nli_verifier(name)
    if kind == "ft":
        return finetuned_verifier(name)
    if kind == "onnx":  # onnx:<model dir>, an ONNX file in <dir>/onnx/model_quantized.onnx
        from verifier_lib import OnnxClassifier

        onnx_file = Path(name) / "onnx" / "model_quantized.onnx"
        return nli_verifier(name, OnnxClassifier(str(onnx_file), name, threads=4))
    if kind == "both":  # both:<rule method>|<NLI method>, e.g. both:rule:claims-v1|onnx:<dir>
        rule_method, _, nli_method = name.partition("|")
        return combined_verifier(build(rule_method), build(nli_method))
    raise SystemExit(f"unknown method {method}")


def combined_verifier(rule, nli):  # type: ignore[no-untyped-def]
    """Lexical and NLI together (rule fixed before any protected-set run, 2026-11-15):
    "supported" only when both say supported; "contradicted" when either says so; otherwise
    the lexical label (partial or nee), or "partial" when only one says supported."""

    def run(pairs):  # type: ignore[no-untyped-def]
        out = []
        for lexical, meaning in zip(rule(pairs), nli(pairs), strict=True):
            if lexical == meaning == "supported":
                out.append("supported")
            elif "contradicted" in (lexical, meaning):
                out.append("contradicted")
            elif "supported" in (lexical, meaning):
                out.append("partial")
            else:
                out.append(lexical)
        return out

    return run


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
            if name in SPLIT_FILTER:
                rows = [row for row in rows if row.get("split") == SPLIT_FILTER[name]]
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
