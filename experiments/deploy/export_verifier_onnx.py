"""Export a claim verifier to ONNX, quantize it to int8, and measure size, speed and agreement.

    python experiments/deploy/export_verifier_onnx.py \
        --model cross-encoder/nli-deberta-v3-xsmall --mapping nli \
        --out experiments/deploy/runs/nli-xsmall --report report.json

``--model`` is a Hugging Face id (read from the local cache) or a local model directory;
``--mapping`` is ``nli`` (3-way NLI model, the fixed 4-label rule of verifier_lib) or ``ft``
(a 4-label model). Three versions are scored on the hand-written test claims
(evaluation/datasets/groundedai-claims-nist-v1): PyTorch fp32, ONNX fp32 and ONNX int8
(onnxruntime dynamic quantization of the weights). Speed is measured one claim at a time
(batch 1, the application's case) with a fixed number of threads, after a warm-up.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "verifier"))

from verifier_lib import (  # noqa: E402
    CrossEncoderClassifier,
    OnnxClassifier,
    finetuned_verifier,
    load,
    metrics,
    nli_verifier,
)

TEST = ROOT / "evaluation" / "datasets" / "groundedai-claims-nist-v1" / "claims.jsonl"


def model_dir(name: str) -> Path:
    path = Path(name)
    if path.is_dir():
        return path
    cache = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + name.replace("/", "--"))
    snapshots = sorted((cache / "snapshots").glob("*"))
    if not snapshots:
        raise SystemExit(f"{name} is not in the local Hugging Face cache")
    return snapshots[-1]


def export(source: Path, out: Path) -> Path:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    out.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source).eval()
    tokenizer.save_pretrained(out)
    shutil.copy(source / "config.json", out / "config.json")
    sample = tokenizer(["The evidence."], ["The claim."], return_tensors="pt")
    # Positional, in the order of the model's forward(): a dict would be matched to the names
    # by position and could swap attention_mask and token_type_ids.
    names = [n for n in ("input_ids", "attention_mask", "token_type_ids") if n in sample]
    path = out / "model.onnx"
    axes = {n: {0: "batch", 1: "tokens"} for n in names}
    axes["logits"] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(sample[n] for n in names),
            str(path),
            input_names=names,
            output_names=["logits"],
            dynamic_axes=axes,
            opset_version=17,
            dynamo=False,
        )
    return path


def quantize(path: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    target = path.with_name("model.int8.onnx")
    quantize_dynamic(str(path), str(target), weight_type=QuantType.QInt8)
    return target


def score(classifier, mapping: str, rows: list[dict], threads: int) -> dict:
    import torch

    torch.set_num_threads(threads)
    make = nli_verifier if mapping == "nli" else finetuned_verifier
    verifier = make("", classifier)
    pairs = [(r["claim"], r["evidence"]) for r in rows]
    verifier(pairs[:8])  # warm-up
    times = []
    predicted = []
    for pair in pairs:
        started = time.perf_counter()
        predicted.extend(verifier([pair]))
        times.append(1000 * (time.perf_counter() - started))
    result = metrics([r["label"] for r in rows], predicted)
    result["ms_per_claim_median"] = round(statistics.median(times), 1)
    result["ms_per_claim_p90"] = round(sorted(times)[int(0.9 * len(times))], 1)
    return {"scores": result, "predicted": predicted}


def size_mb(path: Path) -> float:
    return round(path.stat().st_size / 2**20, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--mapping", choices=["nli", "ft"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    import onnxruntime
    import psutil
    import torch

    source = model_dir(args.model)
    rows = load(TEST)
    onnx_path = export(source, args.out)
    int8_path = quantize(onnx_path)
    weights = list(source.glob("*.safetensors")) + list(source.glob("pytorch_model.bin"))
    versions = {
        "pytorch-fp32": (lambda: CrossEncoderClassifier(str(source)), sum(map(size_mb, weights))),
        "onnx-fp32": (
            lambda: OnnxClassifier(str(onnx_path), str(args.out), threads=args.threads),
            size_mb(onnx_path),
        ),
        "onnx-int8": (
            lambda: OnnxClassifier(str(int8_path), str(args.out), threads=args.threads),
            size_mb(int8_path),
        ),
    }
    report: dict = {
        "model": args.model,
        "mapping": args.mapping,
        "claims": len(rows),
        "threads": args.threads,
        "machine": platform.processor(),
        "power_source": (
            "ac"
            if (battery := psutil.sensors_battery()) is None or battery.power_plugged
            else "battery"
        ),
        "packages": {"torch": torch.__version__, "onnxruntime": onnxruntime.__version__},
        "cpu_load_percent_before_timing": psutil.cpu_percent(interval=3),
        "logical_cpus": psutil.cpu_count(),
        "versions": {},
    }
    reference = None
    for name, (build, megabytes) in versions.items():
        started = time.perf_counter()
        classifier = build()
        load_seconds = round(time.perf_counter() - started, 2)
        result = score(classifier, args.mapping, rows, args.threads)
        if reference is None:
            reference = result["predicted"]
        agreement = sum(a == b for a, b in zip(result["predicted"], reference, strict=True))
        report["versions"][name] = {
            "file_mb": megabytes,
            "load_seconds": load_seconds,
            "agreement_with_pytorch": round(agreement / len(rows), 4),
            **result["scores"],
        }
        s = result["scores"]
        print(
            f"{name:13s} {megabytes:7.1f} MB  macro-F1 {s['macro_f1']:.3f}  acc {s['accuracy']:.3f}"
            f"  median {s['ms_per_claim_median']} ms  agree {agreement}/{len(rows)}",
            flush=True,
        )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
