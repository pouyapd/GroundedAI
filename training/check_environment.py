"""Check that this Python environment can run GroundedAI's LoRA training; write a JSON report.

    python training/check_environment.py --out environment_check.json

Run in a fresh process after the libraries are installed. It records the versions (Python,
PyTorch, CUDA, Transformers, PEFT, Accelerate, torchao), the GPU, RAM and disk, and then does
what training will do, on a tiny model built from a configuration (nothing is downloaded):
creates the LoRA layers with the training settings (this is where an incompatible torchao made
PEFT fail), and runs one forward and backward pass, in fp16 with autocast on a GPU. Exit code 1
and ``"ok": false`` with the error when anything fails.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any


def version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def check() -> dict[str, Any]:
    report: dict[str, Any] = {"python": platform.python_version()}
    for package in ("torch", "transformers", "peft", "accelerate", "torchao", "groundedai"):
        report[package] = version(package)
    import torch

    report["cuda_available"] = torch.cuda.is_available()
    report["cuda_version"] = torch.version.cuda
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report["gpu"] = props.name
        report["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
        report["compute_capability"] = f"{props.major}.{props.minor}"
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
    try:
        import psutil

        report["ram_gb"] = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        report["ram_gb"] = None
    report["disk_free_gb"] = round(shutil.disk_usage(Path.cwd()).free / 2**30, 1)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, Qwen2Config

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from finetune_lora import TARGET_MODULES

    gpu = torch.cuda.is_available()
    config = Qwen2Config(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = AutoModelForCausalLM.from_config(config).to(torch.float16 if gpu else torch.float32)
    if gpu:
        model.to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=TARGET_MODULES,
            task_type="CAUSAL_LM",
        ),
    )
    report["lora_layers_created"] = True
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    ids = torch.randint(0, 512, (1, 16), device="cuda" if gpu else "cpu")
    with torch.autocast(
        "cuda" if gpu else "cpu", dtype=torch.float16 if gpu else torch.bfloat16, enabled=gpu
    ):
        loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    report["forward_backward"] = all(
        g is not None and torch.isfinite(g).all().item() for g in grads
    )
    report["precision_tested"] = "fp16 base, fp32 LoRA, fp16 autocast" if gpu else "fp32 (CPU)"
    report["ok"] = bool(report["forward_backward"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = check()
    except Exception as exc:
        report = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        for package in ("torch", "transformers", "peft", "accelerate", "torchao"):
            report[package] = version(package)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "traceback"}, indent=2))
    if not report["ok"]:
        print(report.get("traceback", ""), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
