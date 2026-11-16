"""LoRA fine-tuning of GroundedAI's answer model (Qwen2.5-1.5B-Instruct by default).

    python training/finetune_lora.py --report-hardware
    python training/finetune_lora.py --out runs/smoke --smoke                 # a few steps
    python training/finetune_lora.py --out /content/drive/MyDrive/groundedai-lora/run1

Data: `training/data/train.jsonl` and `validation.jsonl` (made by `prepare_dataset.py`): the
exact chat messages GroundedAI sends and a target reply. The loss is on the reply tokens only.

Precision and memory: on a CUDA GPU the base weights are loaded in fp16, the LoRA weights stay
fp32, and training uses fp16 autocast with loss scaling (no bf16, no 4-bit, no bitsandbytes, so
it runs on a T4). Gradient checkpointing is on by default on a GPU. ``--max-length auto`` tries
the longest example at 2048, 1536 and 1024 tokens (one forward and backward pass each) and keeps
the largest that fits in GPU memory; examples longer than the chosen length are left out and
counted. On the CPU it uses fp32 (for testing the script, not for training).

Resuming: checkpoints go to ``<out>/checkpoints``; when the script starts and finds one, it
continues from the latest. The best checkpoint by validation loss is kept as the result.

Saved in ``<out>``: ``adapter/`` (LoRA weights and adapter_config.json), ``tokenizer/``,
``training_config.json``, ``environment.json``, ``hardware.json``, ``loss_history.json``,
``data_report.json`` and ``result.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training" / "data"
DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


# ------------------------------------------------------------------------------ hardware
def hardware_report() -> dict[str, Any]:
    import torch

    report: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "platform": platform.platform(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(
            gpu=props.name,
            gpu_memory_gb=round(props.total_memory / 2**30, 1),
            compute_capability=f"{props.major}.{props.minor}",
            bf16_supported=torch.cuda.is_bf16_supported(),
        )
    try:
        import psutil

        memory = psutil.virtual_memory()
        report["ram_gb"] = round(memory.total / 2**30, 1)
        report["ram_available_gb"] = round(memory.available / 2**30, 1)
    except ImportError:
        pass
    disk = shutil.disk_usage(Path.cwd())
    report["disk_free_gb"] = round(disk.free / 2**30, 1)
    gb = report.get("gpu_memory_gb", 0)
    if not report["cuda_available"]:
        verdict = "no GPU: training would run on the CPU (only the smoke test is practical)"
    elif gb >= 14:
        verdict = "suitable: fp16 LoRA on Qwen2.5-1.5B with sequences up to 2048 tokens"
    elif gb >= 10:
        verdict = (
            "probably suitable: fp16 LoRA on Qwen2.5-1.5B; --max-length auto may pick 1536 or 1024"
        )
    else:
        verdict = (
            "too little GPU memory for Qwen2.5-1.5B in fp16; use --base Qwen/Qwen2.5-0.5B-Instruct"
        )
    report["verdict"] = verdict
    return report


def environment() -> dict[str, Any]:
    import peft
    import torch
    import transformers

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
    }
    try:
        import accelerate

        info["accelerate"] = accelerate.__version__
    except ImportError:
        pass
    commit = os.popen(f'git -C "{ROOT}" rev-parse --short HEAD').read().strip()  # noqa: S605
    info["groundedai_commit"] = commit or None
    return info


# ---------------------------------------------------------------------------------- data
def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def encode(
    tokenizer: Any, row: dict[str, Any], max_length: int, truncate: bool
) -> dict[str, list[int]] | None:
    """Chat template + target + end token; labels only on the target. None when too long."""
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(row["target"] + tokenizer.eos_token, add_special_tokens=False)[
        "input_ids"
    ]
    if len(prompt_ids) + len(target_ids) > max_length:
        if not truncate or len(target_ids) >= max_length // 2:
            return None
        prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]  # smoke test only
    ids = prompt_ids + target_ids
    return {"input_ids": ids, "labels": [-100] * len(prompt_ids) + target_ids}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict[str, list[int]]]) -> dict[str, Any]:
        import torch

        width = max(len(item["input_ids"]) for item in batch)
        ids, labels, mask = [], [], []
        for item in batch:
            pad = width - len(item["input_ids"])
            ids.append(item["input_ids"] + [self.pad_id] * pad)
            labels.append(item["labels"] + [-100] * pad)
            mask.append([1] * len(item["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(mask),
        }


# ---------------------------------------------------------------------------------- model
def load_model(
    base: str, rank: int, alpha: int, dropout: float, gpu: bool, checkpointing: bool
) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=torch.float16 if gpu else torch.float32
    )
    if gpu:
        model.to("cuda")
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        model.config.use_cache = False
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, config)
    for (
        parameter
    ) in model.parameters():  # LoRA weights in fp32 (fp16 base): stable with loss scaling
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model, tokenizer


def fits(model: Any, example: dict[str, list[int]]) -> bool:
    """One forward and backward pass on the GPU; False on out-of-memory."""
    import torch

    batch = Collator(0)([example])
    try:
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**{k: v.to("cuda") for k, v in batch.items()}).loss
        loss.backward()
        ok = True
    except torch.OutOfMemoryError:
        ok = False
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return ok


def choose_length(
    model: Any, tokenizer: Any, rows: list[dict[str, Any]]
) -> tuple[int, list[dict[str, Any]]]:
    tried = []
    for length in (2048, 1536, 1024):
        encoded = [e for e in (encode(tokenizer, r, length, False) for r in rows) if e is not None]
        if not encoded:
            continue
        longest = max(encoded, key=lambda e: len(e["input_ids"]))
        ok = fits(model, longest)
        tried.append(
            {"max_length": length, "longest_example": len(longest["input_ids"]), "fits": ok}
        )
        if ok:
            return length, tried
    raise SystemExit(f"no sequence length fits in GPU memory: {tried}; try a smaller --base")


# ---------------------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--report-hardware", action="store_true", help="print the report and stop")
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--train", type=Path, default=DATA / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=DATA / "validation.jsonl")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-length", default="auto", help="'auto' or a number of tokens")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="optimizer steps; -1: by epochs")
    parser.add_argument("--accumulate", type=int, default=8, help="examples per optimizer step")
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--eval-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=20)
    parser.add_argument(
        "--validation-limit", type=int, help="score the first N validation examples"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--no-checkpointing", action="store_true", help="no gradient checkpointing")
    parser.add_argument(
        "--smoke", action="store_true", help="3 optimizer steps at 512 tokens (prompts cut)"
    )
    args = parser.parse_args()

    hardware = hardware_report()
    print(json.dumps(hardware, indent=2), flush=True)
    if args.report_hardware:
        return 0
    if args.out is None:
        parser.error("--out is required")

    import torch
    from transformers import Trainer, TrainingArguments, set_seed

    random.seed(args.seed)
    set_seed(args.seed)
    gpu = torch.cuda.is_available()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model, tokenizer = load_model(
        args.base, args.rank, args.alpha, args.dropout, gpu, gpu and not args.no_checkpointing
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    train_rows, val_rows = load(args.train), load(args.validation)
    if args.validation_limit:
        val_rows = val_rows[: args.validation_limit]
    length_trials: list[dict[str, Any]] = []
    if args.smoke:
        max_length, truncate = 512, True
        args.max_steps, args.eval_steps, args.save_steps = 3, 3, 3
        train_rows, val_rows = train_rows[:24], val_rows[:4]
    elif args.max_length == "auto":
        # On the CPU nothing is tried: 2048 covers every example (only for testing the script).
        max_length, length_trials = (
            choose_length(model, tokenizer, train_rows) if gpu else (2048, [])
        )
        truncate = False
    else:
        max_length, truncate = int(args.max_length), False
    train = [
        e for e in (encode(tokenizer, r, max_length, truncate) for r in train_rows) if e is not None
    ]
    val = [
        e for e in (encode(tokenizer, r, max_length, truncate) for r in val_rows) if e is not None
    ]
    data_report = {
        "max_length": max_length,
        "truncated_prompts": truncate,
        "length_trials": length_trials,
        "train_examples": len(train),
        "train_left_out_too_long": len(train_rows) - len(train),
        "validation_examples": len(val),
        "validation_left_out_too_long": len(val_rows) - len(val),
        "train_file": str(args.train.relative_to(ROOT))
        if args.train.is_relative_to(ROOT)
        else str(args.train),
        "validation_file": str(args.validation.relative_to(ROOT))
        if args.validation.is_relative_to(ROOT)
        else str(args.validation),
    }
    print(json.dumps(data_report, indent=2), flush=True)

    checkpoints = out / "checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulate,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_ratio,  # a float below 1 is a share of all steps
        lr_scheduler_type="linear",
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=gpu,
        logging_steps=1 if args.smoke else 5,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        dataloader_num_workers=0,
        remove_unused_columns=False,
        gradient_checkpointing=False,  # set on the model above (with use_reentrant=False)
        use_cpu=not gpu,
    )
    config = {
        "base_model": args.base,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": TARGET_MODULES,
        },
        "precision": "fp16 base, fp32 LoRA weights, fp16 autocast with loss scaling"
        if gpu
        else "fp32 (CPU)",
        "gradient_checkpointing": gpu and not args.no_checkpointing,
        "batch_size": 1,
        "gradient_accumulation": args.accumulate,
        "learning_rate": args.lr,
        "schedule": f"linear, warm-up {args.warmup_ratio}",
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "eval_and_save_every_steps": [args.eval_steps, args.save_steps],
        "selection": "lowest validation loss (load_best_model_at_end)",
        "seed": args.seed,
        "smoke_test": args.smoke,
        "trainable_parameters": trainable,
        "total_parameters": total,
        **data_report,
    }
    (out / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "environment.json").write_text(
        json.dumps(environment(), indent=2) + "\n", encoding="utf-8"
    )
    (out / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    resume = checkpoints.is_dir() and any(
        p.name.startswith("checkpoint-") for p in checkpoints.iterdir()
    )
    print(
        f"{'resuming from the latest checkpoint' if resume else 'starting'}: {len(train)} "
        f"examples, trainable {trainable:,} of {total:,} parameters",
        flush=True,
    )
    result = trainer.train(resume_from_checkpoint=True if resume else None)
    final_eval = trainer.evaluate()
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "tokenizer")
    history = trainer.state.log_history
    (out / "loss_history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    peak = round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None
    summary = {
        "finished": True,
        "resumed": resume,
        "global_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint
        and Path(trainer.state.best_model_checkpoint).name,
        "best_validation_loss": trainer.state.best_metric,
        "final_validation_loss": final_eval.get("eval_loss"),
        "first_validation_loss": next((h["eval_loss"] for h in history if "eval_loss" in h), None),
        "train_loss": result.training_loss,
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "wall_time_s": round(time.perf_counter() - started, 1),
        "peak_gpu_memory_gb": peak,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "data_report.json").write_text(
        json.dumps(data_report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
