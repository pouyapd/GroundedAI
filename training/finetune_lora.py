"""LoRA fine-tuning of GroundedAI's answer model (Qwen2.5-1.5B-Instruct by default).

    python training/finetune_lora.py --report-hardware
    python training/finetune_lora.py --out runs/smoke --smoke     # staged check, PASS/FAIL
    python training/finetune_lora.py --out /content/groundedai-lora/run1

Data: `training/data/train.jsonl` and `validation.jsonl` (made by `prepare_dataset.py`): the
exact chat messages GroundedAI sends and a target reply. The loss is on the reply tokens only.

Precision and memory: on a CUDA GPU the base weights are loaded in fp16, the LoRA weights stay
fp32, and training uses fp16 autocast with loss scaling (no bf16, no 4-bit, no bitsandbytes, so
it runs on a T4). Gradient checkpointing is on by default on a GPU. ``--max-length auto`` tries
the longest example at 2048, 1536 and 1024 tokens (one forward and backward pass each) and keeps
the largest that fits in GPU memory; examples longer than the chosen length are left out and
counted. On the CPU it uses fp32 (for testing the script, not for training).

Resuming: checkpoints go to ``<out>/checkpoints`` every ``--save-steps`` steps. On start the
newest *valid* checkpoint (trainer state, optimizer, scheduler and adapter weights present) is
resumed; newer incomplete ones are renamed ``incomplete-checkpoint-…`` so they are never used. A
run that already finished (``result.json`` and ``adapter/``) is not trained again. The weights
of the checkpoint with the lowest validation loss are the result.

Saved in ``<out>``: ``adapter/`` (LoRA weights and adapter_config.json) and ``tokenizer/``, both
written only after training succeeded (first to ``*.partial``, then moved), and
``training_config.json``, ``environment.json``, ``hardware.json``, ``data_report.json``,
``loss_history.json`` and, last, ``result.json``.

``--smoke`` checks eight stages in order and prints PASS or FAIL for each (with the full error):
model loading, LoRA creation, a forward and backward pass, three optimizer steps, the validation
loss, saving the adapter, loading it again on a fresh base model, and generating with it. Prompts
are cut to 512 tokens. The result is in ``<out>/smoke_result.json``; the exit code is 1 on failure.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import sys
import time
import traceback
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


# ---------------------------------------------------------------------- checkpoints
CHECKPOINT_FILES = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def checkpoint_valid(path: Path) -> bool:
    """A checkpoint the Trainer can resume from: state, optimizer, scheduler and adapter weights,
    and a trainer_state.json that parses."""
    if not all((path / name).is_file() for name in CHECKPOINT_FILES):
        return False
    if not any((path / name).is_file() for name in ADAPTER_WEIGHTS):
        return False
    try:
        json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return True


def latest_valid_checkpoint(checkpoints: Path) -> Path | None:
    """The newest valid ``checkpoint-N``; newer incomplete ones are moved aside (renamed
    ``incomplete-checkpoint-N-<time>``) so that the Trainer never picks them up."""
    if not checkpoints.is_dir():
        return None
    found = []
    for path in checkpoints.iterdir():
        if path.is_dir() and path.name.startswith("checkpoint-"):
            try:
                found.append((int(path.name.split("-", 1)[1]), path))
            except ValueError:
                continue
    for _step, path in sorted(found, reverse=True):
        if checkpoint_valid(path):
            return path
        aside = path.with_name(f"incomplete-{path.name}-{int(time.time())}")
        path.rename(aside)
        print(f"moved an incomplete checkpoint aside: {aside.name}", flush=True)
    return None


def run_complete(out: Path) -> bool:
    if not (out / "adapter" / "adapter_config.json").is_file():
        return False
    try:
        return bool(json.loads((out / "result.json").read_text(encoding="utf-8")).get("finished"))
    except (OSError, json.JSONDecodeError):
        return False


def save_adapter(model: Any, tokenizer: Any, out: Path) -> None:
    """Write adapter and tokenizer next to their final place, then move them in."""
    for name, save in (
        ("adapter", model.save_pretrained),
        ("tokenizer", tokenizer.save_pretrained),
    ):
        partial = out / f"{name}.partial"
        if partial.exists():
            shutil.rmtree(partial)
        save(partial)
        final = out / name
        if final.exists():
            shutil.rmtree(final)
        partial.rename(final)


# ---------------------------------------------------------------------------------- main
def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--smoke", action="store_true", help="staged check: 3 steps at 512 tokens")
    return parser.parse_args()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def training_arguments(args: argparse.Namespace, checkpoints: Path, gpu: bool) -> Any:
    from transformers import TrainingArguments

    return TrainingArguments(
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
        gradient_checkpointing=False,  # set on the model (with use_reentrant=False)
        use_cpu=not gpu,
    )


def smoke(args: argparse.Namespace, out: Path) -> int:
    """Each stage of training on a few examples, PASS or FAIL, in order; stops at a failure."""
    import torch
    from transformers import Trainer

    gpu = torch.cuda.is_available()
    stages: list[dict[str, Any]] = []
    state: dict[str, Any] = {}

    def stage(name: str, action: Any) -> bool:
        started = time.perf_counter()
        try:
            detail = action()
            stages.append(
                {
                    "stage": name,
                    "passed": True,
                    "seconds": round(time.perf_counter() - started, 1),
                    "detail": detail,
                }
            )
            print(f"PASS  {name}  {detail or ''}", flush=True)
            return True
        except Exception as exc:
            stages.append(
                {
                    "stage": name,
                    "passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)
            return False

    def load_base() -> str:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        state["tokenizer"] = AutoTokenizer.from_pretrained(args.base)
        if state["tokenizer"].pad_token is None:
            state["tokenizer"].pad_token = state["tokenizer"].eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.float16 if gpu else torch.float32
        )
        state["model"] = model.to("cuda") if gpu else model
        return f"{args.base} on {'GPU, fp16' if gpu else 'CPU, fp32'}"

    def add_lora() -> str:
        from peft import LoraConfig, get_peft_model

        model = state["model"]
        if gpu and not args.no_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            model.enable_input_require_grads()
            model.config.use_cache = False
        config = LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, config)
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        state["model"] = model
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return f"{trainable:,} trainable parameters"

    rows = load(args.train)[: 3 * args.accumulate]
    val_rows = load(args.validation)[:4]

    def forward_backward() -> str:
        example = encode(state["tokenizer"], rows[0], 512, True)
        if example is None:
            raise RuntimeError("the first training example could not be encoded")
        batch = {
            k: v.to(state["model"].device)
            for k, v in Collator(state["tokenizer"].pad_token_id)([example]).items()
        }
        with torch.autocast("cuda", dtype=torch.float16, enabled=gpu):
            loss = state["model"](**batch).loss
        loss.backward()
        grads = [p.grad for p in state["model"].parameters() if p.requires_grad]
        if not all(g is not None and torch.isfinite(g).all() for g in grads):
            raise RuntimeError("missing or non-finite gradients")
        state["model"].zero_grad(set_to_none=True)
        return f"loss {loss.item():.3f}"

    def optimizer_steps() -> str:
        train = [
            e for e in (encode(state["tokenizer"], r, 512, True) for r in rows) if e is not None
        ]
        val = [
            e for e in (encode(state["tokenizer"], r, 512, True) for r in val_rows) if e is not None
        ]
        args.max_steps, args.eval_steps, args.save_steps = 3, 3, 3
        trainer = Trainer(
            model=state["model"],
            args=training_arguments(args, out / "checkpoints", gpu),
            train_dataset=train,
            eval_dataset=val,
            data_collator=Collator(state["tokenizer"].pad_token_id),
        )
        result = trainer.train()
        state["trainer"] = trainer
        if trainer.state.global_step != 3:
            raise RuntimeError(f"{trainer.state.global_step} optimizer steps instead of 3")
        return f"3 optimizer steps, training loss {result.training_loss:.3f}"

    def validation_loss() -> str:
        value = state["trainer"].evaluate()["eval_loss"]
        if not math.isfinite(value):
            raise RuntimeError(f"validation loss {value}")
        state["eval_loss"] = value
        return f"validation loss {value:.3f}"

    def save() -> str:
        save_adapter(state["model"], state["tokenizer"], out)
        files = sorted(p.name for p in (out / "adapter").iterdir())
        if "adapter_config.json" not in files or not any(name in files for name in ADAPTER_WEIGHTS):
            raise RuntimeError(f"adapter files missing: {files}")
        return ", ".join(files)

    def reload_adapter() -> str:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        saved = {
            name: p.detach().float().cpu()
            for name, p in state["model"].named_parameters()
            if "lora_" in name
        }
        del state["trainer"], state["model"]
        if gpu:
            torch.cuda.empty_cache()
        base = AutoModelForCausalLM.from_pretrained(
            args.base, dtype=torch.float16 if gpu else torch.float32
        )
        base = base.to("cuda") if gpu else base
        loaded = PeftModel.from_pretrained(base, str(out / "adapter")).eval()
        reloaded = {
            name: p.detach().float().cpu()
            for name, p in loaded.named_parameters()
            if "lora_" in name
        }
        if len(reloaded) != len(saved):
            raise RuntimeError(f"{len(reloaded)} LoRA tensors loaded, {len(saved)} saved")
        worst = max((saved[name] - reloaded[name]).abs().max().item() for name in saved)
        if worst > 1e-3:
            raise RuntimeError(f"loaded LoRA weights differ from the saved ones by {worst}")
        state["model"] = loaded
        return f"{len(reloaded)} LoRA tensors loaded; largest difference {worst:.2e}"

    def inference() -> str:
        tokenizer = state["tokenizer"]
        prompt = tokenizer.apply_chat_template(
            val_rows[0]["messages"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            state["model"].device
        )
        with torch.no_grad():
            output = state["model"].generate(**inputs, max_new_tokens=48, do_sample=False)
        text = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        if not text:
            raise RuntimeError("empty reply")
        return repr(text[:120])

    steps = [
        ("1. model loading", load_base),
        ("2. LoRA adapter creation", add_lora),
        ("3. forward and backward pass", forward_backward),
        ("4. three optimizer steps", optimizer_steps),
        ("5. validation loss", validation_loss),
        ("6. adapter saving", save),
        ("7. adapter loading", reload_adapter),
        ("8. inference with the adapter", inference),
    ]
    for name, action in steps:
        if not stage(name, action):
            break
    passed = len(stages) == len(steps) and all(s["passed"] for s in stages)
    result = {
        "passed": passed,
        "base_model": args.base,
        "device": "cuda" if gpu else "cpu",
        "stages": stages,
        "environment": environment(),
    }
    (out / "smoke_result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("\nSMOKE TEST " + ("PASSED" if passed else "FAILED"), flush=True)
    for item in stages:
        print(f"  {'PASS' if item['passed'] else 'FAIL'}  {item['stage']}")
    for name, _ in steps[len(stages) :]:
        print(f"  ----  {name} (not run)")
    return 0 if passed else 1


def main() -> int:
    args = parse_args()
    hardware = hardware_report()
    print(json.dumps(hardware, indent=2), flush=True)
    if args.report_hardware:
        return 0
    if args.out is None:
        raise SystemExit("--out is required")
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from transformers import Trainer, set_seed

    random.seed(args.seed)
    set_seed(args.seed)
    if args.smoke:
        return smoke(args, out)
    if run_complete(out):
        print(
            f"Training in {out} is already complete; nothing is overwritten. "
            "Use another --out to train again."
        )
        return 0

    gpu = torch.cuda.is_available()
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
    if args.max_length == "auto":
        # On the CPU nothing is tried: 2048 covers every example (only for testing the script).
        max_length, length_trials = (
            choose_length(model, tokenizer, train_rows) if gpu else (2048, [])
        )
    else:
        max_length = int(args.max_length)
    train = [
        e for e in (encode(tokenizer, r, max_length, False) for r in train_rows) if e is not None
    ]
    val = [e for e in (encode(tokenizer, r, max_length, False) for r in val_rows) if e is not None]
    data_report = {
        "max_length": max_length,
        "length_trials": length_trials,
        "train_examples": len(train),
        "train_left_out_too_long": len(train_rows) - len(train),
        "validation_examples": len(val),
        "validation_left_out_too_long": len(val_rows) - len(val),
        "train_file": relative(args.train),
        "validation_file": relative(args.validation),
    }
    checkpoints = out / "checkpoints"
    resume_from = latest_valid_checkpoint(checkpoints)
    steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(len(train) / args.accumulate) * math.ceil(args.epochs)
    )
    print("\nTRAINING PLAN", flush=True)
    for key, value in (
        (
            "GPU",
            f"{hardware.get('gpu')} ({hardware.get('gpu_memory_gb')} GB)" if gpu else "none (CPU)",
        ),
        ("base model", args.base),
        (
            "training examples",
            f"{len(train)} ({len(train_rows) - len(train)} left out as too long)",
        ),
        ("validation examples", f"{len(val)} ({len(val_rows) - len(val)} left out as too long)"),
        ("sequence length", f"{max_length} tokens {length_trials or ''}"),
        (
            "optimizer steps (estimate)",
            f"{steps} ({args.accumulate} examples each, {args.epochs} epochs)",
        ),
        ("output directory", str(out)),
        ("start", f"resumed from {resume_from.name}" if resume_from else "fresh"),
    ):
        print(f"  {key:28s} {value}", flush=True)

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
        "estimated_optimizer_steps": steps,
        "eval_and_save_every_steps": [args.eval_steps, args.save_steps],
        "selection": "lowest validation loss (load_best_model_at_end)",
        "seed": args.seed,
        "trainable_parameters": trainable,
        "total_parameters": total,
        **data_report,
    }
    (out / "training_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (out / "environment.json").write_text(
        json.dumps(environment(), indent=2) + "\n", encoding="utf-8"
    )
    (out / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    (out / "data_report.json").write_text(
        json.dumps(data_report, indent=2) + "\n", encoding="utf-8"
    )

    trainer = Trainer(
        model=model,
        args=training_arguments(args, checkpoints, gpu),
        train_dataset=train,
        eval_dataset=val,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    final_eval = trainer.evaluate()
    history = trainer.state.log_history
    (out / "loss_history.json").write_text(json.dumps(history, indent=1) + "\n", encoding="utf-8")
    save_adapter(model, tokenizer, out)  # the best checkpoint's weights (load_best_model_at_end)
    summary = {
        "finished": True,
        "resumed_from": resume_from.name if resume_from else None,
        "global_steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint
        and Path(trainer.state.best_model_checkpoint).name,
        "best_validation_loss": trainer.state.best_metric,
        "final_validation_loss": final_eval.get("eval_loss"),
        "first_validation_loss": next((h["eval_loss"] for h in history if "eval_loss" in h), None),
        "train_loss": result.training_loss,
        "train_runtime_s": round(result.metrics.get("train_runtime", 0.0), 1),
        "wall_time_s": round(time.perf_counter() - started, 1),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if gpu else None,
    }
    (out / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")  # last
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
