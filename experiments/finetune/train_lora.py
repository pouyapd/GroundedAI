"""LoRA fine-tuning of a small instruct model on GroundedAI's answer format (CPU, fixed seed).

    python experiments/finetune/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct \
        --train data/sft_train.jsonl --val data/sft_val.jsonl --out runs/qwen05-lora

The prompt is exactly what the application sends (recorded by make_sft_data.py); the loss is on
the reply tokens only. LoRA (rank 16, alpha 32, dropout 0.05) on the attention and MLP
projections; the base weights stay frozen. The adapter is merged into a copy of the base model
(``merged/``), which the application loads like any local model
(GROUNDEDAI_GENERATION_PROVIDER=huggingface-local, GROUNDEDAI_HF_GENERATION_MODEL=<dir>).

``--max-steps N`` stops after N training examples (a smoke test: it shows that training runs
and how long one example takes on this machine; it is not a trained model). With
``--smoke`` nothing is saved but ``smoke_log.json``: seconds and loss per example, peak memory
and the library versions.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1792)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N examples")
    parser.add_argument("--val-limit", type=int, default=None, help="use the first N val examples")
    parser.add_argument("--smoke", action="store_true", help="save only smoke_log.json")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

    here = Path(__file__).parent
    resolve = lambda p: p if p.is_absolute() else here / p  # noqa: E731
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def encode(row: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True
        )
        full = prompt + row["target"] + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if len(ids) > args.max_length:
            return None
        labels = [-100] * len(prompt_ids) + ids[len(prompt_ids) :]
        return torch.tensor([ids]), torch.tensor([labels])

    train = [e for e in (encode(r) for r in load(resolve(args.train))) if e is not None]
    val = (
        [e for e in (encode(r) for r in load(resolve(args.val))) if e is not None]
        if args.val
        else []
    )
    if args.val_limit is not None:
        val = val[: args.val_limit]
    peak_rss = [0]

    def note_memory() -> None:
        try:
            import psutil

            peak_rss[0] = max(peak_rss[0], psutil.Process().memory_info().rss)
        except ImportError:
            pass
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    updates = args.epochs * max(1, len(train) // args.accumulate)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.05 * updates)), updates)
    log: dict = {
        "base": args.base,
        "train_examples": len(train),
        "val_examples": len(val),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "args": {k: str(v) for k, v in vars(args).items()},
        "epochs": [],
    }
    print(f"{len(train)} examples, trainable {trainable:,} of {total:,} parameters", flush=True)

    def val_loss() -> float | None:
        if not val:
            return None
        model.eval()
        with torch.no_grad():
            losses = [float(model(input_ids=ids, labels=gold).loss) for ids, gold in val]
        model.train()
        return round(sum(losses) / len(losses), 4)

    log["val_loss_before"] = val_loss()
    print("validation loss before", log["val_loss_before"], flush=True)
    model.train()
    for epoch in range(1, args.epochs + 1):
        random.shuffle(train)
        started = time.perf_counter()
        losses = []
        seconds: list[float] = []
        lengths: list[int] = []
        last = len(train) if args.max_steps is None else min(len(train), args.max_steps)
        for step, (ids, labels) in enumerate(train[:last], start=1):
            step_started = time.perf_counter()
            loss = model(input_ids=ids, labels=labels).loss / args.accumulate
            loss.backward()
            losses.append(float(loss) * args.accumulate)
            if step % args.accumulate == 0 or step == last:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            seconds.append(round(time.perf_counter() - step_started, 2))
            lengths.append(int(ids.shape[1]))
            note_memory()
            if args.max_steps is not None:
                print(f"example {step}/{last}: {seconds[-1]} s, loss {losses[-1]:.4f}", flush=True)
            if step % 20 == 0:
                rate = (time.perf_counter() - started) / step
                print(
                    f"epoch {epoch} step {step}/{len(train)} loss {sum(losses[-20:]) / 20:.3f} "
                    f"({rate:.1f} s/example)",
                    flush=True,
                )
        entry = {
            "epoch": epoch,
            "examples": last,
            "seconds": round(time.perf_counter() - started, 1),
            "seconds_per_example": seconds,
            "tokens_per_example": lengths,
            "train_loss_per_example": [round(x, 4) for x in losses],
            "mean_train_loss": round(sum(losses) / len(losses), 4),
            "val_loss": val_loss(),
        }
        log["epochs"].append(entry)
        print({k: v for k, v in entry.items() if not isinstance(v, list)}, flush=True)
        if args.max_steps is not None:
            break
    out = resolve(args.out)
    if args.smoke:
        import platform

        import peft
        import transformers

        log["smoke_test"] = True
        log["peak_rss_mb"] = round(peak_rss[0] / 2**20) if peak_rss[0] else None
        log["environment"] = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "processor": platform.processor(),
            "threads": torch.get_num_threads(),
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "smoke_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        return 0
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    tokenizer.save_pretrained(out / "merged")
    (out / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
