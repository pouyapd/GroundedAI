"""LoRA fine-tuning of a small instruct model on GroundedAI's answer format (CPU, fixed seed).

    python experiments/finetune/train_lora.py --base Qwen/Qwen2.5-0.5B-Instruct \
        --train data/sft_train.jsonl --val data/sft_val.jsonl --out runs/qwen05-lora

The prompt is exactly what the application sends (recorded by make_sft_data.py); the loss is on
the reply tokens only. LoRA (rank 16, alpha 32, dropout 0.05) on the attention and MLP
projections; the base weights stay frozen. The adapter is merged into a copy of the base model
(``merged/``), which the application loads like any local model
(GROUNDEDAI_GENERATION_PROVIDER=huggingface-local, GROUNDEDAI_HF_GENERATION_MODEL=<dir>).
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
        for step, (ids, labels) in enumerate(train, start=1):
            loss = model(input_ids=ids, labels=labels).loss / args.accumulate
            loss.backward()
            losses.append(float(loss) * args.accumulate)
            if step % args.accumulate == 0 or step == len(train):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if step % 20 == 0:
                rate = (time.perf_counter() - started) / step
                print(
                    f"epoch {epoch} step {step}/{len(train)} loss {sum(losses[-20:]) / 20:.3f} "
                    f"({rate:.1f} s/example)",
                    flush=True,
                )
        entry = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 1),
            "mean_train_loss": round(sum(losses) / len(losses), 4),
            "val_loss": val_loss(),
        }
        log["epochs"].append(entry)
        print(entry, flush=True)
    out = resolve(args.out)
    model.save_pretrained(out / "adapter")
    merged = model.merge_and_unload()
    merged.save_pretrained(out / "merged")
    tokenizer.save_pretrained(out / "merged")
    (out / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
