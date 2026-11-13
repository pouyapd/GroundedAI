"""Fine-tune an NLI cross-encoder into a four-label claim verifier (CPU, fixed seed).

    python experiments/verifier/train_verifier.py --base cross-encoder/nli-deberta-v3-xsmall \
        --train experiments/verifier/data/train.jsonl \
        --val experiments/verifier/data/validation.jsonl \
        --out experiments/finetune/runs/verifier-xsmall

Full fine-tuning (the model is small: 22M backbone parameters plus embeddings), AdamW with a
linear warm-up and decay, cross-entropy over LABELS. The classification head is replaced (four
labels instead of three). After each epoch the model is scored on the validation pairs (made
from the validation documents, never the test documents); the epoch with the best macro-F1 is
kept. The training log records the time per epoch, the losses and the validation scores.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from verifier_lib import LABELS, finetuned_verifier, load, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="cross-encoder/nli-deberta-v3-xsmall")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import numpy as np
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train = load(args.train)
    val = load(args.val)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base,
        num_labels=len(LABELS),
        ignore_mismatched_sizes=True,
        id2label=dict(enumerate(LABELS)),
        label2id={label: i for i, label in enumerate(LABELS)},
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    log: dict = {
        "base": args.base,
        "train_pairs": len(train),
        "val_pairs": len(val),
        "args": {k: str(v) for k, v in vars(args).items()},
        "epochs": [],
    }
    best = -1.0
    args.out.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        started = time.perf_counter()
        losses = []
        for start in range(0, len(order), args.batch):
            batch = [train[i] for i in order[start : start + args.batch]]
            encoded = tokenizer(
                [r["evidence"] for r in batch],
                [r["claim"] for r in batch],
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            labels = torch.tensor([LABELS.index(r["label"]) for r in batch])
            loss = model(**encoded, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss))
            if len(losses) % 25 == 0:
                print(
                    f"epoch {epoch} step {len(losses)} loss {np.mean(losses[-25:]):.3f}", flush=True
                )
        seconds = time.perf_counter() - started
        model.eval()
        epoch_dir = args.out / f"epoch-{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        predicted = finetuned_verifier(str(epoch_dir))([(r["claim"], r["evidence"]) for r in val])
        scores = metrics([r["label"] for r in val], predicted)
        log["epochs"].append(
            {
                "epoch": epoch,
                "seconds": round(seconds, 1),
                "mean_loss": round(float(np.mean(losses)), 4),
                "val_macro_f1": scores["macro_f1"],
                "val_accuracy": scores["accuracy"],
            }
        )
        print(log["epochs"][-1], flush=True)
        if scores["macro_f1"] > best:
            best = scores["macro_f1"]
            log["best_epoch"] = epoch
    (args.out / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("best epoch", log["best_epoch"], "val macro-F1", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
