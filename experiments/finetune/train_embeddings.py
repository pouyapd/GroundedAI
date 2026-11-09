"""Fine-tune a sentence-embedding model for retrieval on the training PDFs (CPU, fixed seed).

    python experiments/finetune/train_embeddings.py --base BAAI/bge-small-en-v1.5 \
        --train data/train_queries.jsonl --val data/val_queries.jsonl --out runs/bge-small-ft

Contrastive training with the multiple-negatives ranking loss (InfoNCE): for each generated
question, its own passage is the positive; the other passages in the batch and one hard negative
are negatives. The hard negative is the passage of the same document that BM25-style word overlap
ranks highest among those not containing the answer phrase. Embeddings are L2-normalised and
compared by cosine similarity with temperature 0.05 (scale 20), as in sentence-transformers.
Queries get the model's retrieval prefix (bge: "Represent this sentence for searching relevant
passages: "). The validation questions (other documents) give Recall@k and MRR over all their
documents' chunks after every epoch; the best epoch is kept.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.core.text import content_tokens, normalize_whitespace  # noqa: E402

BGE_PREFIX = "Represent this sentence for searching relevant passages: "


def load(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def bm25_scores(query: str, passages: list[list[str]], df: Counter, n: int) -> list[float]:
    terms = content_tokens(query)
    avg = sum(len(p) for p in passages) / max(1, len(passages))
    scores = []
    for tokens in passages:
        counts = Counter(tokens)
        score = 0.0
        for term in terms:
            if term not in counts:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            tf = counts[term]
            score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(tokens) / avg))
        scores.append(score)
    return scores


def hard_negatives(rows: list[dict]) -> list[str | None]:
    by_doc: dict[str, list[str]] = {}
    for row in rows:
        by_doc.setdefault(row["doc"], [])
        if row["passage"] not in by_doc[row["doc"]]:
            by_doc[row["doc"]].append(row["passage"])
    out = []
    for row in rows:
        passages = by_doc[row["doc"]]
        tokens = [content_tokens(p) for p in passages]
        df: Counter = Counter(t for ts in tokens for t in set(ts))
        scores = bm25_scores(row["question"], tokens, df, len(passages))
        answer = normalize_whitespace(row["answer"]).lower()
        ranked = sorted(range(len(passages)), key=lambda i: -scores[i])
        choice = next(
            (
                passages[i]
                for i in ranked
                if passages[i] != row["passage"]
                and answer not in normalize_whitespace(passages[i]).lower()
            ),
            None,
        )
        out.append(choice)
    return out


def evaluate(model, rows: list[dict], prefix: str, ks=(1, 5, 10)) -> dict:
    passages = sorted({row["passage"] for row in rows})
    index = {p: i for i, p in enumerate(passages)}
    p_emb = model.encode(passages, normalize_embeddings=True, batch_size=32, convert_to_tensor=True)
    q_emb = model.encode(
        [prefix + r["question"] for r in rows],
        normalize_embeddings=True,
        batch_size=32,
        convert_to_tensor=True,
    )
    sims = q_emb @ p_emb.T
    recall = {k: 0 for k in ks}
    mrr = 0.0
    for i, row in enumerate(rows):
        order = sims[i].argsort(descending=True).tolist()
        rank = order.index(index[row["passage"]]) + 1
        mrr += 1 / rank if rank <= 10 else 0
        for k in ks:
            recall[k] += rank <= k
    n = len(rows)
    return {
        "n": n,
        "passages": len(passages),
        "mrr@10": round(mrr / n, 4),
        **{f"recall@{k}": round(recall[k] / n, 4) for k in ks},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prefix", default=BGE_PREFIX)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import get_linear_schedule_with_warmup

    base = Path(__file__).parent
    resolve = lambda p: p if p.is_absolute() else base / p  # noqa: E731
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    train = load(resolve(args.train))
    val = load(resolve(args.val))
    negatives = hard_negatives(train)
    model = SentenceTransformer(args.base, device="cpu")
    log: dict = {
        "base": args.base,
        "train_pairs": len(train),
        "val_questions": len(val),
        "hard_negatives": sum(n is not None for n in negatives),
        "args": {k: str(v) for k, v in vars(args).items()},
        "epochs": [],
    }
    log["epoch_0"] = evaluate(model, val, args.prefix)
    print("before training", log["epoch_0"], flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * steps), steps)
    out = resolve(args.out)
    best = log["epoch_0"]["mrr@10"]
    log["best_epoch"] = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        started = time.perf_counter()
        losses = []
        for start in range(0, len(order), args.batch):
            batch = [order[i] for i in range(start, min(start + args.batch, len(order)))]
            queries = [args.prefix + train[i]["question"] for i in batch]
            docs = [train[i]["passage"] for i in batch]
            docs += [negatives[i] for i in batch if negatives[i] is not None]

            def embed(texts: list[str]) -> torch.Tensor:
                features = model.tokenize(texts)
                return torch.nn.functional.normalize(model(features)["sentence_embedding"], dim=-1)

            q = embed(queries)
            d = embed(docs)
            scores = 20.0 * q @ d.T
            loss = torch.nn.functional.cross_entropy(scores, torch.arange(len(batch)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(float(loss))
        model.eval()
        scores_val = evaluate(model, val, args.prefix)
        entry = {
            "epoch": epoch,
            "seconds": round(time.perf_counter() - started, 1),
            "mean_loss": round(float(np.mean(losses)), 4),
            "val": scores_val,
        }
        log["epochs"].append(entry)
        print(entry, flush=True)
        if scores_val["mrr@10"] > best:
            best = scores_val["mrr@10"]
            log["best_epoch"] = epoch
            model.save(str(out / "best"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    print("best epoch", log["best_epoch"], "val MRR@10", best)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
