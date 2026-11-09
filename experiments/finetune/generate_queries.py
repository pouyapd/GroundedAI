"""Generate (question, answer, passage) triples from training or validation PDFs with a local model.

    python experiments/finetune/generate_queries.py --split train --per-doc 150 \
        --out data/train_queries.jsonl
    python experiments/finetune/generate_queries.py --split validation --per-doc 70 \
        --out data/val_queries.jsonl

The passages are GroundedAI's own chunks (same parser and chunker as the application). For each
sampled passage a local model (Ollama, default qwen2.5:1.5b, temperature 0, fixed seed) writes
one question and an answer phrase; a pair is kept only when the answer is copied verbatim from
the passage, so every kept question is answerable from its passage. This is the standard
"synthetic query" approach (doc2query / InPars style); the questions are model-written, not
human-written, and are only used for training and validation, never for the test results.
Test documents are refused.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from groundedai.core.text import normalize_whitespace  # noqa: E402
from groundedai.ingestion.chunking import ChunkingConfig, chunk_pages  # noqa: E402
from groundedai.ingestion.parsers import parse_pdf  # noqa: E402

SEED = 20261109
PROMPT = """You write exam questions about one passage from a technical document.
Write ONE question that this passage answers, and the answer copied word for word from the passage
(a short phrase of at most 15 words). Ask the question the way a reader would, using your own
words where natural; do not mention "the passage".
Reply only with JSON: {{"question": "...", "answer": "..."}}

Passage (from "{title}"):
{passage}"""


def corpora(split: str) -> list[tuple[str, str, Path]]:
    if split == "train":
        manifest_path = ROOT / "experiments" / "finetune" / "corpus.json"
    elif split == "validation":
        manifest_path = (
            ROOT / "evaluation" / "datasets" / "groundedai-nist-val-v1" / "manifest.json"
        )
    else:
        raise SystemExit("only the train and validation splits can be used to generate data")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [
        (doc["key"], doc["title"], manifest_path.parent / doc["file"])
        for doc in manifest["documents"]
    ]


def usable(text: str) -> bool:
    if len(text) < 300:
        return False
    if text.count(". .") + text.count("..") > 5:  # table of contents (dot leaders)
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters / len(text) > 0.6


def ask(model: str, url: str, prompt: str) -> dict[str, str] | None:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0, "seed": SEED, "num_predict": 120},
        }
    ).encode()
    request = urllib.request.Request(  # noqa: S310 - local Ollama URL
        f"{url}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 - local Ollama
        reply = json.loads(response.read())["response"]
    try:
        data = json.loads(reply)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    question, answer = str(data.get("question", "")).strip(), str(data.get("answer", "")).strip()
    return {"question": question, "answer": answer} if question and answer else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--per-doc", type=int, default=80)
    parser.add_argument("--model", default="qwen2.5:1.5b")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rng = random.Random(SEED)  # noqa: S311 - sampling, not security
    out = args.out if args.out.is_absolute() else ROOT / "experiments" / "finetune" / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.is_file():
        for line in out.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            done.add((row["doc"], row["chunk_index"]))
    kept = rejected = 0
    with out.open("a", encoding="utf-8", newline="\n") as handle:
        for key, title, path in corpora(args.split):
            parsed = parse_pdf(path.read_bytes(), max_pages=2000, max_chars=10_000_000)
            chunks = [
                c for c in chunk_pages(parsed.pages, ChunkingConfig(800, 120)) if usable(c.text)
            ]
            sample = rng.sample(chunks, min(args.per_doc, len(chunks)))
            for chunk in sample:
                if (key, chunk.index) in done:
                    continue
                passage = chunk.text
                result = ask(args.model, args.ollama, PROMPT.format(title=title, passage=passage))
                ok = (
                    bool(result)
                    and result["question"].endswith("?")
                    and (
                        normalize_whitespace(result["answer"]).lower().strip(" .\"'")
                        in normalize_whitespace(passage).lower()
                    )
                    and len(result["answer"]) >= 2
                )
                if not ok:
                    rejected += 1
                    continue
                kept += 1
                handle.write(
                    json.dumps(
                        {
                            "split": args.split,
                            "doc": key,
                            "title": title,
                            "chunk_index": chunk.index,
                            "page_start": chunk.page_start,
                            "page_end": chunk.page_end,
                            "passage": passage,
                            "question": result["question"],
                            "answer": re.sub(r"\s+", " ", result["answer"]).strip(),
                            "generator": args.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                handle.flush()
            print(f"{key}: kept {kept}, rejected {rejected}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
