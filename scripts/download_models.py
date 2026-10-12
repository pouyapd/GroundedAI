"""Download optional local models explicitly (nothing in GroundedAI downloads implicitly).

Models go to the standard Hugging Face cache (HF_HOME, default ~/.cache/huggingface).
The script shows the download size and asks before downloading unless --yes is given.

    python scripts/download_models.py --list
    python scripts/download_models.py --preset recommended  # bge-small + reranker, ~220 MB
    python scripts/download_models.py --embedding BAAI/bge-small-en-v1.5
    python scripts/download_models.py --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
    python scripts/download_models.py --generation Qwen/Qwen2.5-0.5B-Instruct

Ollama models are managed by Ollama itself: ``ollama pull qwen2.5:1.5b``.
Requires the optional ML extra (huggingface_hub): pip install -e "backend[ml]".
"""

from __future__ import annotations

import argparse
import os
import sys

PRESETS = {
    "recommended": [
        "BAAI/bge-small-en-v1.5",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ],
    "benchmark": [
        "BAAI/bge-small-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ],
}
CATALOGUE = {
    "BAAI/bge-small-en-v1.5": "embedding, 384 dims, MIT licence",
    "sentence-transformers/all-MiniLM-L6-v2": "embedding, 384 dims, Apache-2.0",
    "cross-encoder/ms-marco-MiniLM-L-6-v2": "reranker, Apache-2.0 (trained on MS MARCO)",
    "Qwen/Qwen2.5-0.5B-Instruct": (
        "small instruction LLM for the huggingface-local provider, Apache-2.0"
    ),
}
# Only the files a PyTorch/safetensors runtime needs (skips ONNX, OpenVINO, TF, duplicate .bin).
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.txt",
    "*.model",
    "tokenizer*",
    "vocab*",
    "merges.txt",
    "1_Pooling/*",
    "2_Normalize/*",
    "README.md",
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="show known models and presets")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--embedding", action="append", default=[])
    parser.add_argument("--reranker", action="append", default=[])
    parser.add_argument("--generation", action="append", default=[])
    parser.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    args = parser.parse_args()

    if args.list:
        for model, description in CATALOGUE.items():
            print(f"{model:45s} {description}")
        for name, models in PRESETS.items():
            print(f"preset {name}: {', '.join(models)}")
        return 0

    wanted = [
        *PRESETS.get(args.preset or "", []),
        *args.embedding,
        *args.reranker,
        *args.generation,
    ]
    if not wanted:
        parser.error("choose --preset or at least one model (see --list)")

    os.environ.pop("HF_HUB_OFFLINE", None)  # this script is the one place that may download
    try:
        from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache
    except ImportError:
        print('huggingface_hub is missing: pip install -e "backend[ml]"', file=sys.stderr)
        return 2

    from fnmatch import fnmatch

    api = HfApi()
    plan = []
    for model in dict.fromkeys(wanted):
        if isinstance(try_to_load_from_cache(model, "config.json"), str):
            print(f"{model}: already in the local cache")
            continue
        info = api.model_info(model, files_metadata=True)
        size = sum(
            (sibling.size or 0)
            for sibling in info.siblings or []
            if any(fnmatch(sibling.rfilename, pattern) for pattern in ALLOW_PATTERNS)
        )
        licence = (info.card_data or {}).get("license") if info.card_data else None
        print(f"{model}: about {size / 1024**2:.0f} MB, licence {licence or 'see model card'}")
        plan.append(model)

    if not plan:
        return 0
    if not args.yes and input("Download these models now? [y/N] ").strip().lower() != "y":
        print("Nothing downloaded.")
        return 1
    for model in plan:
        path = snapshot_download(model, allow_patterns=ALLOW_PATTERNS)
        print(f"{model}: downloaded to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
