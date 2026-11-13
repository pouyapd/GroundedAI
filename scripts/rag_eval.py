"""Run the end-to-end RAG evaluation (groundedai.evaluation.rag_eval) and write a JSON report.

    python scripts/fetch_nist_corpus.py
    python scripts/rag_eval.py --dataset evaluation/datasets/groundedai-nist-val-v1 \
        --generator extractive --out report.json
    python scripts/rag_eval.py ... --generator none          # retrieval metrics only
    python scripts/rag_eval.py ... --generator ollama        # GROUNDEDAI_OLLAMA_MODEL, local
    python scripts/rag_eval.py ... --generator hf --hf-model <dir or hub id>
    python scripts/rag_eval.py ... --set chunk_size=600 \
        --set embedding_provider=sentence-transformers

``--set`` takes any GroundedAI setting (the names in .env.example without the prefix). Each
report records the settings, code version, hardware and power source. Validation datasets are
for choosing settings; run the test dataset only with settings fixed in advance.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def _value(text: str) -> Any:
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--generator", choices=["none", "extractive", "ollama", "hf"], default="extractive"
    )
    parser.add_argument("--hf-model", help="model directory or hub id for --generator hf")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    import psutil

    from groundedai.core.config import Settings
    from groundedai.evaluation.dataset import load_dataset
    from groundedai.evaluation.environment import code_version, environment_info
    from groundedai.evaluation.index import build_index
    from groundedai.evaluation.rag_eval import run_rag_eval
    from groundedai.generation.providers import build_generation_provider
    from groundedai.ingestion.chunking import ChunkingConfig
    from groundedai.retrieval.embeddings import build_embedding_provider
    from groundedai.retrieval.rerank import CrossEncoderReranker

    overrides = dict(item.split("=", 1) for item in args.set)
    values = {name: _value(value) for name, value in overrides.items()}
    if args.generator == "ollama":
        values["generation_provider"] = "ollama"
    elif args.generator == "hf":
        values["generation_provider"] = "huggingface-local"
        if args.hf_model:
            values["hf_generation_model"] = args.hf_model
    values.setdefault("generation_timeout_s", 600.0)
    values.setdefault("max_concurrent_generations", 1)
    values.setdefault("generation_queue_timeout_s", 600.0)
    workdir = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    settings = Settings(_env_file=None, data_dir=workdir, rate_limit_enabled=False, **values)

    dataset = load_dataset(args.dataset)
    process = psutil.Process()
    started = time.perf_counter()
    embedder = build_embedding_provider(settings)
    corpus = build_index(
        dataset,
        embedder,
        ChunkingConfig(settings.chunk_size, settings.chunk_overlap, settings.text_cleanup),
        workdir,
        dataset.manifest.name,
        progress=print,
    )
    index_seconds = time.perf_counter() - started
    provider = None
    load_seconds = None
    if args.generator != "none":
        started = time.perf_counter()
        provider = build_generation_provider(settings)
        if args.generator == "hf":  # load the weights now, so load time is measured apart
            from langchain_core.messages import HumanMessage

            provider.chat_model.invoke([HumanMessage(content="Hi")])
        load_seconds = round(time.perf_counter() - started, 2)
    reranker = (
        CrossEncoderReranker(settings.reranker_model, offline=settings.hf_offline)
        if settings.reranker == "cross-encoder"
        else None
    )
    report = run_rag_eval(
        dataset, corpus, provider, settings, reranker=reranker, progress=print, limit=args.limit
    )
    report.update(
        label=args.label,
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        generator=args.generator,
        provider={"name": provider.name, "model": provider.model, "is_llm": provider.is_llm}
        if provider
        else None,
        settings_overrides=overrides,
        settings={
            name: str(getattr(settings, name))
            for name in (
                "chunk_size",
                "chunk_overlap",
                "embedding_provider",
                "embedding_model",
                "retrieval_mode",
                "fusion",
                "hybrid_alpha",
                "rrf_k",
                "top_k",
                "candidate_k",
                "reranker",
                "reranker_model",
                "prompt_version",
                "claim_check",
                "conflict_check",
                "max_context_chars",
                "generation_provider",
                "ollama_model",
                "hf_generation_model",
                "temperature",
                "max_new_tokens",
            )
        },
        cost={
            "index_seconds": round(index_seconds, 2),
            "model_load_seconds": load_seconds,
            "peak_rss_mb": round(process.memory_info().rss / 2**20, 1),
            "chunks_indexed": len(corpus.chunks),
        },
        environment=environment_info(),
        code=code_version(),
    )
    corpus.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(report["summary"], indent=1)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
