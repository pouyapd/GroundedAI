"""Command-line interface: ``groundedai <command>``.

Commands work directly against the configured database (no server needed).
Evaluation commands build their own temporary indexes and never modify it.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

from groundedai.core.config import Settings, get_settings
from groundedai.core.errors import GroundedAIError


def _print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _services(settings: Settings) -> Any:
    from groundedai.services.container import build_services

    return build_services(settings)


# --------------------------------------------------------------------- commands
def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "groundedai.main:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_migrate(_args: argparse.Namespace) -> int:
    from groundedai.persistence.migrate import upgrade_database

    settings = get_settings()
    upgrade_database(settings.resolved_database_url, settings.embedding_dim)
    print("Database is at the latest migration.")
    return 0


def expand_paths(raw_paths: list[str]) -> list[Path]:
    """Files as given, plus the supported files inside any directory (sorted, not recursive)."""
    from groundedai.ingestion.validation import MEDIA_TYPES

    paths: list[Path] = []
    for raw in raw_paths:
        path = Path(raw)
        if path.is_dir():
            paths.extend(
                sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_TYPES)
            )
        else:
            paths.append(path)
    return paths


def cmd_ingest(args: argparse.Namespace) -> int:
    services = _services(get_settings())
    failures = 0
    try:
        for path in expand_paths(args.paths):
            try:
                row = services.ingestion.ingest_bytes(path.name, None, path.read_bytes())
            except GroundedAIError as exc:
                print(f"{path.name}: rejected ({exc.code}) {exc.message}")
                failures += 1
                continue
            except OSError as exc:
                print(f"{path.name}: cannot read file ({exc.strerror})")
                failures += 1
                continue
            if row.status == "ready":
                print(f"{path.name}: ready as {row.id} ({row.chunk_count} chunks)")
            else:
                print(f"{path.name}: {row.status} ({row.error_code}) {row.error_message}")
                failures += 1
    finally:
        services.close()
    return 1 if failures else 0


def cmd_ask(args: argparse.Namespace) -> int:
    from groundedai.schemas.query import QueryRequest

    services = _services(get_settings())
    try:
        response = services.qa.answer(
            QueryRequest(
                question=args.question,
                top_k=args.top_k,
                retrieval_mode=args.mode,
                include_diagnostics=args.json,
            )
        )
    finally:
        services.close()
    if args.json:
        _print_json(response.model_dump())
        return 0
    mode = "LLM" if response.provider.is_llm else "DEMO (extractive, not an LLM)"
    print(f"[{response.status}] provider={response.provider.name} ({mode})")
    print(response.answer or response.abstention_reason or "")
    for source in response.sources:
        if source.cited:
            pages = f" p.{source.page_start}" if source.page_start else ""
            print(f"  [{source.label}] {source.document_title}{pages}: {source.excerpt}")
    for warning in response.warnings:
        print(f"  warning: {warning}")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    from groundedai.schemas.query import RetrieveRequest

    services = _services(get_settings())
    try:
        response = services.qa.retrieve(
            RetrieveRequest(query=args.query, top_k=args.top_k, retrieval_mode=args.mode)
        )
    finally:
        services.close()
    if args.json:
        _print_json(response.model_dump())
        return 0
    for item in response.results:
        scores = item.scores
        print(
            f"{item.rank}. {item.document_title} [{item.chunk_id}] fused={scores.fused} "
            f"lexical_rank={scores.lexical_rank} dense_rank={scores.dense_rank} "
            f"terms={','.join(item.matched_terms)}"
        )
    print(f"timings: {response.diagnostics.timings_ms}")
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    from groundedai.evaluation.dataset import load_dataset
    from groundedai.evaluation.reporting import render_markdown, write_report
    from groundedai.evaluation.retrieval_benchmark import BenchmarkConfig, run_retrieval_benchmark

    settings = get_settings()
    dataset = load_dataset(Path(args.dataset_dir) if args.dataset_dir else settings.dataset_dir)
    config = BenchmarkConfig.from_file(Path(args.config))
    with tempfile.TemporaryDirectory(prefix="groundedai-bench-", ignore_cleanup_errors=True) as tmp:
        report = run_retrieval_benchmark(
            dataset, config, workdir=Path(tmp), progress=lambda message: print(message, flush=True)
        )
    paths = write_report(report, Path(args.out))
    print(render_markdown(report))
    print("Written:", ", ".join(str(path) for path in paths.values()))
    return 0


def cmd_eval_generation(args: argparse.Namespace) -> int:
    from groundedai.evaluation.dataset import load_dataset
    from groundedai.evaluation.generation_eval import run_generation_eval
    from groundedai.evaluation.index import build_index
    from groundedai.evaluation.reporting import render_markdown, write_report
    from groundedai.generation.providers import build_generation_provider
    from groundedai.ingestion.chunking import ChunkingConfig
    from groundedai.retrieval.embeddings import build_embedding_provider
    from groundedai.retrieval.rerank import CrossEncoderReranker

    base = get_settings()
    overrides: dict[str, Any] = {"generation_provider": args.provider}
    if args.prompt_version:
        overrides["prompt_version"] = args.prompt_version
    if args.model:
        key = "ollama_model" if args.provider == "ollama" else "hf_generation_model"
        overrides[key] = args.model
    settings = base.model_copy(update=overrides)
    dataset = load_dataset(Path(args.dataset_dir) if args.dataset_dir else settings.dataset_dir)
    embedding = build_embedding_provider(settings)
    provider = build_generation_provider(settings)
    reranker = (
        CrossEncoderReranker(settings.reranker_model, offline=settings.hf_offline)
        if args.rerank
        else None
    )
    with tempfile.TemporaryDirectory(prefix="groundedai-gen-", ignore_cleanup_errors=True) as tmp:
        corpus = build_index(
            dataset,
            embedding,
            ChunkingConfig(settings.chunk_size, settings.chunk_overlap),
            Path(tmp),
            "generation",
            progress=lambda message: print(message, flush=True),
        )
        try:
            report = run_generation_eval(
                dataset,
                corpus,
                provider,
                retrieval_mode=args.mode,
                fusion=settings.fusion,
                top_k=args.top_k,
                reranker=reranker,
                max_context_chars=settings.max_context_chars,
                timeout_s=args.timeout,
                limit=args.limit,
                progress=lambda message: print(message, flush=True),
                checkpoint=Path(args.checkpoint) if args.checkpoint else None,
                prompt_version=settings.prompt_version,
                run_config={
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                    "max_new_tokens": settings.max_new_tokens,
                    "temperature": settings.temperature,
                    "limit": args.limit,
                },
            )
        finally:
            corpus.close()
    paths = write_report(report, Path(args.out))
    print(render_markdown(report))
    print("Written:", ", ".join(str(path) for path in paths.values()))
    return 0


def cmd_status(_args: argparse.Namespace) -> int:
    services = _services(get_settings())
    try:
        _print_json(
            {
                "readiness": services.status.readiness().model_dump(),
                "system": services.status.system_status().model_dump(),
            }
        )
    finally:
        services.close()
    return 0


# ------------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="groundedai", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="run the API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    sub.add_parser("migrate", help="apply database migrations").set_defaults(func=cmd_migrate)
    sub.add_parser("status", help="readiness and provider status").set_defaults(func=cmd_status)

    ingest = sub.add_parser("ingest", help="ingest files or directories synchronously")
    ingest.add_argument("paths", nargs="+")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="answer a question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=None)
    ask.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default=None)
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    retrieve = sub.add_parser("retrieve", help="inspect retrieval for a query")
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=None)
    retrieve.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default=None)
    retrieve.add_argument("--json", action="store_true")
    retrieve.set_defaults(func=cmd_retrieve)

    evaluate = sub.add_parser("eval", help="offline evaluation")
    eval_sub = evaluate.add_subparsers(dest="eval_command", required=True)

    retrieval = eval_sub.add_parser("retrieval", help="compare retrieval configurations")
    retrieval.add_argument("--config", default="evaluation/configs/retrieval_offline.json")
    retrieval.add_argument("--dataset-dir", default=None)
    retrieval.add_argument("--out", default="evaluation/reports/local")
    retrieval.set_defaults(func=cmd_eval_retrieval)

    generation = eval_sub.add_parser("generation", help="end-to-end answer evaluation")
    generation.add_argument(
        "--provider",
        choices=["extractive-demo", "ollama", "huggingface-local"],
        default="extractive-demo",
    )
    generation.add_argument("--model", default=None, help="override the provider's model")
    generation.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="hybrid")
    generation.add_argument("--top-k", type=int, default=5)
    generation.add_argument("--rerank", action="store_true")
    generation.add_argument("--limit", type=int, default=None)
    generation.add_argument(
        "--prompt-version",
        choices=["answer-v1", "answer-v2"],
        default=None,
        help="answer prompt (default: GROUNDEDAI_PROMPT_VERSION or answer-v2)",
    )
    generation.add_argument("--timeout", type=float, default=300.0)
    generation.add_argument("--dataset-dir", default=None)
    generation.add_argument("--out", default="evaluation/reports/local")
    generation.add_argument(
        "--checkpoint",
        default=None,
        help="JSONL file: save each scored question as it finishes and resume from it",
    )
    generation.set_defaults(func=cmd_eval_generation)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except GroundedAIError as exc:
        print(f"error ({exc.code}): {exc.message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
