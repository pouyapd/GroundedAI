"""Retrieval benchmark: compare retrieval configurations on identical inputs.

Every configuration runs over the same questions, the same chunks and the
same relevance judgments. Embedding models are given separate indexes (their
vectors differ) but chunk ids are identical across indexes, which the runner
checks.
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from groundedai.core.config import Settings
from groundedai.evaluation.dataset import EvalDataset
from groundedai.evaluation.environment import environment_info, hf_model_revision
from groundedai.evaluation.index import IndexedCorpus, build_index, judge_dataset
from groundedai.evaluation.metrics import (
    bootstrap_ci,
    mean,
    ndcg_at_k,
    paired_bootstrap_difference,
    recall_at_k,
    reciprocal_rank,
)
from groundedai.ingestion.chunking import ChunkingConfig
from groundedai.observability.metrics import summarize
from groundedai.persistence.db import session_scope
from groundedai.persistence.search import SqliteSearchBackend
from groundedai.retrieval.embeddings import EmbeddingProvider, build_embedding_provider
from groundedai.retrieval.engine import RetrievalEngine
from groundedai.retrieval.rerank import CrossEncoderReranker, Reranker
from groundedai.retrieval.types import FusionName, RetrievalModeName, RetrievalRequest

Progress = Callable[[str], None]


def _constant[T](value: T) -> Callable[[], T]:
    return lambda: value


@dataclass(frozen=True)
class EmbedderSpec:
    key: str
    provider: str
    model: str
    dim: int
    query_prefix: str = ""


@dataclass(frozen=True)
class ConfigSpec:
    name: str
    mode: str
    embedder: str | None = None
    fusion: str = "rrf"
    alpha: float = 0.5
    rrf_k: int = 60
    reranker: str | None = None
    description: str = ""


@dataclass(frozen=True)
class BenchmarkConfig:
    title: str
    chunk_size: int
    chunk_overlap: int
    k_values: tuple[int, ...]
    candidate_k: int
    rerank_candidates: int
    bootstrap_samples: int
    seed: int
    baseline: str
    embedders: dict[str, EmbedderSpec]
    configurations: list[ConfigSpec] = field(default_factory=list)

    @property
    def primary_k(self) -> int:
        return 5 if 5 in self.k_values else max(self.k_values)

    @classmethod
    def from_file(cls, path: Path) -> BenchmarkConfig:
        raw = json.loads(path.read_text(encoding="utf-8"))
        embedders = {
            key: EmbedderSpec(key=key, **spec) for key, spec in raw.get("embedders", {}).items()
        }
        configurations = [ConfigSpec(**item) for item in raw["configurations"]]
        for item in configurations:
            if item.mode not in ("lexical", "dense", "hybrid"):
                raise ValueError(f"{item.name}: unknown mode {item.mode}")
            if item.mode != "lexical" and item.embedder not in embedders:
                raise ValueError(f"{item.name}: unknown embedder {item.embedder}")
        names = [item.name for item in configurations]
        if raw["baseline"] not in names:
            raise ValueError("baseline must name one of the configurations")
        return cls(
            title=raw.get("title", "Retrieval benchmark"),
            chunk_size=int(raw["chunking"]["chunk_size"]),
            chunk_overlap=int(raw["chunking"]["chunk_overlap"]),
            k_values=tuple(int(k) for k in raw["k_values"]),
            candidate_k=int(raw.get("candidate_k", 30)),
            rerank_candidates=int(raw.get("rerank_candidates", 20)),
            bootstrap_samples=int(raw.get("bootstrap", {}).get("samples", 2000)),
            seed=int(raw.get("bootstrap", {}).get("seed", 0)),
            baseline=raw["baseline"],
            embedders=embedders,
            configurations=configurations,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "chunking": {"chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap},
            "k_values": list(self.k_values),
            "candidate_k": self.candidate_k,
            "rerank_candidates": self.rerank_candidates,
            "bootstrap": {"samples": self.bootstrap_samples, "seed": self.seed},
            "baseline": self.baseline,
        }


def provider_from_spec(spec: EmbedderSpec) -> EmbeddingProvider:
    settings = Settings(
        _env_file=None,
        embedding_provider=spec.provider,
        embedding_model=spec.model,
        embedding_dim=spec.dim,
        embedding_query_prefix=spec.query_prefix,
    )
    return build_embedding_provider(settings)


def _metric_summary(values: list[float], samples: int, seed: int) -> dict[str, Any]:
    interval = bootstrap_ci(values, samples=samples, seed=seed)
    return {
        "mean": round(mean(values) or 0.0, 4),
        "ci95": [round(interval[0], 4), round(interval[1], 4)] if interval else None,
        "n": len(values),
    }


def run_retrieval_benchmark(
    dataset: EvalDataset,
    config: BenchmarkConfig,
    *,
    workdir: Path | None = None,
    provider_factory: Callable[[EmbedderSpec], EmbeddingProvider] = provider_from_spec,
    reranker_factory: Callable[[str], Reranker] = lambda model: CrossEncoderReranker(model),
    progress: Progress | None = None,
) -> dict[str, Any]:
    say = progress or (lambda _message: None)
    chunking = ChunkingConfig(config.chunk_size, config.chunk_overlap)
    temporary = None
    if workdir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="groundedai-bench-", ignore_cleanup_errors=True
        )
        workdir = Path(temporary.name)

    embedders = dict(config.embedders)
    needed = sorted({c.embedder for c in config.configurations if c.embedder})
    if not needed:  # lexical-only run: any embedder works for building the index
        embedders.setdefault("hashing", EmbedderSpec("hashing", "hashing", "v1", 384))
        needed = ["hashing"]
    corpora: dict[str, IndexedCorpus] = {}
    try:
        for key in needed:
            embedder_spec = embedders[key]
            say(
                f"Building index with embedder '{key}' "
                f"({embedder_spec.provider}:{embedder_spec.model})"
            )
            corpora[key] = build_index(
                dataset, provider_factory(embedder_spec), chunking, workdir, key, progress=say
            )
        reference = corpora[needed[0]]
        reference_ids = sorted(chunk.chunk_id for chunk in reference.chunks)
        for corpus in corpora.values():
            if sorted(chunk.chunk_id for chunk in corpus.chunks) != reference_ids:
                raise RuntimeError("chunking differs between indexes; comparison would be unfair")
        judgments = judge_dataset(dataset, reference)
        questions = [q for q in dataset.with_evidence() if q.id in judgments]
        max_k = max(config.k_values)

        rerankers: dict[str, Reranker] = {}
        results: list[dict[str, Any]] = []
        per_query: list[dict[str, Any]] = []
        values_by_config: dict[str, dict[str, list[float]]] = {}
        for spec in config.configurations:
            corpus = corpora[spec.embedder] if spec.embedder else reference
            reranker: Reranker | None = None
            if spec.reranker:
                if spec.reranker not in rerankers:
                    rerankers[spec.reranker] = reranker_factory(spec.reranker)
                reranker = rerankers[spec.reranker]
            engine = RetrievalEngine(
                SqliteSearchBackend(),
                embedding_provider=_constant(corpus.embedding_provider),
                reranker=_constant(reranker),
            )

            def request_for(query: str, spec: ConfigSpec = spec) -> RetrievalRequest:
                return RetrievalRequest(
                    query=query,
                    mode=cast(RetrievalModeName, spec.mode),
                    top_k=max_k,
                    candidate_k=config.candidate_k,
                    fusion=cast(FusionName, spec.fusion),
                    alpha=spec.alpha,
                    rrf_k=spec.rrf_k,
                    use_reranker=bool(spec.reranker),
                    rerank_candidates=config.rerank_candidates,
                )

            say(f"Running configuration {spec.name}")
            with session_scope(corpus.session_factory) as session:  # warm-up, not timed
                engine.retrieve(session, request_for("warm up query"))

            values: dict[str, list[float]] = {"mrr": []}
            for k in config.k_values:
                values[f"recall@{k}"] = []
                values[f"ndcg@{k}"] = []
            latencies: list[float] = []
            for question in questions:
                with session_scope(corpus.session_factory) as session:
                    started = perf_counter()
                    result = engine.retrieve(session, request_for(question.question))
                    latency_ms = (perf_counter() - started) * 1000
                latencies.append(latency_ms)
                ranked = [chunk.chunk_id for chunk in result.chunks]
                evidence_sets = judgments[question.id]
                relevant = set().union(*evidence_sets)
                row: dict[str, Any] = {
                    "configuration": spec.name,
                    "question_id": question.id,
                    "category": question.category,
                    "latency_ms": round(latency_ms, 3),
                    "first_relevant_rank": next(
                        (i for i, cid in enumerate(ranked, 1) if cid in relevant), None
                    ),
                }
                rr = reciprocal_rank(ranked, relevant)
                values["mrr"].append(rr)
                row["mrr"] = round(rr, 4)
                for k in config.k_values:
                    recall = recall_at_k(ranked, evidence_sets, k)
                    ndcg = ndcg_at_k(ranked, relevant, k)
                    values[f"recall@{k}"].append(recall)
                    values[f"ndcg@{k}"].append(ndcg)
                    row[f"recall@{k}"] = round(recall, 4)
                    row[f"ndcg@{k}"] = round(ndcg, 4)
                per_query.append(row)

            values_by_config[spec.name] = values
            results.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "mode": spec.mode,
                    "embedder": spec.embedder,
                    "embedding_model": corpus.embedding_provider.model_id
                    if spec.mode != "lexical"
                    else None,
                    "fusion": spec.fusion if spec.mode == "hybrid" else None,
                    "alpha": spec.alpha if spec.mode == "hybrid" else None,
                    "rrf_k": spec.rrf_k if spec.mode == "hybrid" and spec.fusion == "rrf" else None,
                    "reranker": spec.reranker,
                    "metrics": {
                        name: _metric_summary(series, config.bootstrap_samples, config.seed)
                        for name, series in values.items()
                    },
                    "latency_ms": summarize(latencies),
                }
            )

        comparisons = []
        primary = [f"recall@{config.primary_k}", "mrr", f"ndcg@{config.primary_k}"]
        baseline_values = values_by_config[config.baseline]
        for spec in config.configurations:
            if spec.name == config.baseline:
                continue
            for metric_name in primary:
                diff = paired_bootstrap_difference(
                    values_by_config[spec.name][metric_name],
                    baseline_values[metric_name],
                    samples=config.bootstrap_samples,
                    seed=config.seed,
                )
                comparisons.append(
                    {
                        "configuration": spec.name,
                        "baseline": config.baseline,
                        "metric": metric_name,
                        **{
                            key: round(v, 4) if isinstance(v, float) else v
                            for key, v in diff.items()
                        },
                    }
                )

        embedders_info = {}
        for key, corpus in corpora.items():
            embedder_spec = embedders[key]
            embedders_info[key] = {
                "model_id": corpus.embedding_provider.model_id,
                "dim": embedder_spec.dim,
                "query_prefix": embedder_spec.query_prefix or None,
                "revision": hf_model_revision(embedder_spec.model)
                if embedder_spec.provider == "sentence-transformers"
                else None,
                "index_seconds": corpus.index_seconds,
            }
        generated = datetime.now(UTC)
        return {
            "report_id": f"retrieval-{dataset.manifest.name}-{generated:%Y%m%dT%H%M%SZ}",
            "kind": "retrieval",
            "title": config.title,
            "generated_at": generated.isoformat(timespec="seconds"),
            "measured": True,
            "dataset": {
                "name": dataset.manifest.name,
                "version": dataset.manifest.version,
                "fingerprint": dataset.fingerprint,
                "documents": len(dataset.manifest.documents),
                "questions_total": len(dataset.questions),
                "questions_evaluated": len(questions),
                "chunks": len(reference.chunks),
            },
            "settings": {
                **config.as_dict(),
                "backend": "SQLite: FTS5 BM25 (porter stemmer) + exact cosine scan in NumPy",
                "relevance": "chunk relevant if it covers >= 50% of an evidence quote span",
            },
            "environment": environment_info(),
            "embedders": embedders_info,
            "rerankers": {
                model: {"revision": hf_model_revision(model)} for model in sorted(rerankers)
            },
            "configurations": results,
            "comparisons": comparisons,
            "per_query": per_query,
            "limitations": [
                "Small synthetic corpus written for this project; results are indicative only "
                "and do not transfer to other domains.",
                "95% intervals are percentile bootstrap over questions; with this sample size "
                "many differences between configurations are not distinguishable.",
                "Latency is single-process wall-clock time on the listed CPU, including query "
                "embedding and reranking; the SQLite vector search is an exact linear scan.",
            ],
        }
    finally:
        for corpus in corpora.values():
            corpus.close()
        if temporary is not None:
            temporary.cleanup()
