"""Re-check recorded model outputs with the current answer checks, without running a model.

    python scripts/revalidate_recorded_answers.py REPORT.json [REPORT.json ...]
        [--out evaluation/reports/revalidated] [--query-prefix TEXT]

An answer-evaluation report of scoring version 3 or later stores the model's raw output for
every question. This script rebuilds the evaluation index with the report's settings (dataset,
chunking, embedding model, retrieval mode, top-k, context size, prompt), replays each recorded
output as the model's reply, and scores it with the current checks (claim check, disagreement
check, scoring version 4). It writes a new report; the source report is not changed.

The rebuild is verified before anything is written. The script first replays with the
checks the source run used (``figures-v1``) and requires that, for every question, the
replayed status, citations, grounding and evidence coverage equal the recorded ones; any
difference means the passages are not the ones the model saw, and the script stops.

It then replays with the current checks. Where the current disagreement check would send the
model a different note than the source run did, the recorded output answers a slightly
different prompt; those questions are listed in the report (``model_input_changed``) and
marked on their rows. Latency and token counts are copied from the source report.

Needs the embedding model the source run used (for bge-small: the ML extra and a cached
model). No language model is loaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from langchain_core.callbacks import CallbackManagerForLLMRun  # noqa: E402
from langchain_core.language_models.chat_models import BaseChatModel  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage  # noqa: E402
from langchain_core.outputs import ChatGeneration, ChatResult  # noqa: E402
from pydantic import Field  # noqa: E402

from groundedai.core.config import Settings  # noqa: E402
from groundedai.evaluation.dataset import EvalDataset, load_dataset  # noqa: E402
from groundedai.evaluation.environment import environment_info  # noqa: E402
from groundedai.evaluation.generation_eval import (  # noqa: E402
    _group,
    run_generation_eval,
    summarize_rows,
)
from groundedai.evaluation.index import IndexedCorpus, build_index  # noqa: E402
from groundedai.evaluation.reporting import write_report  # noqa: E402
from groundedai.generation.providers import GenerationProvider  # noqa: E402
from groundedai.ingestion.chunking import ChunkingConfig  # noqa: E402
from groundedai.retrieval.embeddings import build_embedding_provider  # noqa: E402

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
CHECKED_FIELDS = ("status", "citations", "grounding", "evidence_coverage", "answer")
COPIED_FIELDS = ("latency_ms", "generation_ms", "completion_tokens")


class ReplayChatModel(BaseChatModel):
    """Answers each question with the output recorded for it; remembers the prompts it got."""

    outputs: dict[str, str]
    prompts: dict[str, str] = Field(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "replay"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = "\n".join(str(message.content) for message in messages)
        matches = [question for question in self.outputs if question in prompt]
        if not matches:
            raise RuntimeError("no recorded output for this question")
        question = max(matches, key=len)
        self.prompts[question] = prompt
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.outputs[question]))]
        )


def _settings_from(config: dict[str, Any], query_prefix: str) -> Settings:
    provider, _, model = str(config["embedding_model"]).partition(":")
    return Settings(
        _env_file=None,
        embedding_provider=provider,
        embedding_model=model,
        embedding_query_prefix=query_prefix,
        chunk_size=int(config["chunk_size"]),
        chunk_overlap=int(config["chunk_overlap"]),
        hf_offline=True,
    )


def _replay(
    source: dict[str, Any],
    dataset: EvalDataset,
    corpus: IndexedCorpus,
    conflict_check: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    config = source["run_config"]
    rows = {row["question_id"]: row for row in source["per_query"]}
    outputs = {
        question.question: rows[question.id]["model_output"]
        for question in dataset.questions
        if question.id in rows and rows[question.id].get("model_output") is not None
    }
    model = ReplayChatModel(outputs=outputs)
    provider = GenerationProvider(
        name=source["provider"]["name"],
        model=source["provider"]["model"],
        is_llm=True,
        chat_model=model,
    )
    report = run_generation_eval(
        dataset,
        corpus,
        provider,
        retrieval_mode=config["retrieval_mode"],
        fusion=config.get("fusion") or "rrf",
        top_k=int(config["top_k"]),
        max_context_chars=int(config["max_context_chars"]),
        prompt_version=config["prompt_version"],
        conflict_check=conflict_check,  # type: ignore[arg-type]
        run_config={
            "chunk_size": config["chunk_size"],
            "chunk_overlap": config["chunk_overlap"],
            "max_new_tokens": config.get("max_new_tokens"),
            "temperature": config.get("temperature"),
            "limit": None,
        },
    )
    prompts = {
        question.id: model.prompts.get(question.question, "") for question in dataset.questions
    }
    return report, prompts


def _differences(source: dict[str, Any], replayed: dict[str, Any]) -> list[str]:
    recorded = {row["question_id"]: row for row in source["per_query"]}
    problems = []
    for row in replayed["per_query"]:
        before = recorded.get(row["question_id"])
        if before is None or before["status"] == "error":
            continue
        for field in CHECKED_FIELDS:
            if before.get(field) != row.get(field):
                problems.append(
                    f"{row['question_id']} {field}: recorded {before.get(field)!r}, "
                    f"replayed {row.get(field)!r}"
                )
    return problems


def revalidate(path: Path, out_dir: Path, query_prefix: str | None) -> Path:
    source = json.loads(path.read_text(encoding="utf-8"))
    config = source.get("run_config") or {}
    if source.get("kind") != "generation" or int(config.get("scoring_version") or 0) < 3:
        raise SystemExit(f"{path.name}: not an answer evaluation with recorded outputs")
    if config.get("reranker"):
        raise SystemExit(f"{path.name}: runs with a reranker are not supported")
    dataset = load_dataset(REPO_ROOT / "evaluation" / "datasets" / source["dataset"]["name"])
    if dataset.fingerprint != source["dataset"]["fingerprint"]:
        raise SystemExit(f"{path.name}: the dataset changed since the run")
    if query_prefix is None:
        query_prefix = BGE_QUERY_PREFIX if "bge" in str(config["embedding_model"]) else ""
    settings = _settings_from(config, query_prefix)
    embedding = build_embedding_provider(settings)
    with tempfile.TemporaryDirectory(
        prefix="groundedai-replay-", ignore_cleanup_errors=True
    ) as tmp:
        corpus = build_index(
            dataset,
            embedding,
            ChunkingConfig(settings.chunk_size, settings.chunk_overlap),
            Path(tmp),
            "replay",
        )
        try:
            original, original_prompts = _replay(source, dataset, corpus, "figures-v1")
            problems = _differences(source, original)
            if problems:
                print("\n".join(problems[:20]))
                raise SystemExit(
                    f"{path.name}: the rebuilt passages do not reproduce the recorded run "
                    f"({len(problems)} difference(s)); nothing was written"
                )
            current, current_prompts = _replay(source, dataset, corpus, "disagreement-v1")
        finally:
            corpus.close()

    recorded = {row["question_id"]: row for row in source["per_query"]}
    changed = sorted(
        qid
        for qid, prompt in current_prompts.items()
        if prompt and prompt != original_prompts.get(qid, "")
    )
    rows = []
    for row in current["per_query"]:
        before = recorded.get(row["question_id"], {})
        if before.get("status") == "error":
            row = {**before, "replayed": False}
        else:
            for field in COPIED_FIELDS:
                row[field] = before.get(field)
            row["replayed"] = True
            row["model_input_changed"] = row["question_id"] in changed
        rows.append(row)

    generated = datetime.now(UTC)
    report = {
        **current,
        "report_id": f"revalidated-{source['report_id']}-{generated:%Y%m%dT%H%M%SZ}",
        "title": (
            f"Recorded outputs re-checked: {source['provider']['name']} "
            f"({source['provider']['model']}, {config['prompt_version']})"
        ),
        "generated_at": generated.isoformat(timespec="seconds"),
        "provider": source["provider"],
        "summary": summarize_rows(rows),
        "by_category": {
            category: summarize_rows(items) for category, items in _group(rows).items()
        },
        "per_query": rows,
        "environment": environment_info(),
        "revalidation": {
            "source_report_id": source["report_id"],
            "source_generated_at": source["generated_at"],
            "source_code_version": (source.get("environment") or {}).get("code"),
            "model_run": False,
            "embedding_query_prefix": query_prefix,
            "reconstruction_verified": True,
            "questions_replayed": sum(1 for row in rows if row.get("replayed")),
            "model_input_changed": changed,
            "note": (
                "The model was not run. Its recorded raw outputs were passed through the current "
                "checks. Before that, a replay with the source run's checks reproduced the "
                "recorded status, citations, grounding and evidence coverage of every question. "
                "Questions in model_input_changed would get a different application note now, "
                "so a new model run could answer them differently. Latency and token counts are "
                "copied from the source report."
            ),
        },
        "limitations": [
            *current["limitations"],
            "Recorded outputs replayed through the current checks; the model was not run.",
        ],
    }
    paths = write_report(report, out_dir)
    print(
        f"{path.name}: {report['revalidation']['questions_replayed']} questions replayed, "
        f"{len(changed)} with a changed note -> {paths['json']}"
    )
    return paths["json"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "evaluation" / "reports" / "revalidated"
    )
    parser.add_argument(
        "--query-prefix",
        default=None,
        help="embedding query prefix of the source run (default: the bge instruction for bge "
        "models, otherwise none); a wrong value makes the reconstruction check fail",
    )
    args = parser.parse_args()
    for path in args.reports:
        revalidate(path, args.out, args.query_prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
