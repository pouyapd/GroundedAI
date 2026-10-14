"""The benchmark runners on the bundled dataset (offline: hashing embedder, extractive demo)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from groundedai.core.config import Settings
from groundedai.evaluation.dataset import load_dataset
from groundedai.evaluation.generation_eval import run_generation_eval
from groundedai.evaluation.index import build_index
from groundedai.evaluation.reporting import list_reports, load_report, write_report
from groundedai.evaluation.retrieval_benchmark import BenchmarkConfig, run_retrieval_benchmark
from groundedai.generation.providers import build_generation_provider
from groundedai.ingestion.chunking import ChunkingConfig
from groundedai.retrieval.embeddings import EmbeddingProvider, HashingEmbeddings
from tests.conftest import DATASET_DIR, REPO_ROOT

OFFLINE_CONFIG = REPO_ROOT / "evaluation" / "configs" / "retrieval_offline.json"


@pytest.fixture(scope="module")
def offline_report(tmp_path_factory: pytest.TempPathFactory) -> dict:  # type: ignore[type-arg]
    dataset = load_dataset(DATASET_DIR)
    config = BenchmarkConfig.from_file(OFFLINE_CONFIG)
    return run_retrieval_benchmark(dataset, config, workdir=tmp_path_factory.mktemp("bench"))


def test_retrieval_report_structure(offline_report: dict) -> None:  # type: ignore[type-arg]
    assert offline_report["kind"] == "retrieval"
    assert offline_report["measured"] is True
    assert offline_report["dataset"]["questions_evaluated"] == 43
    names = [item["name"] for item in offline_report["configurations"]]
    assert names == ["A-lexical-bm25", "B0-dense-hashing", "C0-hybrid-rrf-hashing"]
    for item in offline_report["configurations"]:
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5"):
            value = item["metrics"][metric]
            assert 0.0 <= value["mean"] <= 1.0
            assert value["n"] == 43
            low, high = value["ci95"]
            assert low <= value["mean"] <= high
        assert item["latency_ms"]["n"] == 43
    assert len(offline_report["comparisons"]) == 6  # 2 configs x 3 primary metrics
    assert offline_report["environment"]["python"]
    assert len(offline_report["per_query"]) == 3 * 43


def test_retrieval_metrics_are_reproducible(
    offline_report: dict,  # type: ignore[type-arg]
    tmp_path: Path,
) -> None:
    dataset = load_dataset(DATASET_DIR)
    again = run_retrieval_benchmark(
        dataset, BenchmarkConfig.from_file(OFFLINE_CONFIG), workdir=tmp_path
    )
    first = {c["name"]: c["metrics"] for c in offline_report["configurations"]}
    second = {c["name"]: c["metrics"] for c in again["configurations"]}
    assert first == second


def test_report_files_round_trip(offline_report: dict, tmp_path: Path) -> None:  # type: ignore[type-arg]
    paths = write_report(offline_report, tmp_path)
    assert json.loads(paths["json"].read_text(encoding="utf-8"))["report_id"]
    assert "| A-lexical-bm25 |" in paths["markdown"].read_text(encoding="utf-8")
    assert paths["csv"].read_text(encoding="utf-8").startswith("configuration,question_id")
    listed = list_reports(tmp_path)
    assert [r["report_id"] for r in listed] == [offline_report["report_id"]]
    assert load_report(tmp_path, offline_report["report_id"]) is not None
    assert load_report(tmp_path, "../../etc/passwd") is None


def test_generation_eval_with_extractive_baseline(settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    dataset = load_dataset(DATASET_DIR)
    provider = EmbeddingProvider("hashing", "v1-384", 384, HashingEmbeddings(384))
    corpus = build_index(dataset, provider, ChunkingConfig(400, 60), tmp_path, "gen")
    try:
        report = run_generation_eval(dataset, corpus, build_generation_provider(settings))
    finally:
        corpus.close()
    summary = report["summary"]
    assert summary["n"] == 56
    assert summary["errors"] == 0
    assert report["provider"]["is_llm"] is False
    assert any("not a language model" in text for text in report["limitations"])
    # The extractive baseline only quotes sources, so it cannot invent citation labels.
    assert summary["citation_validity"] == 1.0
    # It never repeats instruction-like sentences from documents.
    assert summary["injection_resistance"] == 1.0
    assert set(report["by_category"]) >= {"factual", "unanswerable", "injection"}
    # Scoring version 4: the checks used are recorded, and the claim check accepts every
    # sentence the baseline quotes from a passage with its label.
    config = report["run_config"]
    assert (config["scoring_version"], config["claim_check"]) == (4, "claims-v1")
    assert config["conflict_check"] == "disagreement-v1"
    assert summary["claim_match_rate"] == 1.0
    assert summary["cited_claim_precision"] == 1.0
    row = report["per_query"][0]
    assert {source["doc"] for source in row["sources"]} <= {
        document.key for document in dataset.manifest.documents
    }
    conflict_rows = [r for r in report["per_query"] if r["expected_behavior"] == "flag_conflict"]
    assert all("conflict_surfaced" in r for r in conflict_rows)
    assert all("false_conflict" in r for r in report["per_query"] if r not in conflict_rows)


TIMING = {"latency_ms", "generation_ms"}
PIPELINE_ENV = {
    "GROUNDEDAI_CANDIDATE_K": "1",
    "GROUNDEDAI_RRF_K": "7",
    "GROUNDEDAI_HYBRID_ALPHA": "0.8",
    "GROUNDEDAI_RERANK_CANDIDATES": "9",
    "GROUNDEDAI_UNVERIFIED_ANSWERS": "withhold",
    "GROUNDEDAI_MAX_QUESTION_CHARS": "500",
}


def _scored(report: dict) -> list[dict]:  # type: ignore[type-arg]
    return [{k: v for k, v in row.items() if k not in TIMING} for row in report["per_query"]]


def _without_variable_names(config: dict) -> dict:  # type: ignore[type-arg]
    return {k: v for k, v in config.items() if k != "environment_variables"}


def test_environment_settings_are_recorded_and_reproduce_the_run(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Regression: GROUNDEDAI_* variables changed what the answer pipeline did, unrecorded.
    dataset = load_dataset(DATASET_DIR)
    embedder = EmbeddingProvider("hashing", "v1-384", 384, HashingEmbeddings(384))
    corpus = build_index(dataset, embedder, ChunkingConfig(400, 60), tmp_path, "env")
    provider = build_generation_provider(settings)
    try:
        for name, value in PIPELINE_ENV.items():
            monkeypatch.setenv(name, value)
        recorded = run_generation_eval(dataset, corpus, provider, limit=8)
        config = recorded["run_config"]
        assert config["pipeline_settings"] == {
            "candidate_k": 1,
            "rrf_k": 7,
            "hybrid_alpha": 0.8,
            "rerank_candidates": 9,
            "unverified_answers": "withhold",
            "max_question_chars": 500,
        }
        assert config["environment_variables"] == sorted(PIPELINE_ENV)
        assert config["timeout_s"] == 300.0

        for name in PIPELINE_ENV:
            monkeypatch.delenv(name)
        rerun = run_generation_eval(
            dataset, corpus, provider, limit=8, pipeline_settings=config["pipeline_settings"]
        )
        assert _scored(rerun) == _scored(recorded)
        assert _without_variable_names(rerun["run_config"]) == _without_variable_names(config)
        assert rerun["run_config"]["environment_variables"] == []

        # Control: without the recorded values the defaults apply and the results differ, so
        # the comparison above is not vacuous.
        default = run_generation_eval(dataset, corpus, provider, limit=8)
        assert default["run_config"]["pipeline_settings"]["candidate_k"] == 20
        assert _scored(default) != _scored(recorded)
        with pytest.raises(ValueError, match="not pipeline settings: top_k"):
            run_generation_eval(dataset, corpus, provider, limit=1, pipeline_settings={"top_k": 3})
    finally:
        corpus.close()


def test_the_cli_reruns_an_evaluation_from_a_reports_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from groundedai.cli import main
    from groundedai.core.config import get_settings

    monkeypatch.chdir(tmp_path)  # no .env file here
    shell = {
        "GROUNDEDAI_CHUNK_SIZE": "300",
        "GROUNDEDAI_CHUNK_OVERLAP": "40",
        "GROUNDEDAI_EMBEDDING_DIM": "256",
        "GROUNDEDAI_FUSION": "weighted",
        "GROUNDEDAI_HYBRID_ALPHA": "0.3",
        "GROUNDEDAI_CANDIDATE_K": "6",
        "GROUNDEDAI_LOG_LEVEL": "WARNING",
    }
    argv = ["eval", "generation", "--limit", "6", "--dataset-dir", str(DATASET_DIR)]

    def run(out: str, *extra: str) -> tuple[Path, dict]:  # type: ignore[type-arg]
        get_settings.cache_clear()
        assert main([*argv, "--out", str(tmp_path / out), *extra]) == 0
        (path,) = (tmp_path / out).glob("generation-*.json")
        return path, json.loads(path.read_text(encoding="utf-8"))

    try:
        for name, value in shell.items():
            monkeypatch.setenv(name, value)
        first_path, first = run("first")
        recorded = first["run_config"]
        assert {key: recorded["settings"][key] for key in ("chunk_size", "chunk_overlap")} == {
            "chunk_size": 300,
            "chunk_overlap": 40,
        }
        assert (recorded["settings"]["embedding_dim"], recorded["settings"]["fusion"]) == (
            256,
            "weighted",
        )
        assert recorded["pipeline_settings"]["hybrid_alpha"] == 0.3
        assert recorded["pipeline_settings"]["candidate_k"] == 6
        assert recorded["environment_variables"] == sorted(shell)

        for name in shell:
            monkeypatch.delenv(name)
        _, again = run("again", "--settings-from", str(first_path))
        assert "Differs from the recorded run" not in capsys.readouterr().out
        assert _scored(again) == _scored(first)
        assert _without_variable_names(again["run_config"]) == _without_variable_names(recorded)

        # Control: the same command without --settings-from uses the defaults.
        _, plain = run("plain")
        assert plain["run_config"]["settings"]["chunk_size"] == 800
        assert _scored(plain) != _scored(first)

        # A different option is applied, and reported as a difference from the recording.
        run("other", "--settings-from", str(first_path), "--top-k", "3")
        assert "Differs from the recorded run: top_k 3 (recorded 5)" in capsys.readouterr().out
    finally:
        get_settings.cache_clear()


def test_generation_eval_resumes_from_checkpoint(settings, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    # Regression: an interrupted CPU run lost every answer because results were only
    # written at the end.
    from groundedai.evaluation.generation_eval import CheckpointMismatchError
    from groundedai.generation.providers import GenerationProvider

    dataset = load_dataset(DATASET_DIR)
    embedder = EmbeddingProvider("hashing", "v1-384", 384, HashingEmbeddings(384))
    corpus = build_index(dataset, embedder, ChunkingConfig(400, 60), tmp_path, "resume")
    base = build_generation_provider(settings)
    calls: list[str] = []

    class CountingModel(type(base.chat_model)):  # type: ignore[misc]
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):  # type: ignore[no-untyped-def]
            calls.append("x")
            return super()._generate(messages, stop, run_manager, **kwargs)

    provider = GenerationProvider(base.name, base.model, base.is_llm, CountingModel())
    checkpoint = tmp_path / "run.jsonl"
    try:
        full = run_generation_eval(dataset, corpus, provider, limit=4, checkpoint=checkpoint)
        lines = checkpoint.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1 + 4  # header + one row per question
        # Simulate an interruption after two questions, with a half-written third line.
        checkpoint.write_text("\n".join(lines[:3]) + "\n" + lines[3][:20], encoding="utf-8")
        calls.clear()
        resumed = run_generation_eval(dataset, corpus, provider, limit=4, checkpoint=checkpoint)
        assert resumed["resumed_from_checkpoint"] == 2
        timing = {"latency_ms", "generation_ms"}

        def scored(rows: list[dict]) -> list[dict]:  # type: ignore[type-arg]
            return [{k: v for k, v in row.items() if k not in timing} for row in rows]

        assert scored(resumed["per_query"]) == scored(full["per_query"])
        assert len(calls) <= 2  # only the two unfinished questions were asked again
        with pytest.raises(CheckpointMismatchError):
            run_generation_eval(dataset, corpus, provider, top_k=3, limit=4, checkpoint=checkpoint)
    finally:
        corpus.close()
