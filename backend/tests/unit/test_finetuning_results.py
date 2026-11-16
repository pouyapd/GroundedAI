"""The recorded LoRA run (evaluation/reports/finetuning-2026-11/run1) and the figures quoted.

The JSON files are the run's own records, copied unchanged from the Colab run folder; MANIFEST.json
holds their SHA-256 hashes. These tests keep the files unchanged and keep the figures quoted in
docs/FINETUNING.md, docs/PORTFOLIO_SUMMARY.md and the README tied to them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
RUN = REPO / "evaluation" / "reports" / "finetuning-2026-11" / "run1"


def load(name: str) -> Any:
    return json.loads((RUN / name).read_text(encoding="utf-8"))


def test_committed_files_match_the_manifest() -> None:
    manifest = load("MANIFEST.json")
    committed = [f for f in manifest["files"] if f["in_repository"]]
    assert len(committed) == 24
    for entry in committed:
        data = (RUN / entry["path"]).read_bytes()
        assert len(data) == entry["bytes"], entry["path"]
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["path"]
    weights = next(f for f in manifest["files"] if f["path"].endswith("adapter_model.safetensors"))
    assert not weights["in_repository"]
    assert not (RUN / weights["path"]).exists()


def test_training_run() -> None:
    result, config, hardware = (
        load("result.json"),
        load("training_config.json"),
        load("hardware.json"),
    )
    assert result["finished"]
    assert result["resumed_from"] is None
    assert result["global_steps"] == config["estimated_optimizer_steps"] == 180
    assert result["best_checkpoint"] == "checkpoint-180"
    assert round(result["first_validation_loss"], 3) == 0.150
    assert round(result["best_validation_loss"], 3) == 0.064
    assert round(result["train_runtime_s"] / 60, 1) == 40.3
    assert result["peak_gpu_memory_gb"] == 13.15
    assert hardware["gpu"] == "Tesla T4"
    assert hardware["gpu_memory_gb"] == 14.6
    assert config["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert config["lora"]["rank"] == 16
    assert config["lora"]["alpha"] == 32
    assert config["learning_rate"] == 1e-4
    assert config["gradient_accumulation"] == 8
    assert config["epochs"] == 2.0
    assert config["seed"] == 13
    assert config["max_length"] == 2048
    assert (config["train_examples"], config["validation_examples"]) == (713, 269)
    assert config["trainable_parameters"] == 18_464_768
    smoke = load("smoke/smoke_result.json")
    assert smoke["passed"]
    assert smoke["device"] == "cuda"
    assert len(smoke["stages"]) == 8
    assert all(s["passed"] for s in smoke["stages"])
    assert load("reports/merge-check.json") == {"same_answer_as_adapter": 20, "questions": 20}


@pytest.mark.parametrize(
    ("name", "n", "old", "new", "low", "high", "distinguishable"),
    [
        ("test", 40, 34, 23, -0.425, -0.15, True),
        ("nist-val", 15, 14, 6, -0.8, -0.2, True),
        ("nist-val-relevance", 7, 5, 1, -0.8571, -0.1429, True),
        ("synthetic-eval", 166, 151, 147, -0.0904, 0.0422, False),
    ],
)
def test_correct_answers(
    name: str, n: int, old: int, new: int, low: float, high: float, distinguishable: bool
) -> None:
    measure = load(f"reports/{name}-comparison.json")["measures"]["correct (should answer)"]
    assert measure["n"] == n
    assert round(measure["old_mean"] * n) == old  # means are rounded to 4 places in the report
    assert round(measure["new_mean"] * n) == new
    assert (measure["ci95_low"], measure["ci95_high"]) == (low, high)
    assert measure["distinguishable"] is distinguishable
    for key in ("recall@5", "mrr@10"):
        assert load(f"reports/{name}-comparison.json")["measures"][key]["mean_diff"] == 0.0


def test_protected_test_set() -> None:
    base, lora = load("reports/test-base.json"), load("reports/test-lora.json")
    assert base["dataset"]["name"] == lora["dataset"]["name"] == "groundedai-nist-v1"
    assert lora["model"]["adapter_loaded"]
    assert base["model"]["adapter"] is None
    assert (
        base["summary"]["answers"]["false_refusals"],
        lora["summary"]["answers"]["false_refusals"],
    ) == (1, 8)
    assert (base["summary"]["abstention"]["refused"], lora["summary"]["abstention"]["refused"]) == (
        5,
        6,
    )
    assert base["summary"]["abstention"]["should_refuse"] == 6
    assert (base["extra"]["format_valid"], lora["extra"]["format_valid"]) == (27, 53)
    assert base["extra"]["format_checked"] == lora["extra"]["format_checked"] == 53
    citations = base["summary"]["citations"], lora["summary"]["citations"]
    assert [c["answers_without_valid_label"] for c in citations] == [24, 0]
    assert [c["precision"]["n"] for c in citations] == [21, 34]
    assert [c["recall"]["n"] for c in citations] == [43, 34]
    assert [c["precision"]["mean"] for c in citations] == [0.4143, 0.7941]
    assert [c["recall"]["mean"] for c in citations] == [0.3023, 0.75]
    comparison = load("reports/test-comparison.json")
    assert comparison["measures"]["correct (should answer)"]["mean_diff"] == -0.275
    assert len(comparison["changed_answers"]["lost"]) == 11
    assert comparison["changed_answers"]["gained"] == []


def test_synthetic_refusals() -> None:
    measure = load("reports/synthetic-eval-comparison.json")["measures"]["refused (should refuse)"]
    assert measure["n"] == 62
    assert (round(measure["old_mean"] * 62), round(measure["new_mean"] * 62)) == (15, 57)
    assert measure["distinguishable"] is True
