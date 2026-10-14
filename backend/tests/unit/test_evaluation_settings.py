import json
from pathlib import Path

import pytest

from groundedai.cli import main
from groundedai.core.config import Settings
from groundedai.evaluation.generation_eval import (
    ARGUMENT_SETTINGS,
    INDEX_AND_MODEL_SETTINGS,
    NON_RESULT_SETTINGS,
    PIPELINE_SETTINGS,
    environment_variable_names,
)


def test_every_setting_is_classified_for_the_evaluation_record() -> None:
    # A new setting must be placed in one group: either it is recorded with every evaluation
    # run, or it is listed as one that cannot change results.
    groups = (PIPELINE_SETTINGS, INDEX_AND_MODEL_SETTINGS, ARGUMENT_SETTINGS, NON_RESULT_SETTINGS)
    names = [name for group in groups for name in group]
    assert len(names) == len(set(names))
    assert set(names) == set(Settings.model_fields)


def test_only_variable_names_are_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROUNDEDAI_API_KEY", "a-secret-value-of-24-chars")
    monkeypatch.setenv("GROUNDEDAI_CANDIDATE_K", "7")
    monkeypatch.setenv("OTHER_VARIABLE", "x")
    names = environment_variable_names()
    assert names == ["GROUNDEDAI_API_KEY", "GROUNDEDAI_CANDIDATE_K"]
    assert "a-secret-value-of-24-chars" not in json.dumps(names)


def test_settings_from_refuses_a_report_that_did_not_record_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"kind": "generation", "run_config": {"chunk_size": 400}}), "utf-8")
    assert main(["eval", "generation", "--settings-from", str(old)]) == 2
    assert "does not record its settings" in capsys.readouterr().err


def test_settings_from_names_settings_a_report_lacks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    partial = tmp_path / "partial.json"
    config = {
        "settings": {name: None for name in INDEX_AND_MODEL_SETTINGS if name != "chunk_size"},
        "pipeline_settings": {name: None for name in PIPELINE_SETTINGS if name != "rrf_k"},
    }
    partial.write_text(json.dumps({"run_config": config}), "utf-8")
    assert main(["eval", "generation", "--settings-from", str(partial)]) == 2
    assert "does not record these settings: chunk_size, rrf_k" in capsys.readouterr().err
