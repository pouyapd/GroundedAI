"""The Colab fine-tuning notebook and its setup and recovery logic (no PyTorch needed).

The notebook must run with *Run all* after any kernel restart: step 1 is self-contained, every
later cell imports the pipeline again, no cell uses a variable from another cell, and nothing
is passed through shell interpolation. The pipeline must keep a run's settings, resume only from
complete checkpoints, never redo finished work, fall back to local storage when Google Drive
cannot be used, and remove only a torchao that PEFT would refuse.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[3]
TRAINING = REPO / "training"
NOTEBOOK = REPO / "notebooks" / "finetuning" / "GroundedAI_LoRA_Colab.ipynb"
sys.path.insert(0, str(TRAINING))

pipeline = importlib.import_module("colab_pipeline")
finetune = importlib.import_module("finetune_lora")
build_notebook = importlib.import_module("build_notebook")


def code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


# ------------------------------------------------------------------------------ notebook
def test_the_notebook_is_generated_from_build_notebook() -> None:
    assert NOTEBOOK.read_text(encoding="utf-8") == build_notebook.render()


def test_no_shell_lines_or_unfilled_placeholders() -> None:
    """The old notebook ran `!git clone {REPO} ...`: after a restart {REPO} stayed literal."""
    for source in code_cells():
        for line in source.splitlines():
            assert not line.lstrip().startswith(("!", "%")), line
        tree = ast.parse(source)
        in_fstrings = {
            id(v) for node in ast.walk(tree) if isinstance(node, ast.JoinedStr) for v in node.values
        }
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in in_fstrings
            ):
                assert "{REPO" not in node.value
                assert "{OUT" not in node.value


def test_every_cell_after_setup_starts_with_the_preamble() -> None:
    cells = code_cells()
    assert "git" in cells[0]
    assert "clone" in cells[0]
    for source in cells[1:]:
        assert source.startswith(build_notebook.PREAMBLE)
        assert "pipeline." in source


def _undefined_names(source: str) -> set[str]:
    tree = ast.parse(source)
    defined: set[str] = set(dir(builtins))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            defined |= {(alias.asname or alias.name).split(".")[0] for alias in node.names}
        elif isinstance(node, ast.FunctionDef):
            defined.add(node.name)
            defined |= {arg.arg for arg in node.args.args + node.args.kwonlyargs}
            if node.args.vararg:
                defined.add(node.args.vararg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            defined.add(node.id)
        elif isinstance(node, ast.comprehension):
            defined |= {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - defined


def test_no_cell_uses_a_variable_from_another_cell() -> None:
    for source in code_cells():
        assert _undefined_names(source) == set(), source[:200]


def test_the_setup_cell_settings_are_the_experiment() -> None:
    setup = code_cells()[0]
    namespace: dict[str, Any] = {}
    tree = ast.parse(setup)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "SETTINGS" for t in node.targets
        ):
            namespace["SETTINGS"] = ast.literal_eval(node.value)
    settings = namespace["SETTINGS"]
    assert settings == pipeline.DEFAULTS
    assert settings["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert settings["use_drive"] is False
    assert (settings["lora_rank"], settings["lora_alpha"], settings["lora_dropout"]) == (
        16,
        32,
        0.05,
    )
    assert (
        settings["accumulate"],
        settings["epochs"],
        settings["learning_rate"],
        settings["seed"],
    ) == (8, 2, 1e-4, 13)
    assert 'REPO_URL = "https://github.com/pouyapd/GroundedAI.git"' in setup
    assert "subprocess.run" in setup


def test_every_file_the_notebook_needs_is_in_the_repository() -> None:
    assert pipeline.missing_files(REPO) == []
    setup = code_cells()[0]
    needed = ast.literal_eval(setup.split("NEEDED = ", 1)[1].split("\n\n", 1)[0])
    assert all((REPO / name).is_file() for name in needed)


def test_the_training_data_matches_its_hashes() -> None:
    report = json.loads((TRAINING / "data" / "dataset_report.json").read_text(encoding="utf-8"))
    for name, digest in report["sha256"].items():
        assert pipeline.sha256(TRAINING / "data" / name) == digest


# --------------------------------------------------------------------------- settings
@pytest.fixture
def colab_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GROUNDEDAI_COLAB_ROOT", str(tmp_path))
    monkeypatch.delenv("GROUNDEDAI_NOTEBOOK_TEST", raising=False)
    (tmp_path / "GroundedAI").mkdir()
    return tmp_path


def test_configure_writes_settings_that_later_cells_read(colab_root: Path) -> None:
    settings = pipeline.configure({"run_name": "run1"})
    assert settings["output_root"] == str(colab_root / "groundedai-lora")
    assert (colab_root / "groundedai-lora" / "run1").is_dir()
    assert pipeline.load_settings() == settings
    assert settings["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"


def test_later_cells_stop_clearly_without_step_1(colab_root: Path) -> None:
    with pytest.raises(SystemExit, match="run step 1"):
        pipeline.load_settings()


def test_unknown_settings_are_refused(colab_root: Path) -> None:
    with pytest.raises(SystemExit, match="Unknown settings"):
        pipeline.configure({"bas_model": "x"})


def test_drive_that_cannot_be_mounted_falls_back_to_local_storage(
    colab_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = pipeline.configure({"use_drive": True})  # google.colab is not importable here
    assert settings["use_drive"] is False
    assert settings["output_root"] == str(colab_root / "groundedai-lora")
    assert "local storage" in capsys.readouterr().out


def test_a_run_keeps_its_training_settings(colab_root: Path) -> None:
    pipeline.configure({"run_name": "run1"})
    pipeline.configure({"run_name": "run1"})  # the same again: fine
    with pytest.raises(SystemExit, match="new run_name"):
        pipeline.configure({"run_name": "run1", "learning_rate": 2e-4})
    pipeline.configure({"run_name": "run2", "learning_rate": 2e-4})


def test_training_arguments_carry_the_settings(colab_root: Path) -> None:
    settings = pipeline.configure({})
    args = pipeline.training_args(settings, Path("out"), smoke=False)
    assert args[args.index("--base") + 1] == "Qwen/Qwen2.5-1.5B-Instruct"
    for flag, value in (
        ("--rank", "16"),
        ("--alpha", "32"),
        ("--dropout", "0.05"),
        ("--accumulate", "8"),
        ("--seed", "13"),
        ("--epochs", "2"),
    ):
        assert args[args.index(flag) + 1] == value
    assert "--max-steps" not in args
    assert "--smoke" in pipeline.training_args(settings, Path("out"), smoke=True)


# ------------------------------------------------------------------------ environment
@pytest.mark.parametrize(
    ("installed", "action"),
    [
        (None, "absent"),
        ("0.10.0", "remove"),
        ("0.15.9", "remove"),
        ("0.16.0", "keep"),
        ("0.17.1+cu130", "keep"),
    ],
)
def test_only_a_torchao_that_peft_refuses_is_removed(installed: str | None, action: str) -> None:
    assert pipeline.torchao_action(installed) == action


# --------------------------------------------------------------------------- recovery
def _checkpoint(path: Path, complete: bool = True) -> Path:
    path.mkdir(parents=True)
    names = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors"]
    for name in names if complete else names[:2]:
        (path / name).write_text("{}" if name.endswith(".json") else "x", encoding="utf-8")
    return path


def test_resume_uses_the_newest_complete_checkpoint(tmp_path: Path) -> None:
    checkpoints = tmp_path / "checkpoints"
    _checkpoint(checkpoints / "checkpoint-20")
    _checkpoint(checkpoints / "checkpoint-40")
    _checkpoint(checkpoints / "checkpoint-60", complete=False)  # cut off while saving
    found = finetune.latest_valid_checkpoint(checkpoints)
    assert found is not None
    assert found.name == "checkpoint-40"
    names = sorted(p.name for p in checkpoints.iterdir())
    assert "checkpoint-60" not in names
    assert any(name.startswith("incomplete-checkpoint-60") for name in names)


def test_a_damaged_trainer_state_is_not_resumed(tmp_path: Path) -> None:
    broken = _checkpoint(tmp_path / "checkpoints" / "checkpoint-5")
    (broken / "trainer_state.json").write_text("{not json", encoding="utf-8")
    assert finetune.latest_valid_checkpoint(tmp_path / "checkpoints") is None
    assert finetune.latest_valid_checkpoint(tmp_path / "missing") is None


def test_a_finished_run_is_recognised(tmp_path: Path) -> None:
    assert not finetune.run_complete(tmp_path)
    (tmp_path / "adapter").mkdir()
    (tmp_path / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert not finetune.run_complete(tmp_path)  # no result.json: training did not finish
    (tmp_path / "result.json").write_text(json.dumps({"finished": True}), encoding="utf-8")
    assert finetune.run_complete(tmp_path)
    assert pipeline.training_complete(tmp_path)


def test_the_smoke_test_must_have_passed_for_the_same_model(colab_root: Path) -> None:
    settings = pipeline.configure({})
    assert not pipeline.smoke_passed(settings)
    smoke = pipeline.run_dir(settings) / "smoke"
    smoke.mkdir()
    (smoke / "smoke_result.json").write_text(
        json.dumps({"passed": True, "base_model": "Qwen/Qwen2.5-0.5B-Instruct"}), encoding="utf-8"
    )
    assert not pipeline.smoke_passed(settings)
    (smoke / "smoke_result.json").write_text(
        json.dumps({"passed": True, "base_model": settings["base_model"]}), encoding="utf-8"
    )
    assert pipeline.smoke_passed(settings)
    (smoke / "smoke_result.json").write_text(
        json.dumps({"passed": False, "base_model": settings["base_model"]}), encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="smoke test"):
        pipeline.train()  # a failed smoke test blocks training


def test_finished_or_partial_reports(tmp_path: Path) -> None:
    report = tmp_path / "test-base.json"
    assert not pipeline.report_ready(report)
    report.write_text('{"per_question": [', encoding="utf-8")  # cut off
    assert not pipeline.report_ready(report)
    report.write_text(json.dumps({"per_question": []}), encoding="utf-8")
    assert pipeline.report_ready(report)


def test_the_archive_has_results_but_no_checkpoints(tmp_path: Path) -> None:
    run = tmp_path / "run1"
    for name in (
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "reports/test-base.json",
        "result.json",
        "checkpoints/checkpoint-20/optimizer.pt",
        "smoke/checkpoints/checkpoint-3/optimizer.pt",
        "smoke/smoke_result.json",
    ):
        (run / name).parent.mkdir(parents=True, exist_ok=True)
        (run / name).write_text("x", encoding="utf-8")
    target = tmp_path / "results.zip"
    names = pipeline.make_archive(run, target)
    assert target.is_file()
    assert not (tmp_path / "results.zip.partial").exists()
    assert "adapter/adapter_model.safetensors" in names
    assert "reports/test-base.json" in names
    assert "smoke/smoke_result.json" in names
    assert not any("checkpoints" in name for name in names)


def test_estimated_steps() -> None:
    assert pipeline.estimated_steps(713, 8, 2) == 180


def test_a_comparison_with_no_question_of_a_kind_does_not_fail() -> None:
    """A limited evaluation can have no question that should be refused (found in a test run)."""
    sys.path.insert(0, str(REPO / "scripts"))
    compare_rag_reports = importlib.import_module("compare_rag_reports")
    assert compare_rag_reports._paired([], []) == {
        "n": 0,
        "old_mean": None,
        "new_mean": None,
        "gained": 0,
        "lost": 0,
    }
