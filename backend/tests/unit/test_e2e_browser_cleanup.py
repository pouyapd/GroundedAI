"""scripts/e2e_browser.py removes its working folder (uploaded files, database) after a run.

On Windows the server's database can stay locked for a moment after the process is stopped
(a virtual environment's python.exe is a launcher whose child exits after it). A single
``rmtree(..., ignore_errors=True)`` then left the folder behind without a word.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from tests.conftest import REPO_ROOT


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "e2e_browser_cli", REPO_ROOT / "scripts" / "e2e_browser.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _workdir(tmp_path: Path) -> Path:
    work = tmp_path / "groundedai-e2e-test"
    (work / "data").mkdir(parents=True)
    (work / "data" / "groundedai.db").write_bytes(b"x")
    return work


def test_a_folder_that_is_locked_for_a_moment_is_still_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = _script()
    work = _workdir(tmp_path)
    real_rmtree = shutil.rmtree
    calls = {"n": 0}

    def locked_twice(path: Any, *args: Any, **kwargs: Any) -> None:
        calls["n"] += 1
        if calls["n"] <= 2:
            raise PermissionError("the database is still open")
        real_rmtree(path)

    monkeypatch.setattr(script.shutil, "rmtree", locked_twice)
    monkeypatch.setattr(script.time, "sleep", lambda _seconds: None)
    assert script.remove_workdir(work) is True
    assert not work.exists()


def test_a_folder_that_cannot_be_removed_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    script = _script()
    work = _workdir(tmp_path)

    def always_locked(path: Any, *args: Any, **kwargs: Any) -> None:
        raise PermissionError("the database is still open")

    monkeypatch.setattr(script.shutil, "rmtree", always_locked)
    monkeypatch.setattr(script.time, "sleep", lambda _seconds: None)
    assert script.remove_workdir(work) is False
    assert str(work) in capsys.readouterr().out
