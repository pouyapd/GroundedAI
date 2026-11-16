"""The steps of notebooks/finetuning/GroundedAI_LoRA_Colab.ipynb, restart-safe.

The notebook keeps no state in Python variables between cells. Step 1 (self-contained in the
notebook) clones the repository and calls :func:`configure`, which writes the settings to
``<root>/groundedai-lora/settings.json``. Every later cell imports this module again and each
function reads the settings from that file, so a kernel restart loses nothing that matters:
re-running a cell (or *Run all*) rebuilds what it needs. Heavy work (the environment check,
training, evaluation) runs in separate Python processes, so removing a package never needs a
kernel restart.

Storage: by default everything is written under ``<root>/groundedai-lora/<run name>/`` on the
runtime's own disk. It survives a kernel restart, **not** the end of the runtime (Colab deletes
the disk). With ``use_drive`` the run folder is in Google Drive instead; if Drive cannot be
mounted, the notebook says so and uses the local folder.

``GROUNDEDAI_COLAB_ROOT`` changes ``/content`` (for tests). ``GROUNDEDAI_NOTEBOOK_TEST=1`` is a
test mode for running the notebook on a CPU: a small model, few steps and few questions. It is
never on in Colab unless set explicitly.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

REPO_URL = "https://github.com/pouyapd/GroundedAI.git"
DEFAULTS: dict[str, Any] = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "run_name": "run1",
    "use_drive": False,
    "epochs": 2,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "accumulate": 8,
    "seed": 13,
}
REQUIRED_FILES = (
    "training/colab_pipeline.py",
    "training/finetune_lora.py",
    "training/evaluate_finetuned.py",
    "training/check_environment.py",
    "training/requirements-colab.txt",
    "training/data/train.jsonl",
    "training/data/validation.jsonl",
    "training/data/dataset_report.json",
    "training/data/synthetic-eval/manifest.json",
    "backend/pyproject.toml",
    "scripts/fetch_nist_corpus.py",
    "scripts/compare_rag_reports.py",
    "evaluation/datasets/groundedai-nist-val-v1/manifest.json",
    "evaluation/datasets/groundedai-nist-val-relevance-v1/manifest.json",
    "evaluation/datasets/groundedai-nist-v1/manifest.json",
)
TORCHAO_MINIMUM = (0, 16, 0)  # PEFT 0.21 refuses older torchao versions when torchao is installed
VALIDATION_SETS = {
    "synthetic-eval": "training/data/synthetic-eval",
    "nist-val": "evaluation/datasets/groundedai-nist-val-v1",
    "nist-val-relevance": "evaluation/datasets/groundedai-nist-val-relevance-v1",
}
TEST_SET = "evaluation/datasets/groundedai-nist-v1"


# ------------------------------------------------------------------------------- paths
def root() -> Path:
    return Path(os.environ.get("GROUNDEDAI_COLAB_ROOT", "/content"))


def repo_dir() -> Path:
    return root() / "GroundedAI"


def local_output_root() -> Path:
    return root() / "groundedai-lora"


def settings_file() -> Path:
    return local_output_root() / "settings.json"


def archive_path() -> Path:
    return root() / "groundedai-lora-results.zip"


def test_mode() -> bool:
    return os.environ.get("GROUNDEDAI_NOTEBOOK_TEST") == "1"


def run_dir(settings: dict[str, Any]) -> Path:
    return Path(settings["output_root"]) / settings["run_name"]


# ---------------------------------------------------------------------------- helpers
def say(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


def run(cmd: list[str], *, cwd: Path | None = None, quiet: bool = False) -> None:
    """Run a command, showing its output in the notebook as it comes (through Python, so it
    appears in Colab); stop the notebook with the end of the output when it fails."""
    print("$ " + " ".join(cmd), flush=True)
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(  # noqa: S603 - fixed argument lists built in this module
        cmd,
        cwd=cwd or repo_dir(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        env=env,
    )
    tail: list[str] = []
    assert process.stdout is not None  # noqa: S101 - set by stdout=PIPE
    for chunk in iter(lambda: process.stdout.read(512), ""):  # type: ignore[union-attr]
        tail = [*tail, chunk][-16:]
        if not quiet:
            sys.stdout.write(chunk)
            sys.stdout.flush()
    code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"Command failed with exit code {code}: {' '.join(cmd)}\n--- end of its output ---\n"
            + "".join(tail)[-4000:]
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 - git from PATH, as in the notebook
        cwd=repo_dir(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def missing_files(base: Path) -> list[str]:
    return [name for name in REQUIRED_FILES if not (base / name).is_file()]


def verify_repository() -> str:
    """Check the clone has every file the notebook needs; return the commit."""
    missing = missing_files(repo_dir())
    if missing:
        raise SystemExit(f"The repository at {repo_dir()} is incomplete; missing: {missing}")
    commit = git_commit()
    print(f"GroundedAI repository: {repo_dir()} at commit {commit}")
    return commit


# --------------------------------------------------------------------------- settings
def mount_drive() -> Path | None:
    """Mount Google Drive; None (with the reason printed) when that is not possible."""
    try:
        from google.colab import drive  # type: ignore[import-not-found]

        drive.mount("/content/drive")
        target = Path("/content/drive/MyDrive/groundedai-lora")
        target.mkdir(parents=True, exist_ok=True)
        return target
    except Exception as exc:
        print(f"Google Drive could not be used ({type(exc).__name__}: {exc}).")
        print("Continuing with local storage; it is lost when the Colab runtime ends.")
        return None


def configure(user: dict[str, Any] | None = None) -> dict[str, Any]:
    """Merge the user's settings with the defaults, choose the storage, save, and print."""
    settings = {**DEFAULTS, **(user or {})}
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise SystemExit(f"Unknown settings: {sorted(unknown)}")
    if test_mode():
        settings["base_model"] = os.environ.get(
            "GROUNDEDAI_TEST_BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"
        )
        print("TEST MODE (GROUNDEDAI_NOTEBOOK_TEST=1): small model, few steps, few questions.")
    output_root = local_output_root()
    if settings["use_drive"]:
        drive_root = mount_drive()
        if drive_root is not None:
            output_root = drive_root
        else:
            settings["use_drive"] = False
    settings["output_root"] = str(output_root)
    settings["test_mode"] = test_mode()
    local_output_root().mkdir(parents=True, exist_ok=True)
    run_dir(settings).mkdir(parents=True, exist_ok=True)
    check_run_settings(settings)
    settings_file().write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    note_commit(settings)
    print(json.dumps(settings, indent=2))
    where = (
        "Google Drive"
        if settings["use_drive"]
        else "this runtime's disk (lost when the runtime ends)"
    )
    print(f"Results go to {run_dir(settings)} on {where}.")
    return settings


RUN_KEYS = (
    "base_model",
    "epochs",
    "learning_rate",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "accumulate",
    "seed",
    "test_mode",
)


def check_run_settings(settings: dict[str, Any]) -> None:
    """A run folder keeps the training settings it started with; other settings need a new
    run name (so a resumed run never mixes two configurations)."""
    path = run_dir(settings) / "run_settings.json"
    wanted = {key: settings[key] for key in RUN_KEYS}
    if path.is_file():
        first = json.loads(path.read_text(encoding="utf-8"))
        changed = {k: (first.get(k), wanted[k]) for k in RUN_KEYS if first.get(k) != wanted[k]}
        if changed:
            raise SystemExit(
                f"Run {settings['run_name']!r} was started with other settings {changed}; "
                "keep them, or choose a new run_name in step 1."
            )
    else:
        path.write_text(json.dumps(wanted, indent=2) + "\n", encoding="utf-8")


def require_gpu() -> None:
    """Stop before installing anything when the runtime has no GPU (except in test mode)."""
    import torch

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif not test_mode():
        raise SystemExit(
            "No GPU in this runtime: Runtime -> Change runtime type -> T4 GPU, then Run all. "
            "Nothing was installed or trained."
        )
    else:
        print("TEST MODE: no GPU; continuing on the CPU.")


def load_settings() -> dict[str, Any]:
    path = settings_file()
    if not path.is_file():
        raise SystemExit("No settings in this runtime: run step 1 first (or Runtime -> Run all).")
    settings = json.loads(path.read_text(encoding="utf-8"))
    os.chdir(repo_dir())
    return settings


def note_commit(settings: dict[str, Any]) -> None:
    """Record which commit each session used; warn when a run continues with other code."""
    path = run_dir(settings) / "groundedai_commits.json"
    commits = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    commit = git_commit() if (repo_dir() / ".git").exists() else "unknown"
    if commits and commits[-1]["commit"] != commit:
        print(
            f"Note: this run started with commit {commits[0]['commit']}; "
            f"it continues with {commit}."
        )
    commits.append({"commit": commit, "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    path.write_text(json.dumps(commits, indent=2) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------ environment
def version_tuple(text: str) -> tuple[int, ...]:
    parts = []
    for piece in text.split("+")[0].split("."):
        digits = "".join(ch for ch in piece if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def torchao_action(installed: str | None) -> str:
    """'absent', 'keep' (new enough for PEFT) or 'remove' (PEFT would refuse it)."""
    if installed is None:
        return "absent"
    return "keep" if version_tuple(installed) >= TORCHAO_MINIMUM else "remove"


def installed_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def prepare_environment() -> dict[str, Any]:
    """Install the pinned libraries without touching PyTorch, remove an incompatible torchao,
    install GroundedAI's backend, then check the result in a fresh Python process."""
    settings = load_settings()
    say("Libraries")
    torch_before = installed_version("torch")
    if torch_before is None:
        raise SystemExit("PyTorch is not installed in this runtime.")
    constraints = local_output_root() / "constraints.txt"
    constraints.write_text(f"torch=={torch_before}\n", encoding="utf-8")  # never replace PyTorch
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "-r",
            "training/requirements-colab.txt",
            "-c",
            str(constraints),
        ]
    )
    action = torchao_action(installed_version("torchao"))
    print(f"torchao: {installed_version('torchao') or 'not installed'} -> {action}")
    if action == "remove":
        run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchao"])
        if installed_version("torchao") is not None:
            raise SystemExit(
                "torchao could not be removed; PEFT would refuse to create the LoRA layers."
            )
        print("torchao removed (GroundedAI uses plain fp16 LoRA, not torchao).")
    run(
        [sys.executable, "-m", "pip", "install", "--quiet", "-e", "backend", "-c", str(constraints)]
    )
    if installed_version("torch") != torch_before:
        raise SystemExit(
            f"PyTorch changed from {torch_before} to {installed_version('torch')}; stopping."
        )
    say("Compatibility check (fresh Python process)")
    out = run_dir(settings) / "environment_check.json"
    run([sys.executable, "training/check_environment.py", "--out", str(out)])
    report = json.loads(out.read_text(encoding="utf-8"))
    if not report.get("ok"):
        raise SystemExit(f"The environment check failed: {report.get('error')}")
    return report


def hardware(require_gpu: bool = True) -> dict[str, Any]:
    settings = load_settings()
    report = json.loads((run_dir(settings) / "environment_check.json").read_text(encoding="utf-8"))
    say("Hardware")
    for key in (
        "gpu",
        "gpu_memory_gb",
        "compute_capability",
        "cuda_version",
        "torch",
        "python",
        "ram_gb",
        "disk_free_gb",
    ):
        print(f"{key:20s} {report.get(key)}")
    if require_gpu and not report.get("cuda_available") and not settings.get("test_mode"):
        raise SystemExit(
            "No GPU in this runtime: Runtime -> Change runtime type -> T4 GPU, then Run all. "
            "Nothing was trained."
        )
    memory = report.get("gpu_memory_gb") or 0
    if report.get("cuda_available") and memory < 10:
        print(
            f"Warning: {memory} GB of GPU memory is little for {settings['base_model']}; "
            "the length check in training decides."
        )
    return report


def verify_data() -> dict[str, Any]:
    load_settings()
    say("Training data")
    report = json.loads(Path("training/data/dataset_report.json").read_text(encoding="utf-8"))
    for name, digest in report["sha256"].items():
        actual = sha256(Path("training/data") / name)
        if actual != digest:
            raise SystemExit(
                f"training/data/{name} does not match dataset_report.json ({actual} != {digest})"
            )
    manifest = json.loads(
        Path("training/data/synthetic-eval/manifest.json").read_text(encoding="utf-8")
    )
    for document in manifest["documents"]:
        if sha256(Path("training/data/synthetic-eval") / document["file"]) != document["sha256"]:
            raise SystemExit(f"synthetic-eval document {document['file']} does not match its hash")
    print(f"verified: {report['totals']} examples, {len(manifest['documents'])} held-out documents")
    return report


# -------------------------------------------------------------------------- training
def training_args(settings: dict[str, Any], out: Path, smoke: bool) -> list[str]:
    args = [
        sys.executable,
        "training/finetune_lora.py",
        "--base",
        settings["base_model"],
        "--out",
        str(out),
        "--epochs",
        str(settings["epochs"]),
        "--lr",
        str(settings["learning_rate"]),
        "--rank",
        str(settings["lora_rank"]),
        "--alpha",
        str(settings["lora_alpha"]),
        "--dropout",
        str(settings["lora_dropout"]),
        "--accumulate",
        str(settings["accumulate"]),
        "--seed",
        str(settings["seed"]),
    ]
    if smoke:
        args.append("--smoke")
    if settings.get("test_mode"):
        args += ["--accumulate", "2", "--validation-limit", "8"]
        if not smoke:
            args += ["--max-steps", "2", "--eval-steps", "1", "--save-steps", "1"]
    return args


def smoke_test() -> dict[str, Any]:
    settings = load_settings()
    out = run_dir(settings) / "smoke"
    say(f"Smoke test with {settings['base_model']}")
    if out.exists():
        shutil.rmtree(out)  # always from scratch
    try:
        run(training_args(settings, out, smoke=True))
    finally:
        result_file = out / "smoke_result.json"
    if not result_file.is_file():
        raise SystemExit("The smoke test stopped before writing its result; see the output above.")
    result = json.loads(result_file.read_text(encoding="utf-8"))
    if not result.get("passed"):
        raise SystemExit(
            "The smoke test FAILED; full training will not start. See the stage that failed above."
        )
    print("Smoke test PASSED: training may start.")
    return result


def smoke_passed(settings: dict[str, Any]) -> bool:
    path = run_dir(settings) / "smoke" / "smoke_result.json"
    if not path.is_file():
        return False
    result = json.loads(path.read_text(encoding="utf-8"))
    return bool(result.get("passed")) and result.get("base_model") == settings["base_model"]


def training_complete(out: Path) -> bool:
    result = out / "result.json"
    if not result.is_file() or not (out / "adapter" / "adapter_config.json").is_file():
        return False
    try:
        return bool(json.loads(result.read_text(encoding="utf-8")).get("finished"))
    except json.JSONDecodeError:
        return False


def train() -> dict[str, Any]:
    settings = load_settings()
    out = run_dir(settings)
    if not smoke_passed(settings):
        raise SystemExit(
            "The smoke test has not passed in this run folder; run step 4 first. "
            "Training was not started."
        )
    if training_complete(out):
        print(
            f"Training of {settings['run_name']} is already complete ({out}); it is not repeated."
        )
        print("Use a new run_name in step 1 to train again.")
    else:
        say(f"Training {settings['base_model']} with LoRA")
        run(training_args(settings, out, smoke=False))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    print(json.dumps(result, indent=2))
    return result


# ------------------------------------------------------------------------- evaluation
def report_ready(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        return "per_question" in json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False


def evaluate(
    name: str,
    dataset: str,
    label: str,
    settings: dict[str, Any],
    merge: bool = False,
    limit: int | None = None,
) -> Path:
    reports = run_dir(settings) / "reports"
    reports.mkdir(exist_ok=True)
    target = reports / f"{name}-{label}.json"
    if report_ready(target):
        print(f"already done: {target.name}")
        return target
    cmd = [
        sys.executable,
        "training/evaluate_finetuned.py",
        "--base",
        settings["base_model"],
        "--dataset",
        dataset,
        "--label",
        label,
        "--out",
        str(target),
    ]
    if label != "base":
        cmd += ["--adapter", str(run_dir(settings) / "adapter")]
    if merge:
        cmd.append("--merge")
    if settings.get("test_mode"):
        cmd += ["--limit", "2"]
    elif limit:
        cmd += ["--limit", str(limit)]
    run(cmd)
    return target


def compare(name: str, settings: dict[str, Any]) -> dict[str, Any]:
    reports = run_dir(settings) / "reports"
    out = reports / f"{name}-comparison.json"
    run(
        [
            sys.executable,
            "scripts/compare_rag_reports.py",
            str(reports / f"{name}-base.json"),
            str(reports / f"{name}-lora.json"),
            "--out",
            str(out),
        ],
        quiet=True,
    )
    measures = json.loads(out.read_text(encoding="utf-8"))["measures"]
    for measure, values in measures.items():
        print(
            f"  {measure:32s} base {values['old_mean']}  lora {values['new_mean']}  "
            f"diff {values.get('mean_diff')} [{values.get('ci95_low')}, {values.get('ci95_high')}]"
        )
    return measures


def require_training(settings: dict[str, Any]) -> None:
    if not training_complete(run_dir(settings)):
        raise SystemExit("Training is not complete in this run folder; run step 5 first.")


def fetch_corpus() -> None:
    run([sys.executable, "scripts/fetch_nist_corpus.py"], quiet=True)
    print("NIST PDFs downloaded from nist.gov and hash-checked")


def evaluate_validation() -> None:
    settings = load_settings()
    require_training(settings)
    fetch_corpus()
    for name, dataset in VALIDATION_SETS.items():
        say(f"Validation: {name}")
        evaluate(name, dataset, "base", settings)
        evaluate(name, dataset, "lora", settings)
        compare(name, settings)


def evaluate_test() -> None:
    settings = load_settings()
    require_training(settings)
    fetch_corpus()
    say("Protected test set (run once)")
    evaluate("test", TEST_SET, "base", settings)
    evaluate("test", TEST_SET, "lora", settings)
    compare("test", settings)


def merge_check() -> dict[str, Any]:
    settings = load_settings()
    require_training(settings)
    say("Merged model vs adapter")
    merged = evaluate(
        "merge-check",
        VALIDATION_SETS["synthetic-eval"],
        "lora-merged",
        settings,
        merge=True,
        limit=20,
    )
    adapter_report = run_dir(settings) / "reports" / "synthetic-eval-lora.json"
    a = {
        r["question_id"]: r.get("model_output")
        for r in json.loads(adapter_report.read_text(encoding="utf-8"))["per_question"]
    }
    b = {
        r["question_id"]: r.get("model_output")
        for r in json.loads(merged.read_text(encoding="utf-8"))["per_question"]
    }
    same = sum(a.get(q) == b[q] for q in b)
    result = {"same_answer_as_adapter": same, "questions": len(b)}
    (run_dir(settings) / "reports" / "merge-check.json").write_text(
        json.dumps(result) + "\n", encoding="utf-8"
    )
    print(f"merged model: same answer as the adapter on {same} of {len(b)} questions")
    return result


# --------------------------------------------------------------------------- archive
ARCHIVE_SKIP = ("checkpoints", "smoke/checkpoints", "smoke/adapter", "smoke/tokenizer")


def make_archive(run: Path, target: Path) -> list[str]:
    """Zip the run folder without checkpoints (adapter, tokenizer, logs, reports)."""
    names = []
    tmp = target.with_suffix(".zip.partial")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(run.rglob("*")):
            relative = path.relative_to(run).as_posix()
            if path.is_dir() or any(
                relative == s or relative.startswith(s + "/") for s in ARCHIVE_SKIP
            ):
                continue
            archive.write(path, f"{run.name}/{relative}")
            names.append(relative)
    os.replace(tmp, target)
    return names


def archive() -> Path:
    settings = load_settings()
    say("Results archive")
    target = archive_path()
    names = make_archive(run_dir(settings), target)
    size = target.stat().st_size / 2**20
    print(f"{target} ({size:.1f} MB, {len(names)} files; checkpoints are not included)")
    print(
        "Download it: open the Files panel on the left, find groundedai-lora-results.zip "
        "under /content, "
    )
    print("then the three-dot menu -> Download. It is deleted when the runtime ends.")
    try:
        from google.colab import files  # type: ignore[import-not-found]

        files.download(str(target))
    except Exception as exc:
        print(f"(automatic download not started: {type(exc).__name__})")
    return target


def estimated_steps(examples: int, accumulate: int, epochs: float) -> int:
    return math.ceil(examples / accumulate) * math.ceil(epochs)
