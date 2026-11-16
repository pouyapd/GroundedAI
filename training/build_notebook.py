# ruff: noqa: E501 - the long lines are notebook cell text, kept readable as it appears in Colab
"""Write notebooks/finetuning/GroundedAI_LoRA_Colab.ipynb from this file.

    python training/build_notebook.py          # write
    python training/build_notebook.py --check  # fail if the notebook differs (used by the tests)

The notebook is generated so that its structure can be tested: step 1 is self-contained (it
clones the repository with subprocess and explicit arguments), and every later code cell starts
with the same preamble and calls one function of training/colab_pipeline.py. No cell uses a
variable defined in another cell, and no cell uses shell (``!``) or magic (``%``) lines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks" / "finetuning" / "GroundedAI_LoRA_Colab.ipynb"

PREAMBLE = """import importlib, os, sys
TRAINING = os.path.join(os.environ.get("GROUNDEDAI_COLAB_ROOT", "/content"), "GroundedAI", "training")
if not os.path.isfile(os.path.join(TRAINING, "colab_pipeline.py")):
    raise SystemExit("The code is not in this runtime (it was reset): run step 1, or Runtime -> Run all.")
if TRAINING not in sys.path:
    sys.path.insert(0, TRAINING)
import colab_pipeline as pipeline
importlib.reload(pipeline)
"""

SETUP = """# Step 1: settings and code. Everything later cells need is created here; running it again is safe.
import importlib, os, shutil, subprocess, sys, time

SETTINGS = {
    "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
    "run_name": "run1",          # keep it to continue a run; change it to start a new one
    "use_drive": False,          # True: keep results in Google Drive (asks for permission)
    "epochs": 2,
    "learning_rate": 1e-4,
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "accumulate": 8,
    "seed": 13,
}
REPO_URL = "https://github.com/pouyapd/GroundedAI.git"
REPO_BRANCH = os.environ.get("GROUNDEDAI_NOTEBOOK_BRANCH", "main")  # tests may set another branch
ROOT = os.environ.get("GROUNDEDAI_COLAB_ROOT", "/content")
REPO_DIR = os.path.join(ROOT, "GroundedAI")
NEEDED = ["training/colab_pipeline.py", "training/finetune_lora.py", "training/requirements-colab.txt",
          "training/data/dataset_report.json", "backend/pyproject.toml"]


def git(*args):
    result = subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed:\\n{result.stdout}{result.stderr}")
    return result.stdout.strip()


def same_origin(path):
    if not os.path.isdir(os.path.join(path, ".git")):
        return False
    result = subprocess.run(["git", "remote", "get-url", "origin"], cwd=path, capture_output=True, text=True)
    return result.returncode == 0 and result.stdout.strip().rstrip("/") == REPO_URL.rstrip("/")


if same_origin(REPO_DIR):
    print("Updating the existing clone (local changes in it are discarded) ...")
    git("fetch", "--quiet", "origin", REPO_BRANCH)
    git("checkout", "--quiet", "--force", "-B", REPO_BRANCH, "origin/" + REPO_BRANCH)
else:
    if os.path.exists(REPO_DIR):
        aside = f"{REPO_DIR}.unusable-{int(time.time())}"
        shutil.move(REPO_DIR, aside)
        print("Moved an unusable folder aside:", aside)
    print("Cloning", REPO_URL, "...")
    result = subprocess.run(["git", "clone", "--quiet", "--branch", REPO_BRANCH, REPO_URL, REPO_DIR],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"Cloning {REPO_URL} failed:\\n{result.stdout}{result.stderr}")
missing = [name for name in NEEDED if not os.path.isfile(os.path.join(REPO_DIR, name))]
if missing:
    raise SystemExit(f"The clone in {REPO_DIR} is missing {missing}; stopping.")
os.chdir(REPO_DIR)
print("Commit used:", git("rev-parse", "HEAD"))
TRAINING = os.path.join(REPO_DIR, "training")
if TRAINING not in sys.path:
    sys.path.insert(0, TRAINING)
import colab_pipeline as pipeline
importlib.reload(pipeline)
pipeline.verify_repository()
pipeline.configure(SETTINGS)
"""

STEPS = [
    (
        "## 2. Libraries and compatibility check\n\nInstalls the pinned training libraries without "
        "changing Colab's PyTorch, removes a torchao version that PEFT cannot use (GroundedAI does "
        "not use torchao), installs GroundedAI, then creates LoRA layers on a tiny model in a fresh "
        "process to prove that it works. No restart is needed.",
        "pipeline.require_gpu()\npipeline.prepare_environment()\npipeline.hardware(require_gpu=True)",
    ),
    (
        "## 3. Training data\n\nChecks the files against the hashes in `dataset_report.json`.",
        "pipeline.verify_data()",
    ),
    (
        "## 4. Smoke test\n\nEight stages, each PASS or FAIL: model loading, LoRA creation, a forward "
        "and backward pass, three optimizer steps, the validation loss, saving the adapter, loading "
        "it again, and generating with it. If a stage fails, the notebook stops here and training "
        "does not start.",
        "pipeline.smoke_test()",
    ),
    (
        "## 5. Training\n\nPrints the plan first (GPU, data, sequence length, steps, output folder, "
        "fresh or resumed). Checkpoints are saved every 20 steps. After a kernel restart, *Run all* "
        "continues from the last valid checkpoint; a finished run is not trained again.",
        "pipeline.train()",
    ),
    (
        "## 6. Validation: base vs fine-tuned\n\nSame retrieval, prompt, checks and greedy decoding "
        "for both models. Finished reports are not run again.",
        "pipeline.evaluate_validation()",
    ),
    (
        "## 7. Protected test set (once)\n\n"
        "`groundedai-nist-v1` was not used for training or for any "
        "choice. It runs once per run folder; finished reports are not run again.",
        "pipeline.evaluate_test()",
    ),
    (
        "## 8. Merge check and results archive\n\n"
        "The merged model must answer like the adapter. The "
        "archive `/content/groundedai-lora-results.zip` holds the adapter, tokenizer, logs and "
        "reports (not the checkpoints).",
        "pipeline.merge_check()\npipeline.archive()",
    ),
]

INTRO = """# GroundedAI: LoRA fine-tuning of the answer model on a free Colab GPU

Fine-tunes **Qwen/Qwen2.5-1.5B-Instruct** (the model of the GroundedAI demo) with LoRA and
compares it with the original model through GroundedAI's own evaluation pipeline.

**How to run it**
1. *Runtime → Change runtime type →* a free GPU (for example **T4 GPU**) → *Save*.
2. *Runtime → Run all.* Nothing has to be edited.
3. When the last cell has finished, download **`/content/groundedai-lora-results.zip`**: it
   starts automatically in most browsers; otherwise open the *Files* panel on the left, find the
   file under `/content`, and choose *Download* in its menu.

**Free tier only.** Nothing here needs Colab Pro or compute units. If Colab offers no GPU, try
again later; the notebook stops in step 2 when there is no GPU and does not work around limits.

**Storage.** By default everything is written to `/content/groundedai-lora/run1/` on the
runtime's disk. It survives a kernel restart, but **it is deleted when the runtime ends**
(disconnect, idle timeout, *Disconnect and delete runtime*). Download the archive at the end.
To keep checkpoints across runtimes, set `"use_drive": True` in step 1; if Drive cannot be
mounted, the notebook continues with local storage.

**If something stops:** after a kernel restart, *Run all* again; it re-creates everything,
continues training from the last valid checkpoint and skips finished evaluations.

Details: [docs/FINETUNING.md](https://github.com/pouyapd/GroundedAI/blob/main/docs/FINETUNING.md)."""

OUTRO = """## Not done here: the model in the browser demo

The demo runs Qwen2.5-1.5B through WebLLM as an MLC 4-bit build. Converting the merged
fine-tuned weights (MLC LLM `convert_weight` with `q4f16_1`, then `gen_config`) and hosting them
needs the MLC LLM package, a place to host the weights and a change to the demo. It has not
been done or tested; see docs/FINETUNING.md."""


def lines(text: str) -> list[str]:
    return text.splitlines(True)


def build() -> dict:
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": lines(INTRO)},
        {"cell_type": "markdown", "metadata": {}, "source": lines("## 1. Settings and code")},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": lines(SETUP),
        },
    ]
    for markdown, call in STEPS:
        cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(markdown)})
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": lines(PREAMBLE + call + "\n"),
            }
        )
    cells.append({"cell_type": "markdown", "metadata": {}, "source": lines(OUTRO)})
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4", "name": "GroundedAI_LoRA_Colab.ipynb"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }


def render() -> str:
    return json.dumps(build(), indent=1, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    text = render()
    if args.check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.is_file() else ""
        if current != text:
            print(f"{TARGET} is out of date: run python training/build_notebook.py")
            return 1
        print("notebook up to date")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
