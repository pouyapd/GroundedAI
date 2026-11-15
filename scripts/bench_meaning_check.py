"""Measure the online demo's meaning check (NLI) in a real browser.

    python scripts/bench_meaning_check.py --out report.json            # frontend/dist
    python scripts/bench_meaning_check.py --url https://.../GroundedAI/ --out report.json

What it measures, on the CPU (WebAssembly, one thread; the demo does not use WebGPU for it):

* the first load with an empty browser profile (download + start) and a second load from the
  browser cache, in a new browser with the same profile;
* the bytes the browser received for the meaning check (model files and the ONNX Runtime
  WebAssembly file), as reported by the browser;
* the memory of the browser processes before the check is switched on, after it is loaded and
  after the pairs are classified (sum over all Chromium processes, and the page's renderer);
* the time per premise/statement pair, in the worker;
* agreement with onnxruntime in Python on the same int8 model file and the same pairs (top
  label and the largest probability difference), so the browser path is checked against the
  path that was evaluated;
* one end-to-end question with the meaning check on: the per-statement line and the summary
  row must appear, and the answer time is recorded.

The pairs come from the hand-written claims of groundedai-versions-v1 (dev and validation)
and from the rule-made validation claims. The protected test sets are not used here.

The page exposes the model to this script only when it is opened with ``#nli-bench``.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from e2e_demo import _serve  # noqa: E402

MODEL_DIR = ROOT / "experiments" / "deploy" / "runs" / "xenova-nli-xsmall"
VERSIONS_CLAIMS = ROOT / "evaluation" / "datasets" / "groundedai-versions-v1" / "claims.jsonl"
VALIDATION_CLAIMS = ROOT / "experiments" / "verifier" / "data" / "validation.jsonl"
FERRY_QUESTION = "How much does a single adult ticket to Mirrow cost?"
LOAD_TIMEOUT_MS = 15 * 60 * 1000
LABELS = ("entailment", "neutral", "contradiction")


def load_pairs(validation_sample: int, seed: int) -> list[dict[str, str]]:
    """Premise/statement pairs: all versions-v1 claims and a fixed sample of validation claims."""
    pairs = []
    for line in VERSIONS_CLAIMS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        pairs.append(
            {"set": f"versions-{row['split']}", "premise": row["evidence"], "claim": row["claim"]}
        )
    rows = [json.loads(line) for line in VALIDATION_CLAIMS.read_text(encoding="utf-8").splitlines()]
    sample = random.Random(seed).sample(rows, min(validation_sample, len(rows)))  # noqa: S311
    for row in sample:
        pairs.append({"set": "rule-validation", "premise": row["evidence"], "claim": row["claim"]})
    return pairs


def python_probabilities(pairs: list[dict[str, str]]) -> tuple[list[dict[str, float]], float]:
    """The same int8 file in onnxruntime (one thread, no padding: one pair per run)."""
    sys.path.insert(0, str(ROOT / "experiments" / "verifier"))
    from verifier_lib import OnnxClassifier

    classifier = OnnxClassifier(
        str(MODEL_DIR / "onnx" / "model_quantized.onnx"), str(MODEL_DIR), max_length=512, threads=1
    )
    started = time.perf_counter()
    out = []
    for pair in pairs:
        # OnnxClassifier takes (claim, evidence) and feeds the evidence first, as the browser does.
        out.extend(classifier.probabilities([(pair["claim"], pair["premise"])], batch=1))
    return out, (time.perf_counter() - started) * 1000


def chromium_memory(profile: Path) -> dict[str, Any] | None:
    """Memory of the Chromium processes that use this profile (psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    marker = f"--user-data-dir={profile}"
    main = None
    for proc in psutil.process_iter(["cmdline"]):
        cmdline = proc.info.get("cmdline") or []
        if any(arg == marker for arg in cmdline) and not any(
            a.startswith("--type=") for a in cmdline
        ):
            main = proc
            break
    if main is None:
        return None
    procs = [main, *main.children(recursive=True)]
    total_rss = total_private = 0
    renderer_rss = renderer_private = 0
    for proc in procs:
        try:
            info = proc.memory_info()
            cmdline = proc.cmdline()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        private = getattr(info, "private", getattr(info, "uss", 0)) or 0
        total_rss += info.rss
        total_private += private
        if "--type=renderer" in cmdline and info.rss > renderer_rss:
            renderer_rss, renderer_private = info.rss, private
    mb = 2**20
    return {
        "processes": len(procs),
        "all_rss_mb": round(total_rss / mb, 1),
        "all_private_mb": round(total_private / mb, 1),
        "renderer_rss_mb": round(renderer_rss / mb, 1),
        "renderer_private_mb": round(renderer_private / mb, 1),
    }


class Downloads:
    """Bytes the browser received for the meaning check, from Playwright's request events."""

    def __init__(self) -> None:
        self.files: dict[str, int] = {}

    def watch(self, page: Any) -> None:
        def finished(request: Any) -> None:
            url = request.url
            if "huggingface.co" in url or "hf.co" in url or "ort-wasm" in url or "nliWorker" in url:
                try:
                    size = request.sizes()["responseBodySize"]
                except Exception:  # a size the browser could not report
                    size = -1
                key = url.split("?")[0]
                # one URL can be read twice (the runtime file: counted for progress, then by
                # ONNX Runtime from the HTTP cache); the browser reports each request's bytes
                self.files[key] = self.files.get(key, 0) + max(size, 0)

        page.on("requestfinished", finished)


def open_demo(playwright: Any, profile: Path, url: str, headed: bool) -> tuple[Any, Any]:
    context = playwright.chromium.launch_persistent_context(
        str(profile), headless=not headed, viewport={"width": 1280, "height": 900}
    )
    page = context.pages[0] if context.pages else context.new_page()
    page.goto(url + "#nli-bench")
    page.wait_for_selector("text=Ferry guide", timeout=5 * 60 * 1000)
    return context, page


def switch_on(page: Any) -> dict[str, Any]:
    started = time.perf_counter()
    box = page.get_by_label("Meaning check with an NLI model")
    box.click()
    if not box.is_checked():
        raise RuntimeError("The meaning-check box did not show the choice while loading.")
    page.wait_for_function(
        "window.__groundedaiNli && window.__groundedaiNli.loaded", timeout=LOAD_TIMEOUT_MS
    )
    page.wait_for_function(
        "!document.querySelector('.nli-progress')", timeout=60_000
    )  # the panel finished
    return {
        "wall_ms": round((time.perf_counter() - started) * 1000),
        "worker_load_ms": round(page.evaluate("window.__groundedaiNli.loadMs")),
        # bytes the worker read (network or cache), model files and runtime file
        "worker_bytes_read": page.evaluate("window.__groundedaiNli.downloadedBytes"),
    }


def classify(
    page: Any, pairs: list[dict[str, str]], batch: int
) -> tuple[list[dict[str, float]], list[float]]:
    probabilities: list[dict[str, float]] = []
    per_pair_ms: list[float] = []
    for start in range(0, len(pairs), batch):
        chunk = [[p["premise"], p["claim"]] for p in pairs[start : start + batch]]
        result = page.evaluate("(pairs) => window.__groundedaiNli.classify(pairs)", chunk)
        probabilities.extend(result["probabilities"])
        per_pair_ms.append(result["ms"] / len(chunk))
    return probabilities, per_pair_ms


def _top(probabilities: dict[str, float]) -> str:
    return max(LABELS, key=lambda k: probabilities.get(k, 0.0))


def agreement(browser: list[dict[str, float]], python: list[dict[str, float]]) -> dict[str, Any]:
    same = 0
    differences = []
    for b, p in zip(browser, python, strict=True):
        same += _top(b) == _top(p)
        differences.append(max(abs(b.get(k, 0.0) - p.get(k, 0.0)) for k in LABELS))
    ordered = sorted(differences)
    return {
        "pairs": len(browser),
        "same_top_label": same,
        "max_probability_difference": round(ordered[-1], 6),
        "median_probability_difference": round(statistics.median(ordered), 6),
        "pairs_differing_by_more_than_0.05": sum(d > 0.05 for d in ordered),
    }


def largest_differences(
    pairs: list[dict[str, str]],
    browser: list[dict[str, float]],
    python: list[dict[str, float]],
    keep: int,
) -> list[dict[str, Any]]:
    rows = []
    for pair, b, p in zip(pairs, browser, python, strict=True):
        gap = max(abs(b.get(k, 0.0) - p.get(k, 0.0)) for k in LABELS)
        rows.append(
            {
                "set": pair["set"],
                "claim": pair["claim"],
                "premise": pair["premise"],
                "browser": {k: round(b.get(k, 0.0), 4) for k in LABELS},
                "python": {k: round(p.get(k, 0.0), 4) for k in LABELS},
                "same_top_label": _top(b) == _top(p),
                "difference": round(gap, 4),
            }
        )
    rows.sort(key=lambda row: -row["difference"])
    return rows[:keep]


def ask_ferry(page: Any) -> dict[str, Any]:
    started = time.perf_counter()
    page.locator("button.chip-example", has_text=FERRY_QUESTION).first.click()
    page.wait_for_selector("text=Meaning check (NLI)", timeout=5 * 60 * 1000)
    wall = round((time.perf_counter() - started) * 1000)
    page.locator("summary", has_text="Checks that ran on this answer").first.click()
    row = page.locator(
        "tr", has=page.locator("th", has_text="Meaning (entailment)")
    ).first.inner_text()
    lines = page.locator("text=/Meaning check \\(NLI\\):/").all_inner_texts()
    return {"wall_ms": wall, "summary_row": " ".join(row.split()), "statement_lines": lines}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", help="measure this deployed demo instead of frontend/dist")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validation-sample", type=int, default=120)
    parser.add_argument(
        "--batch", type=int, default=12, help="pairs per classify call (about one answer)"
    )
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--headed", action="store_true")
    parser.add_argument("--skip-python", action="store_true")
    parser.add_argument("--label", default="", help="a name for this build, kept in the report")
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    server = site = None
    url = args.url
    if not url:
        dist = ROOT / "frontend" / "dist"
        if not (dist / "runtime" / "manifest.json").is_file():
            print("Build the demo first (see scripts/e2e_demo.py).")
            return 2
        server, url, site = _serve(dist)
    pairs = load_pairs(args.validation_sample, args.seed)
    profile = Path(tempfile.mkdtemp(prefix="groundedai-nli-bench-"))
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "target": "deployed" if args.url else "local build (frontend/dist)",
        "label": args.label,
        "url": args.url,
        "machine": {"processor": platform.processor(), "system": platform.platform(terse=True)},
        "backend": "onnxruntime-web WebAssembly, 1 thread (no WebGPU)",
        "pairs": {
            name: sum(p["set"] == name for p in pairs) for name in sorted({p["set"] for p in pairs})
        },
    }
    try:
        with sync_playwright() as playwright:
            # 1. Empty profile: download, start, classify, one question.
            context, page = open_demo(playwright, profile, url, args.headed)
            report["browser"] = page.evaluate("navigator.userAgent")
            downloads = Downloads()
            downloads.watch(page)
            report["memory_before_mb"] = chromium_memory(profile)
            report["first_load"] = switch_on(page)
            page.wait_for_timeout(1000)
            report["memory_loaded_mb"] = chromium_memory(profile)
            report["received"] = downloads.files
            known = [size for size in downloads.files.values() if size >= 0]
            report["received_total_mb"] = round(sum(known) / 2**20, 1) if known else None
            browser_probs, per_pair = classify(page, pairs, args.batch)
            report["memory_after_classify_mb"] = chromium_memory(profile)
            report["per_pair_ms"] = {
                "median": round(statistics.median(per_pair), 1),
                "mean": round(statistics.fmean(per_pair), 1),
                "max_batch_mean": round(max(per_pair), 1),
                "batches": len(per_pair),
                "batch_size": args.batch,
            }
            report["end_to_end"] = ask_ferry(page)
            # Switching the check off stops its worker, which frees the model's memory.
            page.get_by_label("Meaning check with an NLI model").click()
            page.wait_for_function("window.__groundedaiNli && !window.__groundedaiNli.loaded")
            page.wait_for_timeout(2000)
            report["memory_after_switch_off_mb"] = chromium_memory(profile)
            context.close()
            # 2. Same profile, new browser: the files come from the cache.
            context, page = open_demo(playwright, profile, url, args.headed)
            cached = Downloads()
            cached.watch(page)
            report["cached_load"] = switch_on(page)
            report["cached_received"] = cached.files
            context.close()
    finally:
        if server is not None:
            server.shutdown()
        if site is not None:
            shutil.rmtree(site, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)
    if not args.skip_python and (MODEL_DIR / "onnx" / "model_quantized.onnx").is_file():
        python_probs, python_ms = python_probabilities(pairs)
        report["python_onnxruntime"] = {
            "per_pair_ms": round(python_ms / len(pairs), 1),
            "threads": 1,
        }
        report["agreement_with_python"] = agreement(browser_probs, python_probs)
        by_set: dict[str, Any] = {}
        for name in report["pairs"]:
            index = [i for i, p in enumerate(pairs) if p["set"] == name]
            by_set[name] = agreement(
                [browser_probs[i] for i in index], [python_probs[i] for i in index]
            )
        report["agreement_by_set"] = by_set
        report["largest_differences"] = largest_differences(pairs, browser_probs, python_probs, 5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    hidden = ("received", "cached_received", "largest_differences")
    print(json.dumps({k: v for k, v in report.items() if k not in hidden}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
