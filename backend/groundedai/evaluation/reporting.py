"""Writing benchmark reports as JSON, CSV and Markdown."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

REPORT_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _metric_cell(metric: dict[str, Any] | None) -> str:
    if not metric:
        return "–"
    ci = metric.get("ci95")
    if ci:
        return f"{metric['mean']:.3f} [{ci[0]:.2f}, {ci[1]:.2f}]"
    return f"{metric['mean']:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    if report["kind"] == "retrieval":
        return _render_retrieval(report)
    return _render_generation(report)


def _environment_lines(report: dict[str, Any]) -> list[str]:
    env = report.get("environment", {})
    return [
        f"- Hardware: {env.get('cpu')} ({env.get('logical_cpus')} logical CPUs), "
        f"{env.get('ram_gb')} GB RAM, no GPU; power source: {env.get('power_source')}",
        f"- Software: Python {env.get('python')} on {env.get('os')}",
    ]


def _render_retrieval(report: dict[str, Any]) -> str:
    dataset = report["dataset"]
    settings = report["settings"]
    k = 5 if 5 in settings["k_values"] else max(settings["k_values"])
    lines = [
        f"# {report['title']}",
        "",
        f"Generated {report['generated_at']} · report `{report['report_id']}`",
        "",
        f"- Dataset: {dataset['name']} v{dataset['version']} (fingerprint "
        f"`{dataset['fingerprint']}`), {dataset['documents']} documents, {dataset['chunks']} "
        f"chunks, {dataset['questions_evaluated']} questions with evidence",
        f"- Chunking: {settings['chunking']['chunk_size']} chars, overlap "
        f"{settings['chunking']['chunk_overlap']}; candidates per retriever: "
        f"{settings['candidate_k']}",
        f"- Backend: {settings['backend']}",
        *_environment_lines(report),
        "",
        "Mean over questions with 95% bootstrap interval "
        f"({settings['bootstrap']['samples']} resamples, seed {settings['bootstrap']['seed']}).",
        "",
        f"| Configuration | Recall@1 | Recall@{k} | MRR | nDCG@{k} | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in report["configurations"]:
        metrics = item["metrics"]
        latency = item["latency_ms"]
        lines.append(
            f"| {item['name']} | {_metric_cell(metrics.get('recall@1'))} | "
            f"{_metric_cell(metrics.get(f'recall@{k}'))} | {_metric_cell(metrics.get('mrr'))} | "
            f"{_metric_cell(metrics.get(f'ndcg@{k}'))} | {_fmt(latency.get('p50'), 1)} | "
            f"{_fmt(latency.get('p95'), 1)} |"
        )
    lines += [
        "",
        f"## Paired differences against `{settings['baseline']}`",
        "",
        "| Configuration | Metric | Mean difference | 95% interval | Interval excludes 0 |",
        "|---|---|---|---|---|",
    ]
    for item in report["comparisons"]:
        lines.append(
            f"| {item['configuration']} | {item['metric']} | {item['mean_diff']:+.3f} | "
            f"[{item['ci95_low']:+.3f}, {item['ci95_high']:+.3f}] | "
            f"{'yes' if item['distinguishable'] else 'no'} |"
        )
    lines += ["", "## Limitations", ""] + [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def _render_generation(report: dict[str, Any]) -> str:
    summary = report["summary"]
    provider = report["provider"]
    retrieval = report["retrieval"]
    lines = [
        f"# {report['title']}",
        "",
        f"Generated {report['generated_at']} · report `{report['report_id']}`",
        "",
        f"- Provider: {provider['name']} / {provider['model']} "
        f"({'language model' if provider['is_llm'] else 'extractive baseline, not an LLM'})",
        f"- Retrieval: {retrieval['mode']} ({retrieval.get('fusion') or '-'}), top-k "
        f"{retrieval['top_k']}, embeddings {retrieval['embedding_model']}, reranker "
        f"{retrieval.get('reranker') or 'none'}",
        f"- Dataset: {report['dataset']['name']} v{report['dataset']['version']}, "
        f"{report['dataset']['questions_evaluated']} questions",
        *_environment_lines(report),
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in (
        "behavior_accuracy",
        "abstention_recall",
        "abstention_precision",
        "citation_validity",
        "grounded_rate",
        "evidence_coverage",
        "key_fact_recall",
        "injection_resistance",
        "failure_rate",
    ):
        lines.append(f"| {key.replace('_', ' ')} | {_fmt(summary.get(key))} |")
    latency = summary["latency_ms"]
    lines.append(
        f"| latency p50 / p95 (ms) | {_fmt(latency['p50'], 0)} / {_fmt(latency['p95'], 0)} |"
    )
    lines += [
        "",
        "## By category",
        "",
        "| Category | n | Behaviour ok | Evidence coverage | Key facts | Injection resisted |",
        "|---|---|---|---|---|---|",
    ]
    for category, item in report["by_category"].items():
        lines.append(
            f"| {category} | {item['n']} | {_fmt(item['behavior_accuracy'])} | "
            f"{_fmt(item['evidence_coverage'])} | {_fmt(item['key_fact_recall'])} | "
            f"{_fmt(item['injection_resistance'])} |"
        )
    lines += ["", "## Limitations", ""] + [f"- {text}" for text in report["limitations"]]
    return "\n".join(lines) + "\n"


def write_report(report: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write ``<id>.json``, ``<id>.md`` and a per-query ``<id>.csv``."""
    report_id = str(report["report_id"])
    if not REPORT_ID_RE.fullmatch(report_id):
        raise ValueError("invalid report id")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / f"{report_id}.json",
        "markdown": out_dir / f"{report_id}.md",
        "csv": out_dir / f"{report_id}.csv",
    }
    paths["json"].write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    rows = report.get("per_query", [])
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with paths["csv"].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {k: json.dumps(v) if isinstance(v, list | dict) else v for k, v in row.items()}
                )
    return paths


def list_reports(directory: Path) -> list[dict[str, Any]]:
    if not directory.is_dir():
        return []
    reports = []
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and "report_id" in data and "kind" in data:
            reports.append(data)
    reports.sort(key=lambda item: str(item.get("generated_at", "")), reverse=True)
    return reports


def load_report(directory: Path, report_id: str) -> dict[str, Any] | None:
    if not REPORT_ID_RE.fullmatch(report_id):
        return None
    path = (directory / f"{report_id}.json").resolve()
    if path.parent != directory.resolve() or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else None
