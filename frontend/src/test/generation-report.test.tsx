import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { App } from "../App";
import type { GenerationReport } from "../api/types";
import { invalidCount, promptLabel, reportOptionLabel } from "../pages/EvaluationPage";
import { fakeApi } from "./fixtures";

const SUMMARY = {
  n: 5,
  answered: 1,
  abstained: 1,
  errors: 0,
  failure_rate: 0,
  behavior_accuracy: 0.2,
  abstention_recall: 1,
  abstention_precision: 0.25,
  citation_validity: 1,
  grounded_rate: 0,
  evidence_coverage: 0,
  key_fact_recall: 0,
  injection_resistance: null,
  latency_ms: { n: 5, mean: 1, p50: 32300, p95: 52300, max: 60000 },
};

const REPORT: GenerationReport = {
  report_id: "generation-test",
  kind: "generation",
  title: "Answer evaluation: test model",
  generated_at: "2026-10-12T14:58:59+00:00",
  measured: true,
  dataset: { name: "groundedai-synthetic-v1", version: "1.0.0", questions_evaluated: 5 },
  provider: { name: "huggingface-local", model: "Qwen/Qwen2.5-0.5B-Instruct", is_llm: true },
  retrieval: { mode: "hybrid", top_k: 5, embedding_model: "x", reranker: null },
  environment: { cpu: "CPU", power_source: "ac" },
  summary: SUMMARY,
  by_category: { multi_chunk: SUMMARY, out_of_scope: { ...SUMMARY, answered: 0, abstained: 5 } },
  limitations: ["Deterministic checks only."],
};

describe("generation report", () => {
  it("counts replies that were neither answers nor abstentions", () => {
    expect(invalidCount(SUMMARY)).toBe(3);
    expect(invalidCount({ n: 2, answered: 3, abstained: 0, errors: 0 })).toBe(0);
  });

  it("names the prompt so two runs of the same model can be told apart", () => {
    const item = { id: "x", kind: "generation", title: "Answer evaluation: ollama (qwen2.5:1.5b)", generated_at: null, dataset: null, n_configurations: 1 };
    expect(reportOptionLabel({ ...item, prompt_version: "answer-v2" })).toMatch(/^Answer evaluation: ollama \(qwen2\.5:1\.5b\) · answer-v2 · /);
    expect(reportOptionLabel({ ...item, title: "Answer evaluation: ollama (qwen2.5:1.5b, answer-v2)", prompt_version: "answer-v2" })).not.toMatch(/answer-v2\) · answer-v2/);
    expect(promptLabel({})).toBe("answer-v1");
    expect(promptLabel({ prompt_version: "answer-v2", run_config: { prompt_fingerprint: "e928c4235d03" } })).toBe("answer-v2 (e928c4235d03)");
  });

  it("shows the scoring-version-3 measures only when the report has them", async () => {
    window.location.hash = "evaluation";
    const v3 = { ...REPORT, prompt_version: "answer-v2", summary: { ...SUMMARY, answer_rate_answerable: 1, incorrect_refusal_rate: 0, citation_format_compliance: 0.265, evidence_support_rate: 0.722 } };
    const api = fakeApi({
      listReports: vi.fn().mockResolvedValue([{ id: "generation-test", kind: "generation", title: REPORT.title, generated_at: REPORT.generated_at, dataset: "x", n_configurations: 1, prompt_version: "answer-v2" }]),
      getReport: vi.fn().mockResolvedValue(v3),
    });
    render(<App api={api} />);
    expect(await screen.findByText("Wrong refusals (answerable)")).toBeInTheDocument();
    expect(screen.getByText(/prompt answer-v2/)).toBeInTheDocument();
  });

  it("shows invalid outputs, latency and readable category names", async () => {
    window.location.hash = "evaluation";
    const api = fakeApi({
      listReports: vi.fn().mockResolvedValue([
        { id: "generation-test", kind: "generation", title: REPORT.title, generated_at: REPORT.generated_at, dataset: "x", n_configurations: 1 },
      ]),
      getReport: vi.fn().mockResolvedValue(REPORT),
    });
    render(<App api={api} />);
    expect(await screen.findByText(/1 answered, 1 abstained, 3\s+invalid output, 0 errors/)).toBeInTheDocument();
    expect(screen.getByText(/p50 32\.3 s, p95\s+52\.3 s/)).toBeInTheDocument();
    expect(screen.getByText("multi chunk")).toBeInTheDocument();
    expect(screen.getByText("out of scope")).toBeInTheDocument();
    expect(screen.queryByText("Baseline, not a language model")).not.toBeInTheDocument();
    expect(screen.queryByText("Wrong refusals (answerable)")).not.toBeInTheDocument(); // old report: no v3 fields
    expect(screen.getByText(/prompt answer-v1/)).toBeInTheDocument();
  });
});
