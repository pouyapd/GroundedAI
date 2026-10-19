import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { App } from "../App";
import type { AnswerResponse, GenerationReport } from "../api/types";
import { AnswerView } from "../components/AnswerView";
import { ANSWER, STATUS, fakeApi } from "./fixtures";

const EVIDENCE = {
  chunk_id: "doc_x:0000",
  document_id: "doc_x",
  document_title: "Travel Policy",
  page_start: 2,
  page_end: 2,
};

const CHECKED: AnswerResponse = {
  ...ANSWER,
  answer: "[Source 1]: The hotel limit is €140 per night in capital cities. Taxis are free.",
  grounding: "ungrounded",
  sources: ANSWER.sources.map((source) => ({ ...source, cited: false, supports_claims: source.label === "S1" ? [0] : [] })),
  claims: [
    {
      text: "The hotel limit is €140 per night in capital cities.",
      citations: [],
      status: "added_source",
      evidence: [{ ...EVIDENCE, label: "S1", quote: "The hotel limit is €140 per night in capital cities.", quote_start: 15, quote_end: 67 }],
      missing_figures: [],
      word_overlap: 1,
      flags: [],
    },
    { text: "Taxis are free.", citations: ["S2"], status: "unsupported", evidence: [], missing_figures: [], word_overlap: 0, flags: [] },
  ],
  verification: {
    verdict: "partially_verified",
    claims: 2,
    supported: 0,
    wrong_source: 0,
    added_source: 1,
    partial: 0,
    unsupported: 1,
    notes: 0,
    references_standard: 0,
    references_variant: 1,
    method: "Lexical: words and figures, not meaning.",
  },
  conflicts: [
    {
      kind: "conflict",
      explanation: "Sources S1 and S3 disagree: S1 says €140, S3 says €120.",
      answer_reports: ["S1"],
      statements: [
        { ...EVIDENCE, label: "S1", quote: "The hotel limit is €140 per night.", quote_start: 0, quote_end: 33, values: ["€140"], denies_requirement: false, dates: [] },
        {
          chunk_id: "doc_z:0000",
          document_id: "doc_z",
          document_title: "Finance FAQ",
          page_start: null,
          page_end: null,
          label: "S3",
          quote: "The nightly hotel limit is €120.",
          quote_start: 0,
          quote_end: 32,
          values: ["€120"],
          denies_requirement: false,
          dates: ["updated May 2025"],
        },
      ],
    },
  ],
};

describe("claim check in the answer view", () => {
  it("shows the verdict, each statement's status and the exact supporting quote", () => {
    render(<AnswerView response={CHECKED} />);
    expect(screen.getByText("Some statements not found in the sources")).toBeInTheDocument();
    expect(screen.getByText(/model citations: no valid citations/)).toBeInTheDocument();
    expect(screen.getByText(/1 of 2 statements found in the retrieved passages/)).toBeInTheDocument();
    expect(screen.getByText("Found in a passage, not cited")).toBeInTheDocument();
    expect(screen.getByText("Not found in the sources")).toBeInTheDocument();
    expect(screen.getByText("“The hotel limit is €140 per night in capital cities.”")).toBeInTheDocument();
    expect(screen.getByText(/another form than \[S#\]/)).toBeInTheDocument();
    // a passage that supports a claim is listed as evidence even though the model did not cite it
    expect(screen.getByRole("heading", { name: "Passages used as evidence" })).toBeInTheDocument();
  });

  it("shows both sides of a disagreement without choosing one", () => {
    render(<AnswerView response={CHECKED} />);
    const panel = screen.getByRole("region", { name: "Retrieved passages that disagree" });
    expect(within(panel).getByText("Sources disagree")).toBeInTheDocument();
    expect(within(panel).getByText("“The hotel limit is €140 per night.”")).toBeInTheDocument();
    expect(within(panel).getByText("“The nightly hotel limit is €120.”")).toBeInTheDocument();
    expect(within(panel).getByText(/Value: €140 · The answer reports this value\./)).toBeInTheDocument();
    expect(within(panel).getByText(/Value: €120 · Dates: updated May 2025/)).toBeInTheDocument();
  });

  it("names both sides when one passage requires something and the other says it is not required", () => {
    // Regression: the requiring side was shown as "S1 says ," and "Value:" with nothing after it.
    const explanation =
      "Sources S1 and S2 disagree: S1 says it is required, S2 says there is no such requirement. GroundedAI does not decide which is right; check both documents.";
    const side = { ...EVIDENCE, quote_start: 0, values: [], dates: [] };
    const requirement: AnswerResponse = {
      ...CHECKED,
      conflicts: [
        {
          kind: "conflict",
          explanation,
          answer_reports: [],
          statements: [
            { ...side, label: "S1", quote: "Contractors must complete the security training.", quote_end: 48, denies_requirement: false, affirms_requirement: true },
            {
              ...side,
              label: "S2",
              document_id: "doc_y",
              document_title: "Contractor FAQ",
              quote: "Contractors do not have to complete the security training.",
              quote_end: 58,
              denies_requirement: true,
              affirms_requirement: false,
            },
          ],
        },
      ],
    };
    render(<AnswerView response={requirement} />);
    const panel = screen.getByRole("region", { name: "Retrieved passages that disagree" });
    expect(within(panel).getByText(explanation)).toBeInTheDocument();
    const values = within(panel).getAllByText(/^Value:/).map((element) => element.textContent);
    expect(values).toEqual(["Value: required", "Value: not required"]);
  });

  it("scrolls to a source when its label in a quote is clicked", async () => {
    const scroll = vi.fn();
    Element.prototype.scrollIntoView = scroll;
    render(<AnswerView response={CHECKED} />);
    const [first] = screen.getAllByRole("button", { name: "Show source S1" });
    await userEvent.click(first as HTMLElement);
    expect(scroll).toHaveBeenCalled();
  });

  it("keeps a withheld answer available on request", async () => {
    const withheld: AnswerResponse = {
      ...CHECKED,
      status: "insufficient_evidence",
      answer: null,
      abstention_reason: "No statement in the model's answer could be matched.",
      unverified_answer: "Taxis are free.",
      conflicts: [],
    };
    render(<AnswerView response={withheld} />);
    await userEvent.click(screen.getByText(/Show the withheld answer/));
    expect(screen.getByText("Taxis are free.", { selector: "p" })).toBeVisible();
  });

  it("falls back to the strict check for recordings made before the claim check", () => {
    render(<AnswerView response={ANSWER} />);
    expect(screen.getByText("Partly grounded")).toBeInTheDocument();
    expect(screen.queryByText(/Claim check/)).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Cited sources" })).toBeInTheDocument();
  });
});

describe("dense retrieval with the hashing embedder", () => {
  it("says that the embedder matches words, not meaning", async () => {
    window.location.hash = "ask";
    render(<App api={fakeApi({ systemStatus: vi.fn().mockResolvedValue({ ...STATUS, demo_mode: false }) })} />);
    await userEvent.click(await screen.findByRole("button", { name: "Dense" }));
    expect(screen.getByText(/hashing embedder, which matches words, not meaning/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Lexical" }));
    expect(screen.queryByText(/hashing embedder/)).not.toBeInTheDocument();
  });
});

describe("scoring version 4 in the evaluation page", () => {
  it("shows the claim and disagreement measures and labels re-checked outputs", async () => {
    window.location.hash = "evaluation";
    const summary = {
      n: 2,
      answered: 2,
      abstained: 0,
      errors: 0,
      failure_rate: 0,
      behavior_accuracy: 1,
      abstention_recall: null,
      abstention_precision: null,
      citation_validity: 1,
      grounded_rate: 0.14,
      evidence_coverage: 0.2,
      key_fact_recall: 0.94,
      injection_resistance: 1,
      latency_ms: { n: 2, mean: 1, p50: 10000, p95: 16000, max: 16000 },
      claim_match_rate: 0.983,
      cited_claim_precision: 0.884,
      answers_verified_rate: 0.959,
      key_fact_attribution: 0.895,
      conflict_surfaced_rate: 0.5,
      false_conflict_rate: 0.019,
    };
    const report: GenerationReport = {
      report_id: "revalidated-x",
      kind: "generation",
      title: "Recorded outputs re-checked: ollama (qwen2.5:1.5b, answer-v2)",
      generated_at: "2026-10-14T07:41:12+00:00",
      measured: true,
      prompt_version: "answer-v2",
      revalidation: {
        source_report_id: "generation-ollama-x",
        source_generated_at: "2026-10-13T11:36:18+00:00",
        model_run: false,
        model_input_changed: ["q002", "q052"],
        note: "",
      },
      dataset: { name: "groundedai-synthetic-v1", version: "1.0.0", questions_evaluated: 2 },
      provider: { name: "ollama", model: "qwen2.5:1.5b", is_llm: true },
      retrieval: { mode: "hybrid", top_k: 5, embedding_model: "x", reranker: null },
      environment: { cpu: "CPU", power_source: "ac" },
      summary,
      by_category: { factual: summary },
      limitations: [],
    };
    const api = fakeApi({
      listReports: vi.fn().mockResolvedValue([
        { id: "revalidated-x", kind: "generation", title: report.title, generated_at: report.generated_at, dataset: "x", n_configurations: 1, prompt_version: "answer-v2" },
      ]),
      getReport: vi.fn().mockResolvedValue(report),
    });
    render(<App api={api} />);
    expect(await screen.findByText("Recorded outputs re-checked: the model was not run")).toBeInTheDocument();
    expect(screen.getByText(/For q002, q052 the current pipeline would send the model a different note/)).toBeInTheDocument();
    expect(screen.getByText("Statements found in a passage")).toBeInTheDocument();
    expect(screen.getByText("98%")).toBeInTheDocument();
    expect(screen.getByText("Other questions: false disagreement")).toBeInTheDocument();
  });
});

describe("warnings next to the disagreement panel", () => {
  it("does not repeat a disagreement's explanation in the warning list", () => {
    const explanation = CHECKED.conflicts?.[0]?.explanation ?? "";
    render(<AnswerView response={{ ...CHECKED, warnings: [explanation, "Another warning."] }} />);
    expect(screen.getAllByText(explanation)).toHaveLength(1);
    expect(screen.getByText("Another warning.")).toBeInTheDocument();
  });
});
