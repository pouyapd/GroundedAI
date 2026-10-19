import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import { compareWithExpected, StaticDemoApi, type RecordedAnswers, type RecordedExpectation } from "../api/demo";
import type { AnswerResponse, RetrievalDiagnostics, RetrievalMode } from "../api/types";
import { ANSWER } from "./fixtures";

const EXPECT = (behavior: RecordedExpectation["behavior"]): RecordedExpectation => ({
  question_id: "q1",
  category: "factual",
  behavior,
  reference_answer: null,
  notes: null,
});

function diagnostics(mode: RetrievalMode): RetrievalDiagnostics {
  return {
    mode,
    fusion: mode === "hybrid" ? "rrf" : null,
    reranker: null,
    embedding_model: mode === "lexical" ? null : "bge",
    lexical_candidates: mode === "dense" ? 0 : 20,
    dense_candidates: mode === "lexical" ? 0 : 20,
    fused_candidates: 20,
    duplicates_removed: 0,
    timings_ms: { total_ms: 5 },
    warnings: [],
  };
}

function recorded(mode: RetrievalMode, topK: number): AnswerResponse {
  return { ...ANSWER, answer: `${mode}/${topK} answer [S1].`, retrieval: diagnostics(mode) };
}

export const RECORDED: RecordedAnswers = {
  recorded_at: "2026-10-11 18:00 UTC",
  note: "Recorded from a local run.",
  modes: ["hybrid", "lexical"],
  top_k_values: [3, 5],
  defaults: { retrieval_mode: "hybrid", top_k: 5 },
  items: [
    {
      question: "What is the hotel limit?",
      expected: EXPECT("answer"),
      recordings: (["hybrid", "lexical"] as const).flatMap((mode) =>
        [3, 5].map((topK) => ({ retrieval_mode: mode, top_k: topK, key_fact_recall: 1, response: recorded(mode, topK) })),
      ),
    },
  ],
};

function mockFetch(files: Record<string, unknown>) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input).replace(/\?.*$/, "");
    const name = Object.keys(files).find((file) => url.endsWith(`/demo/${file}`));
    if (!name) return new Response("missing", { status: 404 });
    return new Response(JSON.stringify(files[name]), { status: 200 });
  });
}

afterEach(() => vi.restoreAllMocks());

describe("StaticDemoApi requests", () => {
  it("adds the export time to data requests and always revalidates the manifest", async () => {
    const calls: { url: string; cache?: RequestCache }[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      calls.push({ url: String(input), cache: init?.cache });
      const body = String(input).includes("manifest.json") ? { exported_at: "2026-10-18T08:00:00Z" } : { ok: true };
      return new Response(JSON.stringify(body), { status: 200 });
    });
    const api = new StaticDemoApi("./demo");
    await api.systemStatus();
    await api.manifest();
    expect(calls.find((call) => call.url.includes("status.json"))?.url).toBe(
      "./demo/status.json?v=2026-10-18T08%3A00%3A00Z",
    );
    expect(calls.filter((call) => call.url.endsWith("manifest.json")).every((call) => call.cache === "no-cache")).toBe(true);
  });
});

describe("StaticDemoApi", () => {
  it("replays the recording for the default settings, ignoring case and surrounding spaces", async () => {
    mockFetch({ "answers.json": RECORDED });
    const api = new StaticDemoApi("./demo");
    expect(api.mode).toBe("static-demo");
    const response = await api.query({ question: "  what is the HOTEL limit? ", include_diagnostics: true });
    expect(response.answer).toBe("hybrid/5 answer [S1].");
    expect(response.retrieval?.mode).toBe("hybrid");
  });

  it("selects the recording made with the requested mode and passage count", async () => {
    mockFetch({ "answers.json": RECORDED });
    const api = new StaticDemoApi("./demo");
    const lexical = await api.query({ question: "What is the hotel limit?", retrieval_mode: "lexical", top_k: 3, include_diagnostics: true });
    expect(lexical.answer).toBe("lexical/3 answer [S1].");
    expect(lexical.retrieval?.mode).toBe("lexical");
    expect(lexical.retrieval?.dense_candidates).toBe(0);
  });

  it("leaves out diagnostics unless they are requested, like the live API", async () => {
    mockFetch({ "answers.json": RECORDED });
    const response = await new StaticDemoApi("./demo").query({ question: "What is the hotel limit?", include_diagnostics: false });
    expect(response.retrieval).toBeNull();
    expect(response.answer).toBe("hybrid/5 answer [S1].");
  });

  it("says when a setting was not recorded instead of substituting another recording", async () => {
    mockFetch({ "answers.json": RECORDED });
    const error = await new StaticDemoApi("./demo")
      .query({ question: "What is the hotel limit?", retrieval_mode: "dense", top_k: 5 })
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("demo_setting_not_recorded");
    expect((error as ApiError).message).toMatch(/not recorded with dense retrieval and 5 passages/);
  });

  it("refuses questions that were not recorded instead of inventing an answer", async () => {
    mockFetch({ "answers.json": RECORDED });
    const error = await new StaticDemoApi("./demo").query({ question: "Something new?" }).catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("demo_not_recorded");
  });

  it("is read-only", async () => {
    const api = new StaticDemoApi("./demo");
    await expect(api.uploadDocument()).rejects.toMatchObject({ code: "demo_read_only" });
    await expect(api.startEvaluationRun()).rejects.toMatchObject({ code: "demo_read_only" });
  });

  it("reports missing demo files clearly", async () => {
    mockFetch({});
    await expect(new StaticDemoApi("./demo").systemStatus()).rejects.toMatchObject({
      code: "demo_data_missing",
    });
  });
});

describe("compareWithExpected", () => {
  it("applies the evaluation's behaviour rule to the recorded status", () => {
    expect(compareWithExpected(EXPECT("answer"), "answered").matches).toBe(true);
    expect(compareWithExpected(EXPECT("answer"), "insufficient_evidence").matches).toBe(false);
    expect(compareWithExpected(EXPECT("answer"), "invalid_output").matches).toBe(false);
    expect(compareWithExpected(EXPECT("abstain"), "insufficient_evidence").matches).toBe(true);
    expect(compareWithExpected(EXPECT("abstain"), "answered").matches).toBe(false);
    expect(compareWithExpected(EXPECT("flag_conflict"), "insufficient_evidence").matches).toBe(false);
  });

  it("does not claim a verdict it cannot check from the status alone", () => {
    expect(compareWithExpected(EXPECT("flag_conflict"), "answered").matches).toBeNull();
    expect(compareWithExpected(EXPECT("answer_or_clarify"), "answered").matches).toBeNull();
  });
});
