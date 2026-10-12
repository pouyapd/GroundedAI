import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import { compareWithExpected, StaticDemoApi, type RecordedExpectation } from "../api/demo";
import { ANSWER } from "./fixtures";

const EXPECT = (behavior: RecordedExpectation["behavior"]): RecordedExpectation => ({
  question_id: "q1",
  category: "factual",
  behavior,
  reference_answer: null,
  notes: null,
  key_fact_recall: null,
});

const RECORDED = {
  recorded_at: "2026-10-11 18:00 UTC",
  note: "Recorded from a local run.",
  items: [{ question: "What is the hotel limit?", response: ANSWER }],
};

function mockFetch(files: Record<string, unknown>) {
  return vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const url = String(input);
    const name = Object.keys(files).find((file) => url.endsWith(`/demo/${file}`));
    if (!name) return new Response("missing", { status: 404 });
    return new Response(JSON.stringify(files[name]), { status: 200 });
  });
}

afterEach(() => vi.restoreAllMocks());

describe("StaticDemoApi", () => {
  it("replays recorded answers, ignoring case and surrounding spaces", async () => {
    mockFetch({ "answers.json": RECORDED });
    const api = new StaticDemoApi("./demo");
    expect(api.mode).toBe("static-demo");
    await expect(api.query({ question: "  what is the HOTEL limit? " })).resolves.toEqual(ANSWER);
  });

  it("refuses questions that were not recorded instead of inventing an answer", async () => {
    mockFetch({ "answers.json": RECORDED });
    const error = await new StaticDemoApi("./demo").query({ question: "Something new?" }).catch((e) => e);
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
