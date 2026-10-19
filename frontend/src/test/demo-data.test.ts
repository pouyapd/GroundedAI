/** Checks the exported static-demo data itself (frontend/public/demo/answers.json). */
import { describe, expect, it } from "vitest";

import answersJson from "../../public/demo/answers.json";
import type { RecordedAnswers } from "../api/demo";

const answers = answersJson as unknown as RecordedAnswers;

describe("exported demo recordings", () => {
  it("has one recording per question for every mode and passage count the UI offers", () => {
    expect(answers.modes).toEqual(["hybrid", "lexical", "dense"]);
    expect(answers.top_k_values).toEqual([3, 5, 8, 10]);
    for (const item of answers.items) {
      const settings = item.recordings.map((r) => `${r.retrieval_mode}/${r.top_k}`).sort();
      const wanted = answers.modes.flatMap((mode) => answers.top_k_values.map((k) => `${mode}/${k}`)).sort();
      expect(settings, item.question).toEqual(wanted);
      expect(item.expected, item.question).toBeDefined();
    }
  });

  it("records what each mode really did: diagnostics match the requested setting", () => {
    for (const item of answers.items) {
      for (const recording of item.recordings) {
        const { response, retrieval_mode: mode, top_k: topK } = recording;
        const where = `${item.question} ${mode}/${topK}`;
        expect(response.retrieval?.mode, where).toBe(mode);
        expect(response.sources.length, where).toBeLessThanOrEqual(topK);
        if (mode === "lexical") expect(response.retrieval?.dense_candidates, where).toBe(0);
        if (mode === "dense") expect(response.retrieval?.lexical_candidates, where).toBe(0);
      }
    }
  });

  it("gives different modes different retrieved passages (the controls are not cosmetic)", () => {
    for (const item of answers.items) {
      const passageLists = new Set(item.recordings.map((r) => r.response.sources.map((s) => s.chunk_id).join(",")));
      expect(passageLists.size, item.question).toBeGreaterThan(1);
    }
  });
});
