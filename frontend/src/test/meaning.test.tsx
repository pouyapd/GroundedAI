import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { AnswerResponse } from "../api/types";
import { NliModel, meaningLoadErrorMessage } from "../browser/nliModel";
import type { NliRequest, NliResponse } from "../browser/nliProtocol";
import { AnswerView } from "../components/AnswerView";
import { checkRows } from "../components/CheckSummary";
import { checkMeaning } from "../nli/check";
import { judge, nearestPremises, splitSentences, windowsOf, type NliProbabilities } from "../nli/premises";
import { ANSWER } from "./fixtures";

const ENTAIL: NliProbabilities = { entailment: 0.9, neutral: 0.08, contradiction: 0.02 };
const NEUTRAL: NliProbabilities = { entailment: 0.1, neutral: 0.8, contradiction: 0.1 };
const CONTRA: NliProbabilities = { entailment: 0.05, neutral: 0.15, contradiction: 0.8 };

describe("meaning check rules", () => {
  it("splits passages into sentences and windows of one or two sentences", () => {
    expect(splitSentences("Tickets cost 14 euros. Children travel free.\nBikes cost 3 euros.")).toEqual([
      "Tickets cost 14 euros.",
      "Children travel free.",
      "Bikes cost 3 euros.",
    ]);
    const windows = windowsOf([{ label: "S1", text: "One fact. Two facts. Three facts." }]);
    expect(windows.map((w) => w.text)).toEqual([
      "One fact.",
      "Two facts.",
      "Three facts.",
      "One fact. Two facts.",
      "Two facts. Three facts.",
    ]);
  });

  it("keeps the windows that share the most words with the statement", () => {
    const premises = [
      { label: "S1", text: "The library opens at nine." },
      { label: "S1", text: "A single adult ticket costs 14 euros." },
      { label: "S2", text: "Children under five travel free." },
    ];
    expect(nearestPremises("An adult ticket costs 14 euros.", premises, 1)).toEqual([premises[1]]);
    expect(nearestPremises("anything", premises, 5)).toEqual(premises);
  });

  it("is entailed when a window entails it, contradicted only when none does", () => {
    const premise = { label: "S1", text: "x" };
    expect(judge([{ premise, probabilities: NEUTRAL }, { premise, probabilities: ENTAIL }]).meaning).toBe("entailed");
    expect(judge([{ premise, probabilities: CONTRA }, { premise, probabilities: ENTAIL }]).meaning).toBe("entailed");
    expect(judge([{ premise, probabilities: NEUTRAL }, { premise, probabilities: CONTRA }]).meaning).toBe("contradicted");
    expect(judge([{ premise, probabilities: NEUTRAL }]).meaning).toBe("not_entailed");
  });
});

const ANSWERED: AnswerResponse = {
  ...ANSWER,
  status: "answered",
  sources: [
    { ...ANSWER.sources[0]!, label: "S1", text: "A single adult ticket costs 14 euros. Children under five travel free." },
  ],
  claims: [
    { text: "A ticket costs 14 euros.", citations: ["S1"], status: "supported", evidence: [], missing_figures: [], word_overlap: 1, flags: [] },
    { text: "Children pay half price.", citations: ["S1"], status: "partial", evidence: [], missing_figures: [], word_overlap: 0.4, flags: [] },
  ],
  checks_run: ["citation_labels", "lexical_claims"],
};

describe("checkMeaning", () => {
  it("adds one outcome per statement and lists the check as run", async () => {
    const classify = async (pairs: [string, string][]) => ({
      probabilities: pairs.map(([premise, hypothesis]) =>
        hypothesis.includes("14 euros") && premise.includes("14 euros") ? ENTAIL : hypothesis.includes("half") && premise.includes("free") ? CONTRA : NEUTRAL,
      ),
      ms: 5,
    });
    const checked = await checkMeaning(ANSWERED, classify, { model: "test model", backend: "wasm" });
    expect(checked.meaning?.claims.map((claim) => claim.meaning)).toEqual(["entailed", "contradicted"]);
    expect(checked.meaning?.claims[0]?.source).toBe("S1");
    expect(checked.checks_run).toContain("meaning_nli");
    expect(checked.claims).toEqual(ANSWERED.claims); // the lexical results are not changed
  });

  it("leaves refusals alone", async () => {
    const refusal = { ...ANSWERED, status: "insufficient_evidence" as const };
    const classify = async () => {
      throw new Error("must not be called");
    };
    expect(await checkMeaning(refusal, classify, { model: "m", backend: "wasm" })).toBe(refusal);
  });

  it("shows the outcome per statement and in the summary, without merging it with the lexical result", async () => {
    const classify = async (pairs: [string, string][]) => ({ probabilities: pairs.map(() => ENTAIL), ms: 1 });
    const checked = await checkMeaning(
      { ...ANSWERED, verification: { verdict: "partially_verified", claims: 2, supported: 1, wrong_source: 0, added_source: 0, partial: 1, unsupported: 0, notes: 0, references_standard: 2, references_variant: 0, method: "Lexical." } },
      classify,
      { model: "DeBERTa test", backend: "wasm" },
    );
    const rows = Object.fromEntries(checkRows(checked).map((row) => [row.property, row.result]));
    expect(rows["Meaning (entailment)"]).toBe("2 of 2 statements entailed by a passage");
    expect(rows["Lexical match"]).toBe("1 of 2 statements matched retrieved text");
    render(<AnswerView response={checked} />);
    expect(screen.getAllByText(/Meaning check \(NLI\): entailed by S1/)).toHaveLength(2);
  });
});

class FakeWorker {
  onmessage: ((event: MessageEvent<NliResponse>) => void) | null = null;
  onerror: ((event: ErrorEvent) => void) | null = null;
  terminated = false;
  constructor(private readonly reply: (request: NliRequest) => NliResponse[]) {}
  postMessage(request: NliRequest) {
    queueMicrotask(() => {
      for (const message of this.reply(request)) this.onmessage?.({ data: message } as MessageEvent<NliResponse>);
    });
  }
  terminate() {
    this.terminated = true;
  }
}

describe("NliModel", () => {
  it("reports download progress, then classifies", async () => {
    const worker = new FakeWorker((request) =>
      request.kind === "load"
        ? [
            { id: request.id, kind: "progress", fraction: 0.5, bytes: 55_400_000 },
            { id: request.id, kind: "loaded", backend: "wasm", ms: 1200, bytes: 110_753_412 },
          ]
        : [{ id: request.id, kind: "result", probabilities: request.pairs.map(() => ENTAIL), ms: 3 }],
    );
    const model = new NliModel(() => worker as unknown as Worker);
    const progress: string[] = [];
    await model.load((fraction, text) => progress.push(`${fraction} ${text}`));
    expect(progress).toEqual(["0.5 55 of 111 MB"]);
    expect(model.loaded).toBe(true);
    const result = await model.classify([["premise", "claim"]]);
    expect(result.probabilities).toEqual([ENTAIL]);
  });

  it("can be loaded again after a failed download", async () => {
    let attempt = 0;
    const model = new NliModel(
      () =>
        new FakeWorker((request) => {
          attempt += 1;
          return attempt === 1
            ? [{ id: request.id, kind: "error", message: "Download failed (503) for onnx/model_quantized.onnx" }]
            : [{ id: request.id, kind: "loaded", backend: "wasm", ms: 10, bytes: 1 }];
        }) as unknown as Worker,
    );
    await expect(model.load(() => undefined)).rejects.toThrow("503");
    expect(model.loaded).toBe(false);
    await model.load(() => undefined);
    expect(model.loaded).toBe(true);
  });

  it("stops its worker when switched off and can load again", async () => {
    const workers: FakeWorker[] = [];
    const model = new NliModel(() => {
      const worker = new FakeWorker((request) => [{ id: request.id, kind: "loaded", backend: "wasm", ms: 5, bytes: 1 }]);
      workers.push(worker);
      return worker as unknown as Worker;
    });
    await model.load(() => undefined);
    model.unload();
    expect(model.loaded).toBe(false);
    expect(workers[0]?.terminated).toBe(true);
    await expect(model.classify([["premise", "claim"]])).rejects.toThrow("not loaded");
    await model.load(() => undefined);
    expect(model.loaded).toBe(true);
    expect(workers).toHaveLength(2);
  });

  it("explains a failed load and that word matching still runs", () => {
    expect(meaningLoadErrorMessage(new TypeError("Failed to fetch"))).toBe(
      "The meaning check could not be loaded (Failed to fetch). Answers are still checked by word matching. Switch it on again to retry.",
    );
  });
});
