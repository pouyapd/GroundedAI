import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import type { AnswerResponse } from "../api/types";
import { AnswerView } from "../components/AnswerView";
import { checkRows } from "../components/CheckSummary";
import { ANSWER } from "./fixtures";

const VERIFICATION = {
  verdict: "partially_verified" as const,
  claims: 2,
  supported: 1,
  wrong_source: 0,
  added_source: 0,
  partial: 1,
  unsupported: 0,
  notes: 0,
  references_standard: 2,
  references_variant: 0,
  method: "Lexical: words and figures, not meaning.",
};

const CHECKED: AnswerResponse = {
  ...ANSWER,
  answer: "A ticket costs 14 euros [S1]. It includes a free hot meal on board [S1]. Children travel free [S2].",
  citation_check: { cited: ["S1", "S2"], valid: ["S1"], invalid: ["S2"], all_valid: false, sentences_total: 3, sentences_cited: 3 },
  checks_run: ["citation_labels", "relevance", "lexical_claims", "added_details", "disagreement"],
  relevance: { method: "lexical", asked: ["ticket", "cost", "meal"], unmatched: ["meal"], answer_coverage: 1 },
  verification: VERIFICATION,
  claims: [
    { text: "A ticket costs 14 euros.", citations: ["S1"], status: "supported", evidence: [], missing_figures: [], word_overlap: 1, flags: [] },
    {
      text: "It includes a free hot meal on board.",
      citations: ["S1"],
      status: "partial",
      evidence: [],
      missing_figures: [],
      word_overlap: 0.6,
      flags: ["added_detail"],
      added_details: ["hot meal"],
    },
  ],
};

describe("checks that ran on an answer", () => {
  it("lists each property on its own line and never claims a meaning or correctness check", () => {
    const rows = Object.fromEntries(checkRows(CHECKED).map((row) => [row.property, row.result]));
    expect(rows["Citation labels"]).toBe("1 label(s) name a source that was not provided: S2");
    expect(rows["Lexical match"]).toBe("1 of 2 statements matched retrieved text");
    expect(rows["Added details"]).toBe("1 statement(s) add words that no passage states");
    expect(rows["Question relevance"]).toBe("the passages do not mention: meal");
    expect(rows["Meaning (entailment)"]).toBe("not checked in this answer");
    expect(rows["Answer correctness"]).toBe("not assessed");
  });

  it("shows the checks and the added detail in the answer view", async () => {
    render(<AnswerView response={CHECKED} />);
    await userEvent.click(screen.getByText("Checks that ran on this answer"));
    const table = screen.getByRole("table", { name: "Checks that ran" });
    expect(within(table).getByText("Question relevance")).toBeInTheDocument();
    expect(screen.getByText(/Words no retrieved passage states \(an added detail\): hot meal/)).toBeInTheDocument();
  });

  it("is not shown for answers recorded before the checks were listed", () => {
    render(<AnswerView response={{ ...CHECKED, checks_run: undefined }} />);
    expect(screen.queryByText("Checks that ran on this answer")).not.toBeInTheDocument();
  });
});

describe("two versions of the same document", () => {
  it("names each version and the requirement word", () => {
    const response: AnswerResponse = {
      ...ANSWER,
      conflicts: [
        {
          kind: "different_versions",
          explanation: "Sources S1 and S2 come from two versions of the same document.",
          answer_reports: [],
          statements: [
            {
              chunk_id: "a:1",
              document_id: "a",
              document_title: "Access Standard (Revision A)",
              page_start: 2,
              page_end: 2,
              page: 2,
              label: "S1",
              quote: "Administrators SHOULD NOT share accounts.",
              quote_start: 0,
              quote_end: 41,
              values: [],
              denies_requirement: false,
              dates: [],
              edition: "Revision A, November 2018",
              requirement: "SHOULD NOT",
            },
            {
              chunk_id: "b:1",
              document_id: "b",
              document_title: "Access Standard (Revision B)",
              page_start: 2,
              page_end: 2,
              page: 2,
              label: "S2",
              quote: "Administrators SHALL NOT share accounts.",
              quote_start: 0,
              quote_end: 40,
              values: [],
              denies_requirement: false,
              dates: [],
              edition: "Revision B, March 2024",
              requirement: "SHALL NOT",
            },
          ],
        },
      ],
    };
    render(<AnswerView response={response} />);
    expect(screen.getByText("Two versions of the same document differ")).toBeInTheDocument();
    expect(screen.getByText(/Version: Revision A, November 2018 · Value: "SHOULD NOT"/)).toBeInTheDocument();
    expect(screen.getByText(/Version: Revision B, March 2024 · Value: "SHALL NOT"/)).toBeInTheDocument();
  });
});
