import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { AnswerView } from "../components/AnswerView";
import { CitedText, splitCitations } from "../components/CitedText";
import { formatBytes, formatMs, formatPercent, pageRange } from "../lib/format";
import { validateFile } from "../pages/DocumentsPage";
import { ANSWER } from "./fixtures";

describe("splitCitations", () => {
  it("separates text and labels, including grouped citations", () => {
    expect(splitCitations("Limit is 140 [S1]. Also [S2, s3].")).toEqual([
      { kind: "text", value: "Limit is 140 " },
      { kind: "citation", value: "S1" },
      { kind: "text", value: ". Also " },
      { kind: "citation", value: "S2" },
      { kind: "citation", value: "S3" },
      { kind: "text", value: "." },
    ]);
  });
});

describe("CitedText", () => {
  it("renders valid citations as buttons and invalid ones as marked text", async () => {
    const onCite = vi.fn();
    render(<CitedText text="Fact [S1]. Other [S9]." validLabels={new Set(["S1"])} onCite={onCite} />);
    await userEvent.click(screen.getByRole("button", { name: "Show source S1" }));
    expect(onCite).toHaveBeenCalledWith("S1");
    expect(screen.getByTitle(/S9 was not among the sources/)).toHaveTextContent("S9");
    expect(screen.queryByRole("button", { name: "Show source S9" })).not.toBeInTheDocument();
  });
});

describe("AnswerView", () => {
  it("shows grounding status, demo label, warnings and sources", () => {
    render(<AnswerView response={ANSWER} />);
    expect(screen.getByText("Partly grounded")).toBeInTheDocument();
    expect(screen.getByText(/not an LLM/)).toBeInTheDocument();
    expect(screen.getByText(/Those citations are invalid/)).toBeInTheDocument();
    expect(screen.getByText("Cited sources")).toBeInTheDocument();
    expect(screen.getByText("Travel Policy")).toBeInTheDocument();
    expect(screen.getByText("p. 2")).toBeInTheDocument();
    expect(screen.getByText("Retrieved but not cited (1)")).toBeInTheDocument();
    expect(screen.getByText("Contains instruction-like text")).toBeInTheDocument();
  });

  it("explains abstentions instead of showing an answer", () => {
    render(
      <AnswerView
        response={{ ...ANSWER, status: "insufficient_evidence", answer: null, grounding: null, abstention_reason: "No match.", sources: [], sentences: [], warnings: [] }}
      />,
    );
    expect(screen.getByText("Not enough evidence")).toBeInTheDocument();
    expect(screen.getByText(/do not contain enough evidence/)).toBeInTheDocument();
  });

  it("labels LLM output with provider and model", () => {
    render(<AnswerView response={{ ...ANSWER, provider: { name: "ollama", model: "qwen2.5:1.5b", is_llm: true } }} />);
    expect(screen.getByText("qwen2.5:1.5b")).toBeInTheDocument();
    expect(screen.queryByText(/not an LLM/)).not.toBeInTheDocument();
  });

  it("marks a recorded answer that the dataset says should have been a refusal", () => {
    render(
      <AnswerView
        response={ANSWER}
        recordedNote="Recorded from a local run."
        expected={{
          question_id: "q046",
          category: "out_of_scope",
          behavior: "abstain",
          reference_answer: null,
          notes: "General knowledge; not in the documents.",
          key_fact_recall: null,
        }}
      />,
    );
    expect(screen.getByText("Behaviour does not match")).toBeInTheDocument();
    expect(screen.getByText("It answered, but the dataset expects a refusal.")).toBeInTheDocument();
    expect(screen.getByText(/General knowledge; not in the documents\./)).toBeInTheDocument();
    expect(screen.getByText(/category out of scope/)).toBeInTheDocument();
  });

  it("expands the full passage on demand", async () => {
    render(<AnswerView response={ANSWER} />);
    await userEvent.click(screen.getAllByRole("button", { name: "Show full passage" })[0]!);
    expect(screen.getByText(/Accommodation\. The hotel limit/)).toBeInTheDocument();
  });
});

describe("validateFile", () => {
  const file = (name: string, size: number) => new File([new Uint8Array(size)], name);
  it("checks extension and size before upload", () => {
    expect(validateFile(file("a.exe", 10), [".pdf", ".txt"], 100)).toMatch(/Unsupported/);
    expect(validateFile(file("a.txt", 0), [".txt"], 100)).toMatch(/empty/);
    expect(validateFile(file("a.txt", 200), [".txt"], 100)).toMatch(/larger/);
    expect(validateFile(file("Report.PDF", 10), [".pdf"], 100)).toBeNull();
  });
});

describe("format helpers", () => {
  it("formats values for display", () => {
    expect(formatBytes(2048)).toBe("2.0 KB");
    expect(formatMs(0.4)).toBe("<1 ms");
    expect(formatMs(1500)).toBe("1.50 s");
    expect(formatMs(null)).toBe("–");
    expect(formatPercent(0.965, 1)).toBe("96.5%");
    expect(pageRange(3, 4)).toBe("pp. 3–4");
    expect(pageRange(null, null)).toBe("");
  });
});
