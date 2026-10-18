/**
 * One test per interactive control (see docs/UI_CONTROL_AUDIT.md): navigation, retrieval mode,
 * passage count, diagnostics, question submission, recorded questions, theme, and the
 * static-demo behaviour of each page.
 */
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { StaticDemoApi, type RecordedAnswers } from "../api/demo";
import type { AnswerResponse, OperationalMetrics, RetrievalMode } from "../api/types";
import { ANSWER, STATUS, fakeApi } from "./fixtures";

const METRICS: OperationalMetrics = {
  uptime_s: 60,
  documents_by_status: { ready: 2, failed: 0 },
  documents_total: 2,
  chunks_total: 9,
  queries_total: 3,
  queries_window_hours: 24,
  queries_in_window: 0,
  queries_by_status: { answered: 3 },
  abstention_rate: 0,
  citation_valid_rate: 1,
  latency_ms: {},
  hourly: [],
  http_requests_total: 10,
  http_server_errors_total: 0,
};

afterEach(() => {
  vi.restoreAllMocks();
  document.documentElement.removeAttribute("data-theme");
});

async function askLive(api = fakeApi()) {
  render(<App api={api} />);
  await userEvent.type(screen.getByLabelText("Question"), "What is the hotel limit?");
  return api;
}

describe("navigation", () => {
  it("switches the visible page for each tab", async () => {
    render(<App api={fakeApi({ metrics: vi.fn().mockResolvedValue(METRICS) })} />);
    const nav = screen.getByRole("navigation", { name: "Sections" });
    await userEvent.click(within(nav).getByRole("button", { name: "Documents" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Documents" })).toBeInTheDocument();
    expect(window.location.hash).toBe("#documents");
    await userEvent.click(within(nav).getByRole("button", { name: "Evaluation" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Evaluation" })).toBeInTheDocument();
    await userEvent.click(within(nav).getByRole("button", { name: "Operations" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Operations" })).toBeInTheDocument();
    await userEvent.click(within(nav).getByRole("button", { name: "Ask" }));
    expect(screen.getByRole("heading", { level: 1, name: "Ask your documents" })).toBeInTheDocument();
    expect(within(nav).getByRole("button", { name: "Ask" })).toHaveAttribute("aria-current", "page");
  });

  it("keeps the answer and settings on the Ask page after visiting another tab", async () => {
    const api = await askLive();
    await userEvent.click(screen.getByRole("button", { name: "Lexical" }));
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(await screen.findByText("Cited sources")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Documents" }));
    expect(screen.getByText("Cited sources")).not.toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Ask" }));
    expect(screen.getByText("Cited sources")).toBeVisible();
    expect(screen.getByRole("button", { name: "Lexical" })).toHaveAttribute("aria-pressed", "true");
    expect(api.query).toHaveBeenCalledTimes(1);
  });

  it("still switches pages when the hosting frame blocks history changes", async () => {
    vi.spyOn(window.history, "pushState").mockImplementation(() => {
      throw new DOMException("blocked", "SecurityError");
    });
    render(<App api={fakeApi()} />);
    await userEvent.click(screen.getByRole("button", { name: "Documents" }));
    expect(await screen.findByRole("heading", { level: 1, name: "Documents" })).toBeInTheDocument();
  });

  it("follows Back/Forward (popstate) and direct links", async () => {
    render(<App api={fakeApi()} />);
    window.history.pushState(null, "", "#evaluation");
    fireEvent.popState(window);
    expect(await screen.findByRole("heading", { level: 1, name: "Evaluation" })).toBeInTheDocument();
  });
});

describe("Ask controls against the live API", () => {
  it.each<[string, RetrievalMode]>([
    ["Lexical", "lexical"],
    ["Dense", "dense"],
    ["Hybrid", "hybrid"],
  ])("%s sends retrieval_mode=%s to the API", async (label, mode) => {
    const api = await askLive();
    await userEvent.click(screen.getByRole("button", { name: label }));
    expect(screen.getByRole("button", { name: label })).toHaveAttribute("aria-pressed", "true");
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(api.query).toHaveBeenLastCalledWith(expect.objectContaining({ retrieval_mode: mode }));
  });

  it("describes the selected retrieval mode", async () => {
    render(<App api={fakeApi()} />);
    await userEvent.click(screen.getByRole("button", { name: "Lexical" }));
    expect(screen.getByText(/BM25 keyword search only/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Dense" }));
    expect(screen.getByText(/Embedding similarity search only/)).toBeInTheDocument();
  });

  it("sends the chosen passage count as top_k", async () => {
    const api = await askLive();
    await userEvent.selectOptions(screen.getByLabelText("Passages"), "8");
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(api.query).toHaveBeenLastCalledWith(expect.objectContaining({ top_k: 8 }));
  });

  it("requests diagnostics only when the box is ticked", async () => {
    const api = await askLive();
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(api.query).toHaveBeenLastCalledWith(expect.objectContaining({ include_diagnostics: true }));
    await userEvent.click(screen.getByLabelText("Retrieval diagnostics"));
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(api.query).toHaveBeenLastCalledWith(expect.objectContaining({ include_diagnostics: false }));
  });

  it("submits with Ctrl+Enter", async () => {
    const api = await askLive();
    await userEvent.keyboard("{Control>}{Enter}{/Control}");
    expect(api.query).toHaveBeenCalledWith(expect.objectContaining({ question: "What is the hotel limit?" }));
  });

  it("asks an example question when its chip is clicked", async () => {
    const api = fakeApi();
    render(<App api={api} />);
    await userEvent.click(screen.getByRole("button", { name: "What are the payment terms for Aldervale suppliers?" }));
    expect(api.query).toHaveBeenCalledWith(
      expect.objectContaining({ question: "What are the payment terms for Aldervale suppliers?" }),
    );
    expect(await screen.findByText("Cited sources")).toBeInTheDocument();
  });

  it("each new question produces a new API call (no cached replay)", async () => {
    const api = await askLive();
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(api.query).toHaveBeenCalledTimes(2);
  });

  it("says when the settings changed after an answer and re-asks with the new ones", async () => {
    const api = await askLive();
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(await screen.findByText(/Hybrid retrieval · 2 of up to 5 passages used/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Dense" }));
    expect(screen.getByText(/The answer below used Hybrid retrieval with 5 passages/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Ask again with Dense, 5 passages" }));
    expect(api.query).toHaveBeenLastCalledWith(expect.objectContaining({ retrieval_mode: "dense", top_k: 5 }));
  });

  it("shows a loading state while the answer is generated", async () => {
    let resolve: (value: AnswerResponse) => void = () => undefined;
    const api = fakeApi({ query: vi.fn().mockReturnValue(new Promise<AnswerResponse>((done) => (resolve = done))) });
    await askLive(api);
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(screen.getByText(/Retrieving passages and generating an answer/)).toBeInTheDocument();
    resolve(ANSWER);
    expect(await screen.findByText("Cited sources")).toBeInTheDocument();
  });
});

describe("theme", () => {
  it("cycles system, light and dark and applies it to the page", async () => {
    render(<App api={fakeApi()} />);
    const button = screen.getByRole("button", { name: /Theme:/ });
    await userEvent.click(button);
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    await userEvent.click(button);
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    await userEvent.click(button);
    expect(document.documentElement).not.toHaveAttribute("data-theme");
  });
});

// ------------------------------------------------------------------ static demo
function recording(mode: RetrievalMode, topK: number): AnswerResponse {
  return {
    ...ANSWER,
    answer: `Answer recorded with ${mode} retrieval and ${topK} passages [S1].`,
    retrieval: {
      mode,
      fusion: null,
      reranker: null,
      embedding_model: null,
      lexical_candidates: mode === "dense" ? 0 : 10,
      dense_candidates: mode === "lexical" ? 0 : 10,
      fused_candidates: 10,
      duplicates_removed: 0,
      timings_ms: { total_ms: 3 },
      warnings: [],
    },
  };
}

const MODES: RetrievalMode[] = ["hybrid", "lexical", "dense"];
const DEMO_ANSWERS: RecordedAnswers = {
  recorded_at: "2026-10-12 19:08 UTC",
  note: "Recorded locally.",
  modes: MODES,
  top_k_values: [3, 5],
  defaults: { retrieval_mode: "hybrid", top_k: 5 },
  items: [
    {
      question: "What is the hotel limit?",
      expected: { question_id: "q1", category: "factual", behavior: "answer", reference_answer: "€140.", notes: null },
      recordings: MODES.flatMap((mode) =>
        [3, 5].map((topK) => ({ retrieval_mode: mode, top_k: topK, key_fact_recall: 1, response: recording(mode, topK) })),
      ),
    },
  ],
};

function demoApi() {
  const files: Record<string, unknown> = {
    "status.json": STATUS,
    "manifest.json": { exported_at: "2026-10-12 19:08 UTC", note: "Recorded locally." },
    "answers.json": DEMO_ANSWERS,
    "documents.json": { items: [], total: 0, offset: 0, limit: 100 },
    "metrics.json": METRICS,
    "reports/index.json": { items: [] },
  };
  vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
    const name = Object.keys(files).find((file) => String(input).replace(/\?.*$/, "").endsWith(`/demo/${file}`));
    return name ? new Response(JSON.stringify(files[name]), { status: 200 }) : new Response("missing", { status: 404 });
  });
  return new StaticDemoApi("./demo");
}

describe("static demo", () => {
  it("keeps the mode and passage controls working by replaying the matching recording", async () => {
    render(<App api={demoApi()} />);
    const chip = await screen.findByRole("button", { name: "What is the hotel limit?" });
    expect(screen.getByRole("button", { name: "Lexical" })).toBeEnabled();
    expect(screen.getByLabelText("Passages")).toBeEnabled();
    expect(screen.getByText(/replays a recording made with exactly that setting/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Lexical" }));
    await userEvent.selectOptions(screen.getByLabelText("Passages"), "3");
    await userEvent.click(chip);
    expect(await screen.findByText(/Answer recorded with lexical retrieval and 3 passages/)).toBeInTheDocument();
    expect(screen.getByText("Recorded answer (static demo)")).toBeInTheDocument();
    expect(screen.getByText("Behaviour matches")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Dense" }));
    await userEvent.click(screen.getByRole("button", { name: "Ask again with Dense, 3 passages" }));
    expect(await screen.findByText(/Answer recorded with dense retrieval and 3 passages/)).toBeInTheDocument();
  });

  it("explains that free-text questions need the live application", async () => {
    render(<App api={demoApi()} />);
    await screen.findByRole("button", { name: "What is the hotel limit?" });
    await userEvent.type(screen.getByLabelText("Question"), "Something that was never recorded?");
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(await screen.findByText(/can only replay the recorded example questions/)).toBeInTheDocument();
  });

  it("hides upload and refresh actions that need a backend, and says why", async () => {
    render(<App api={demoApi()} />);
    await userEvent.click(screen.getByRole("button", { name: "Documents" }));
    expect(await screen.findByText("Uploads are disabled in the static demo")).toBeInTheDocument();
    expect(screen.queryByRole("form", { name: "Upload a document" })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Operations" }));
    expect(await screen.findByText("Snapshot, not live")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Refresh" })).not.toBeInTheDocument();
  });
});
