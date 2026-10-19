import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { App } from "../App";
import { ApiError } from "../api/client";
import { ANSWER, REPORT, STATUS, documentPage, fakeApi, networkError } from "./fixtures";

describe("App", () => {
  it("shows the demo-generator notice when no LLM is configured", async () => {
    render(<App api={fakeApi()} />);
    expect(await screen.findByText("Demo generator active (not an LLM)")).toBeInTheDocument();
  });

  it("shows setup guidance when the backend is offline", async () => {
    render(<App api={fakeApi({ systemStatus: vi.fn().mockRejectedValue(networkError()) })} />);
    expect(await screen.findByText("The GroundedAI API is not reachable")).toBeInTheDocument();
    expect(screen.getAllByText(/groundedai serve/).length).toBeGreaterThan(0);
  });

  it("warns when the configured model is unavailable", async () => {
    const status = {
      ...STATUS,
      demo_mode: false,
      generation: { provider: "ollama", model: "qwen2.5:1.5b", is_llm: true, available: false, detail: "Ollama is not reachable." },
    };
    render(<App api={fakeApi({ systemStatus: vi.fn().mockResolvedValue(status) })} />);
    expect(await screen.findByText(/Model unavailable: ollama/)).toBeInTheDocument();
    expect(screen.getByText(/does not\s+switch to another model/)).toBeInTheDocument();
  });

  it("asks a question and renders the answer", async () => {
    const api = fakeApi();
    render(<App api={api} />);
    await userEvent.type(screen.getByLabelText("Question"), "What is the hotel limit?");
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(await screen.findByText("Cited sources")).toBeInTheDocument();
    expect(api.query).toHaveBeenCalledWith(
      expect.objectContaining({ question: "What is the hotel limit?", retrieval_mode: "hybrid", top_k: 5 }),
    );
  });

  it("disables Ask for too-short questions", async () => {
    render(<App api={fakeApi()} />);
    await userEvent.type(screen.getByLabelText("Question"), "hi");
    expect(screen.getByRole("button", { name: /Ask question/ })).toBeDisabled();
  });

  it("explains a model-unavailable error from the API", async () => {
    const api = fakeApi({
      query: vi.fn().mockRejectedValue(new ApiError(503, "model_unavailable", "Cannot reach Ollama at http://localhost:11434.")),
    });
    render(<App api={api} />);
    await userEvent.type(screen.getByLabelText("Question"), "What is the hotel limit?");
    await userEvent.click(screen.getByRole("button", { name: /Ask question/ }));
    expect(await screen.findByText("The language model is not available")).toBeInTheDocument();
    expect(screen.getByText(/Cannot reach Ollama/)).toBeInTheDocument();
    expect(ANSWER.status).toBe("answered");
  });

  it("shows the empty state on the documents tab", async () => {
    window.location.hash = "/documents";
    render(<App api={fakeApi({ listDocuments: vi.fn().mockResolvedValue(documentPage([])) })} />);
    expect(await screen.findByText("No documents yet")).toBeInTheDocument();
  });

  it("renders a measured retrieval report on the evaluation tab", async () => {
    window.location.hash = "/evaluation";
    const api = fakeApi({
      listReports: vi.fn().mockResolvedValue([
        { id: "retrieval-test", kind: "retrieval", title: REPORT.title, generated_at: REPORT.generated_at, dataset: "x", n_configurations: 1 },
      ]),
    });
    render(<App api={api} />);
    expect(await screen.findByText("Measured")).toBeInTheDocument();
    expect(screen.getByRole("img", { name: "Recall@5" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show table" }));
    expect(screen.getByText("0.965")).toBeInTheDocument();
    await waitFor(() => expect(api.getReport).toHaveBeenCalledWith("retrieval-test"));
  });
});
