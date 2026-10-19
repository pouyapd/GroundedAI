import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError } from "../api/client";
import { BrowserApi, type RequestBody, type RuntimeLike } from "../browser/BrowserApi";
import type { PreparedQuery, WireResponse } from "../browser/protocol";
import { ANSWER } from "./fixtures";

const encoder = new TextEncoder();

function wire(status: number, body: unknown): WireResponse {
  return { status, headers: [["content-type", "application/json"]], body: encoder.encode(JSON.stringify(body)) };
}

/** A runtime that records requests and answers like the Python side would. */
class FakeRuntime implements RuntimeLike {
  readonly ready = Promise.resolve();
  readonly calls: RequestBody[] = [];
  constructor(private readonly reply: (request: RequestBody) => unknown) {}
  call<T>(request: RequestBody): Promise<T> {
    this.calls.push(request);
    return Promise.resolve(this.reply(request) as T);
  }
  terminate() {}
}

function stubSamples() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) =>
      String(url).endsWith("manifest.json")
        ? new Response(JSON.stringify({ samples: ["a.pdf", "b.pdf"] }))
        : new Response(new Blob(["%PDF-1.4"])),
    ),
  );
}

function first(runtimes: FakeRuntime[]): FakeRuntime {
  const runtime = runtimes[0];
  if (!runtime) throw new Error("no runtime was started");
  return runtime;
}

const created: BrowserApi[] = [];

function makeApi(handler: (request: RequestBody) => unknown) {
  stubSamples();
  const runtimes: FakeRuntime[] = [];
  const api = new BrowserApi("/demo/", () => {
    const runtime = new FakeRuntime(handler);
    runtimes.push(runtime);
    return runtime;
  });
  created.push(api);
  return { api, runtimes };
}

function server(request: RequestBody): unknown {
  if (request.kind === "fetch" && request.path === "/api/v1/documents") return wire(202, { id: "doc_1" });
  if (request.kind === "fetch" && request.path === "/api/v1/query") return wire(200, ANSWER);
  if (request.kind === "prepare") {
    return { generate: true, messages: [{ role: "user", content: "q" }], max_tokens: 64 } satisfies PreparedQuery;
  }
  if (request.kind === "answer") return wire(200, ANSWER);
  return true;
}

afterEach(async () => {
  // Sample loading continues after a test's last call; let it finish before removing the stub.
  await Promise.allSettled(created.splice(0).map((api) => api.whenReady()));
  vi.unstubAllGlobals();
});

describe("BrowserApi", () => {
  it("loads the two sample PDFs through the upload endpoint", async () => {
    const { api, runtimes } = makeApi(server);
    await api.listReports().catch(() => undefined);
    await api.metrics().catch(() => undefined);
    const uploads = first(runtimes).calls.filter((call) => call.kind === "fetch" && call.method === "POST");
    expect(uploads.map((call) => call.kind === "fetch" && call.path)).toEqual(["/api/v1/documents", "/api/v1/documents"]);
  });

  it("sends questions to the in-browser application with the extractive generator", async () => {
    const { api, runtimes } = makeApi(server);
    const result = await api.query({ question: "How much?", document_ids: ["doc_1"] });
    expect(result.answer).toBe(ANSWER.answer);
    const query = first(runtimes).calls.find((call) => call.kind === "fetch" && call.path === "/api/v1/query");
    expect(query && query.kind === "fetch" && new TextDecoder().decode(query.body)).toContain('"document_ids":["doc_1"]');
    expect(first(runtimes).calls.some((call) => call.kind === "prepare")).toBe(false);
  });

  it("asks the page model in two steps and reports its real time and tokens", async () => {
    const { api, runtimes } = makeApi(server);
    await api.selectGenerator("webllm");
    const generate = vi
      .spyOn(api.pageModel, "generate")
      .mockResolvedValue({ text: "It costs 14 euros [S1].", promptTokens: 300, completionTokens: 9, ms: 4200 });
    const result = await api.query({ question: "How much?" });
    const kinds = first(runtimes).calls.map((call) => call.kind).filter((kind) => kind !== "fetch" && kind !== "init");
    expect(kinds).toEqual(["select", "prepare", "answer"]);
    const answerCall = first(runtimes).calls.find((call) => call.kind === "answer");
    expect(answerCall && answerCall.kind === "answer" && answerCall.reply).toBe("It costs 14 euros [S1].");
    expect(generate).toHaveBeenCalledWith([{ role: "user", content: "q" }], expect.objectContaining({ max_tokens: 64 }));
    expect(result.timings.generation_ms).toBe(4200);
    expect(result.usage.completion_tokens).toBe(9);
    expect(result.usage.total_tokens).toBe(309);
  });

  it("does not call the page model when the application needs none", async () => {
    const { api } = makeApi((request) => (request.kind === "prepare" ? { generate: false } : server(request)));
    await api.selectGenerator("webllm");
    const generate = vi.spyOn(api.pageModel, "generate");
    await api.query({ question: "Anything?" });
    expect(generate).not.toHaveBeenCalled();
  });

  it("turns an error from the application into an ApiError", async () => {
    const conflict = { error: { code: "context_changed", message: "The documents changed.", details: {} } };
    const { api } = makeApi((request) => (request.kind === "answer" ? wire(409, conflict) : server(request)));
    await api.selectGenerator("webllm");
    vi.spyOn(api.pageModel, "generate").mockResolvedValue({ text: "x", promptTokens: null, completionTokens: null, ms: 1 });
    const error = await api.query({ question: "How much?" }).catch((caught: unknown) => caught);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).code).toBe("context_changed");
  });

  it("starts a fresh runtime on reset", async () => {
    const { api, runtimes } = makeApi(server);
    await api.reset();
    expect(runtimes).toHaveLength(2);
  });

  it("does not offer evaluation runs", () => {
    const { api } = makeApi(server);
    expect(() => api.startEvaluationRun()).toThrow(ApiError);
  });
});
