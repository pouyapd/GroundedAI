/**
 * The API of the online demo. Every request goes to GroundedAI's own FastAPI application,
 * which runs in this browser tab (Pyodide in a Web Worker) instead of on a server. The
 * evaluation dashboards show the recorded reports bundled with the static demo.
 */
import { ApiError, LiveApi, toApiError, type Api } from "../api/client";
import { StaticDemoApi } from "../api/demo";
import type { AnswerResponse, DocumentOut, QueryRequest } from "../api/types";
import { PAGE_MODEL } from "./gpu";
import { PageModel } from "./pageModel";
import type { PreparedQuery, WireResponse, WorkerMessage, WorkerRequest } from "./protocol";

export type Generator = "extractive" | "webllm";
export type RequestBody = WorkerRequest extends infer R ? (R extends { id: number } ? Omit<R, "id"> : never) : never;

export interface RuntimeState {
  phase: "starting" | "ready" | "failed";
  stage: string;
  error?: string;
}

/** What BrowserApi needs from the worker (tests replace it). */
export interface RuntimeLike {
  readonly ready: Promise<void>;
  call<T>(request: RequestBody, transfer?: Transferable[]): Promise<T>;
  terminate(): void;
}

export type RuntimeFactory = (runtimeUrl: string, onStage: (stage: string) => void) => RuntimeLike;

/** The worker that runs the Python application, with request/response matching. */
class Runtime implements RuntimeLike {
  private readonly worker: Worker;
  private next = 1;
  private readonly pending = new Map<number, { resolve: (value: unknown) => void; reject: (error: Error) => void }>();
  readonly ready: Promise<void>;

  constructor(runtimeUrl: string, onStage: (stage: string) => void) {
    this.worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
    this.worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
      const message = event.data;
      if (message.kind === "progress") return onStage(message.stage);
      const waiting = this.pending.get(message.id);
      if (!waiting) return;
      this.pending.delete(message.id);
      if (message.kind === "result") waiting.resolve(message.result);
      else waiting.reject(new Error(message.error));
    };
    this.worker.onerror = (event) => {
      const error = new Error(event.message || "The demo runtime stopped.");
      for (const waiting of this.pending.values()) waiting.reject(error);
      this.pending.clear();
    };
    this.ready = this.call({ kind: "init", runtimeUrl }).then(() => undefined);
  }

  call<T>(request: RequestBody, transfer: Transferable[] = []): Promise<T> {
    const id = this.next++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (value: unknown) => void, reject });
      this.worker.postMessage({ ...request, id }, transfer);
    });
  }

  terminate() {
    this.worker.terminate();
    for (const waiting of this.pending.values()) waiting.reject(new Error("The demo was reset."));
    this.pending.clear();
  }
}

function toResponse(wire: WireResponse): Response {
  const empty = [101, 204, 205, 304].includes(wire.status);
  return new Response(empty ? null : (wire.body as BodyInit), { status: wire.status, headers: wire.headers });
}

export class BrowserApi implements Api {
  readonly mode = "browser" as const;
  readonly pageModel = new PageModel();
  private runtime: RuntimeLike;
  private live: LiveApi;
  private readonly recorded: StaticDemoApi;
  private readonly runtimeUrl: string;
  private seeded: Promise<void>;
  private listeners = new Set<(state: RuntimeState) => void>();
  private state: RuntimeState = { phase: "starting", stage: "Starting" };
  generator: Generator = "extractive";

  constructor(
    base = import.meta.env.BASE_URL,
    private readonly createRuntime: RuntimeFactory = (url, onStage) => new Runtime(url, onStage),
  ) {
    this.runtimeUrl = new URL(`${base}runtime/`, window.location.href).href;
    this.recorded = new StaticDemoApi(`${base}demo`);
    this.runtime = this.boot();
    this.live = this.liveApi();
    this.seeded = this.seed();
    this.seeded.catch(() => undefined); // reported through the runtime state; API calls rethrow
  }

  /** Resolves when the runtime has started and the sample PDFs are indexed. */
  whenReady(): Promise<void> {
    return this.seeded;
  }

  // ------------------------------------------------------------------ runtime state
  subscribe(listener: (state: RuntimeState) => void): () => void {
    this.listeners.add(listener);
    listener(this.state);
    return () => this.listeners.delete(listener);
  }

  private setState(state: RuntimeState) {
    this.state = state;
    for (const listener of this.listeners) listener(state);
  }

  private boot(): RuntimeLike {
    this.setState({ phase: "starting", stage: "Starting the demo" });
    return this.createRuntime(this.runtimeUrl, (stage) => this.setState({ phase: "starting", stage }));
  }

  private liveApi(): LiveApi {
    const runtime = this.runtime;
    return new LiveApi("", async (input, init) => {
      await runtime.ready;
      const request = new Request(new URL(String(input), "http://groundedai.browser"), init);
      const url = new URL(request.url);
      const hasBody = !["GET", "HEAD"].includes(request.method);
      const body = hasBody ? new Uint8Array(await request.arrayBuffer()) : undefined;
      const wire = await runtime.call<WireResponse>(
        { kind: "fetch", method: request.method, path: url.pathname + url.search, headers: [...request.headers], body },
        body ? [body.buffer] : [],
      );
      return toResponse(wire);
    });
  }

  /** Loads the two sample PDFs through the normal upload endpoint, once per runtime. */
  private async seed(): Promise<void> {
    try {
      await this.runtime.ready;
      this.setState({ phase: "starting", stage: "Indexing the sample PDFs" });
      const manifest = (await (await fetch(`${this.runtimeUrl}manifest.json`)).json()) as { samples: string[] };
      for (const name of manifest.samples) {
        const blob = await (await fetch(`${this.runtimeUrl}samples/${name}`)).blob();
        await this.live.uploadDocument(new File([blob], name, { type: "application/pdf" }));
      }
      if (this.generator === "webllm") await this.runtime.call({ kind: "select", generator: "webllm", model: PAGE_MODEL.id });
      this.setState({ phase: "ready", stage: "Ready" });
    } catch (error) {
      this.setState({ phase: "failed", stage: "", error: error instanceof Error ? error.message : String(error) });
      throw error;
    }
  }

  /** Starts again: a new runtime with only the sample documents. The model stays loaded. */
  reset(): Promise<void> {
    this.runtime.terminate();
    this.runtime = this.boot();
    this.live = this.liveApi();
    this.seeded = this.seed();
    this.seeded.catch(() => undefined);
    return this.seeded;
  }

  async selectGenerator(generator: Generator): Promise<void> {
    await this.seeded;
    await this.runtime.call({ kind: "select", generator, model: PAGE_MODEL.id });
    this.generator = generator;
  }

  private async ready(): Promise<LiveApi> {
    await this.seeded;
    return this.live;
  }

  // ----------------------------------------------------------------------- the API
  async systemStatus() {
    return (await this.ready()).systemStatus();
  }
  async listDocuments(offset: number, limit: number) {
    return (await this.ready()).listDocuments(offset, limit);
  }
  async listChunks(documentId: string) {
    return (await this.ready()).listChunks(documentId);
  }
  async uploadDocument(file: File, title?: string): Promise<DocumentOut> {
    return (await this.ready()).uploadDocument(file, title);
  }
  async deleteDocument(documentId: string) {
    return (await this.ready()).deleteDocument(documentId);
  }
  async reingestDocument(documentId: string) {
    return (await this.ready()).reingestDocument(documentId);
  }
  async metrics() {
    return (await this.ready()).metrics();
  }

  async query(request: QueryRequest): Promise<AnswerResponse> {
    const live = await this.ready();
    if (this.generator !== "webllm") return live.query(request);
    const started = performance.now();
    const body = JSON.stringify(request);
    const runtime = this.runtime;
    const prepared = await runtime.call<PreparedQuery>({ kind: "prepare", body });
    let generation: Awaited<ReturnType<PageModel["generate"]>> | null = null;
    if (prepared.generate && prepared.messages) {
      generation = await this.pageModel.generate(prepared.messages, prepared);
    }
    const response = toResponse(await runtime.call<WireResponse>({ kind: "answer", body, reply: generation?.text }));
    if (!response.ok) throw await toApiError(response);
    const answer = (await response.json()) as AnswerResponse;
    if (generation) {
      // The pipeline replayed the reply; report the time and tokens the page model really took.
      answer.timings = {
        ...answer.timings,
        generation_ms: generation.ms,
        total_ms: performance.now() - started,
      };
      answer.usage = {
        ...answer.usage,
        prompt_tokens: generation.promptTokens,
        completion_tokens: generation.completionTokens,
        total_tokens:
          generation.promptTokens !== null && generation.completionTokens !== null
            ? generation.promptTokens + generation.completionTokens
            : null,
      };
    }
    return answer;
  }

  listReports() {
    return this.recorded.listReports();
  }
  getReport(reportId: string) {
    return this.recorded.getReport(reportId);
  }
  startEvaluationRun(): never {
    throw new ApiError(501, "not_available", "Evaluation runs need the full application (see the README).");
  }
  getEvaluationRun(): never {
    throw new ApiError(501, "not_available", "Evaluation runs need the full application (see the README).");
  }
}
