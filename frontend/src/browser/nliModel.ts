/**
 * The optional meaning check of the online demo (main-thread side). The model runs in
 * nliWorker.ts on WebAssembly; this class starts the worker on first use and passes messages.
 */
import type { NliProbabilities } from "../nli/premises";
import { NLI_MODEL, type NliRequest, type NliResponse } from "./nliProtocol";

export type NliProgress = (fraction: number, text: string) => void;
export type Classify = (pairs: [string, string][]) => Promise<{ probabilities: NliProbabilities[]; ms: number }>;

export class NliModel {
  private worker: Worker | null = null;
  private loading: Promise<void> | null = null;
  private nextId = 1;
  private waiting = new Map<number, { resolve: (value: NliResponse) => void; reject: (error: Error) => void; progress?: NliProgress }>();
  loaded = false;
  loadMs: number | null = null;
  downloadedBytes: number | null = null;
  readonly model = NLI_MODEL.label;
  readonly backend = "wasm" as const;

  constructor(private readonly createWorker: () => Worker = defaultWorker) {}

  private send(request: NliRequest, progress?: NliProgress): Promise<NliResponse> {
    if (!this.worker) {
      this.worker = this.createWorker();
      this.worker.onmessage = (event: MessageEvent<NliResponse>) => this.receive(event.data);
      this.worker.onerror = (event) => this.failAll(new Error(event.message || "The meaning check stopped."));
    }
    return new Promise((resolve, reject) => {
      this.waiting.set(request.id, { resolve, reject, progress });
      this.worker?.postMessage(request);
    });
  }

  private receive(message: NliResponse) {
    const entry = this.waiting.get(message.id);
    if (!entry) return;
    if (message.kind === "progress") {
      const mb = (message.bytes / 1e6).toFixed(0);
      entry.progress?.(message.fraction, `${mb} of ${NLI_MODEL.downloadMB} MB`);
      return;
    }
    this.waiting.delete(message.id);
    if (message.kind === "error") entry.reject(new Error(message.message));
    else entry.resolve(message);
  }

  private failAll(error: Error) {
    for (const entry of this.waiting.values()) entry.reject(error);
    this.waiting.clear();
    this.worker?.terminate();
    this.worker = null;
    this.loading = null;
    this.loaded = false;
  }

  /** Download (once; then from the browser cache) and start the model. */
  load(progress: NliProgress): Promise<void> {
    if (this.loaded) return Promise.resolve();
    if (!this.loading) {
      this.loading = this.send({ id: this.nextId++, kind: "load" }, progress)
        .then((message) => {
          if (message.kind === "loaded") {
            this.loaded = true;
            this.loadMs = message.ms;
            this.downloadedBytes = message.bytes;
          }
        })
        .catch((error: unknown) => {
          this.failAll(error instanceof Error ? error : new Error(String(error)));
          throw error;
        });
    }
    return this.loading;
  }

  /** Stop the worker; this frees the model's memory. Loading again reads the cached files. */
  unload() {
    this.failAll(new Error("The meaning check was switched off."));
    this.loadMs = null;
    this.downloadedBytes = null;
  }

  classify: Classify = async (pairs) => {
    if (!this.loaded) throw new Error("The meaning check is not loaded.");
    const message = await this.send({ id: this.nextId++, kind: "classify", pairs });
    if (message.kind !== "result") throw new Error("Unexpected reply from the meaning check.");
    return { probabilities: message.probabilities, ms: message.ms };
  };
}

/** What the page says when the meaning check could not be loaded. */
export function meaningLoadErrorMessage(error: unknown): string {
  const detail = (error instanceof Error ? error.message : String(error)).replace(/[.\s]+$/, "");
  return `The meaning check could not be loaded (${detail}). Answers are still checked by word matching. Switch it on again to retry.`;
}

function defaultWorker(): Worker {
  return new Worker(new URL("./nliWorker.ts", import.meta.url), { type: "module" });
}
