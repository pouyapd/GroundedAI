/// <reference lib="webworker" />
/**
 * The demo's backend: GroundedAI's FastAPI application running in Pyodide inside this worker.
 * Python packages come from the pinned Pyodide release on jsDelivr and from wheels served
 * next to the page (scripts/build_browser_runtime.py); nothing is installed from PyPI here.
 */
import bootstrapSource from "./bootstrap.py?raw";
import type { PreparedQuery, RuntimeManifest, WireResponse, WorkerMessage, WorkerRequest } from "./protocol";

interface PyodideInterface {
  loadPackage(names: string[], options?: { messageCallback?: (message: string) => void }): Promise<unknown>;
  runPythonAsync(code: string): Promise<unknown>;
  globals: { set(name: string, value: unknown): void };
  toPy(value: unknown): unknown;
}

interface Handlers {
  fetch(method: string, path: string, headersJson: string, body?: Uint8Array): Promise<WireResponse>;
  prepare(body: string): Promise<PreparedQuery>;
  answer(body: string, reply?: string): Promise<WireResponse>;
  select(generator: string, model?: string): void;
}

const scope = self as unknown as DedicatedWorkerGlobalScope;
let handlers: Promise<Handlers> | null = null;

function post(message: WorkerMessage, transfer: Transferable[] = []) {
  scope.postMessage(message, transfer);
}

async function start(runtimeUrl: string): Promise<Handlers> {
  post({ kind: "progress", stage: "Downloading the Python runtime (Pyodide)" });
  const manifestResponse = await fetch(`${runtimeUrl}manifest.json`, { cache: "no-cache" });
  if (!manifestResponse.ok) throw new Error(`The demo runtime is missing (${manifestResponse.status}).`);
  const manifest = (await manifestResponse.json()) as RuntimeManifest;
  const { loadPyodide } = (await import(/* @vite-ignore */ `${manifest.pyodide.index_url}pyodide.mjs`)) as {
    loadPyodide: (options: { indexURL: string }) => Promise<PyodideInterface>;
  };
  const pyodide = await loadPyodide({ indexURL: manifest.pyodide.index_url });
  post({ kind: "progress", stage: "Loading Python packages" });
  await pyodide.loadPackage(manifest.packages, { messageCallback: () => undefined });
  post({ kind: "progress", stage: "Installing GroundedAI" });
  const wheelUrls = manifest.wheels.map((file) => new URL(`${runtimeUrl}wheels/${file}`, scope.location.href).href);
  pyodide.globals.set("wheel_urls", pyodide.toPy(wheelUrls));
  await pyodide.runPythonAsync("import micropip\nawait micropip.install(wheel_urls, deps=False)");
  post({ kind: "progress", stage: "Starting the application (database and migrations)" });
  return (await pyodide.runPythonAsync(bootstrapSource)) as Handlers;
}

/** A plain copy of a response from Python, whose body can be transferred without touching
 * Pyodide's memory. */
function wire(response: WireResponse): { result: WireResponse; transfer: Transferable[] } {
  const body = new Uint8Array(response.body);
  return {
    result: { status: response.status, headers: response.headers.map(([k, v]) => [k, v]), body },
    transfer: [body.buffer],
  };
}

async function handle(request: WorkerRequest): Promise<{ result: unknown; transfer: Transferable[] }> {
  if (request.kind === "init") {
    handlers ??= start(request.runtimeUrl);
    await handlers;
    return { result: true, transfer: [] };
  }
  if (!handlers) throw new Error("The demo runtime has not been started.");
  const api = await handlers;
  if (request.kind === "fetch") {
    const headers = JSON.stringify(Object.fromEntries(request.headers));
    return wire(await api.fetch(request.method, request.path, headers, request.body));
  }
  if (request.kind === "prepare") return { result: await api.prepare(request.body), transfer: [] };
  if (request.kind === "answer") {
    return wire(await api.answer(request.body, request.reply));
  }
  api.select(request.generator, request.model);
  return { result: true, transfer: [] };
}

scope.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  handle(request).then(
    ({ result, transfer }) => post({ kind: "result", id: request.id, result }, transfer),
    (error: unknown) => {
      if (request.kind === "init") handlers = null;
      post({ kind: "failure", id: request.id, error: error instanceof Error ? error.message : String(error) });
    },
  );
};
