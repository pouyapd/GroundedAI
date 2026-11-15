/// <reference lib="webworker" />
// The meaning check's model, in its own worker: ONNX Runtime Web on WebAssembly (the CPU) and
// the model's own tokenizer. Files are downloaded once from a pinned Hugging Face revision and
// kept in the browser's Cache Storage. Nothing the visitor types or uploads leaves this tab.
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-web/wasm";
import wasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

import type { NliProbabilities } from "../nli/premises";
import { NLI_MODEL, type NliRequest, type NliResponse } from "./nliProtocol";

// ONNX Runtime fetches and compiles this file itself (streamed, so the browser can keep the
// compiled code for the next visit).
ort.env.wasm.wasmPaths = { wasm: wasmUrl };
// Several threads need cross-origin isolation (SharedArrayBuffer), which GitHub Pages does
// not provide; one thread works everywhere.
ort.env.wasm.numThreads = 1;

const CACHE = "groundedai-nli-v1";
const MAX_TOKENS = 512;

let session: ort.InferenceSession | null = null;
let tokenizer: Tokenizer | null = null;
let labels: string[] = [];

function post(message: NliResponse) {
  (self as unknown as DedicatedWorkerGlobalScope).postMessage(message);
}

/** A model file from the pinned revision; kept in Cache Storage after the first download. */
async function cachedBytes(path: string, onBytes: (bytes: number) => void): Promise<ArrayBuffer> {
  const url = `https://huggingface.co/${NLI_MODEL.repo}/resolve/${NLI_MODEL.revision}/${path}`;
  const cache = "caches" in self ? await caches.open(CACHE) : null;
  const hit = cache ? await cache.match(url) : undefined;
  if (hit) {
    const bytes = await hit.arrayBuffer();
    onBytes(bytes.byteLength);
    return bytes;
  }
  const bytes = await download(url, path, onBytes);
  if (cache) {
    await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  }
  return bytes;
}

/** A file with progress. Without `keep`, the bytes are only counted (they stay in the HTTP cache). */
async function download(url: string, name: string, onBytes: (bytes: number) => void, keep = true): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok || !response.body) throw new Error(`Download failed (${response.status}) for ${name}`);
  const reader = response.body.getReader();
  const parts: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (keep) parts.push(value);
    received += value.byteLength;
    onBytes(received);
  }
  if (!keep) return new ArrayBuffer(0);
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const part of parts) {
    bytes.set(part, offset);
    offset += part.byteLength;
  }
  return bytes.buffer;
}

async function load(id: number) {
  const started = performance.now();
  const totals = [...NLI_MODEL.files, "runtime"].map(() => 0);
  const expected = NLI_MODEL.bytes + NLI_MODEL.runtimeBytes;
  const report = (index: number, bytes: number) => {
    totals[index] = bytes;
    const done = totals.reduce((a, b) => a + b, 0);
    post({ id, kind: "progress", fraction: Math.min(1, done / expected), bytes: done });
  };
  // The runtime file is fetched here only to show its download in the progress; ONNX Runtime
  // then reads it from the HTTP cache.
  const runtime = download(wasmUrl, "the ONNX Runtime WebAssembly file", (bytes) => report(NLI_MODEL.files.length, bytes), false);
  runtime.catch(() => undefined); // awaited below; this only avoids an unhandled rejection
  const [config, tokenizerJson, tokenizerConfig, model] = await Promise.all(
    NLI_MODEL.files.map((path, index) => cachedBytes(path, (bytes) => report(index, bytes))),
  );
  await runtime;
  const decode = (bytes: ArrayBuffer | undefined) => JSON.parse(new TextDecoder().decode(bytes)) as Record<string, unknown>;
  const id2label = decode(config).id2label as Record<string, string>;
  labels = Object.keys(id2label)
    .sort((a, b) => Number(a) - Number(b))
    .map((key) => String(id2label[key]).toLowerCase());
  tokenizer = new Tokenizer(decode(tokenizerJson), decode(tokenizerConfig));
  session = await ort.InferenceSession.create(new Uint8Array(model as ArrayBuffer), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  post({
    id,
    kind: "loaded",
    backend: "wasm",
    ms: performance.now() - started,
    bytes: totals.reduce((a, b) => a + b, 0),
  });
}

function tensor(values: number[]): ort.Tensor {
  return new ort.Tensor("int64", BigInt64Array.from(values.map((value) => BigInt(value))), [1, values.length]);
}

async function classify(id: number, pairs: [string, string][]) {
  if (!session || !tokenizer) throw new Error("The meaning check is not loaded.");
  const started = performance.now();
  const out: NliProbabilities[] = [];
  for (const [premise, hypothesis] of pairs) {
    const encoded = tokenizer.encode(premise, { text_pair: hypothesis, return_token_type_ids: true });
    const keep = Math.min(encoded.ids.length, MAX_TOKENS);
    const feeds: Record<string, ort.Tensor> = {
      input_ids: tensor(encoded.ids.slice(0, keep)),
      attention_mask: tensor(encoded.attention_mask.slice(0, keep)),
    };
    if (session.inputNames.includes("token_type_ids")) {
      feeds.token_type_ids = tensor(encoded.token_type_ids.slice(0, keep));
    }
    const result = await session.run(feeds);
    const logits = Array.from((result[session.outputNames[0] as string] as ort.Tensor).data as Float32Array);
    const top = Math.max(...logits);
    const exp = logits.map((value) => Math.exp(value - top));
    const sum = exp.reduce((a, b) => a + b, 0);
    const probabilities = { entailment: 0, neutral: 0, contradiction: 0 } as NliProbabilities;
    labels.forEach((label, index) => {
      if (label === "entailment" || label === "neutral" || label === "contradiction") {
        probabilities[label] = (exp[index] ?? 0) / sum;
      }
    });
    out.push(probabilities);
  }
  post({ id, kind: "result", probabilities: out, ms: performance.now() - started });
}

self.onmessage = (event: MessageEvent<NliRequest>) => {
  const request = event.data;
  const task = request.kind === "load" ? load(request.id) : classify(request.id, request.pairs);
  task.catch((error: unknown) =>
    post({ id: request.id, kind: "error", message: error instanceof Error ? error.message : String(error) }),
  );
};
