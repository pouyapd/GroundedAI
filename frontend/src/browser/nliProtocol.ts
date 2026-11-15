import type { NliProbabilities } from "../nli/premises";

/**
 * The meaning check's model: cross-encoder/nli-deberta-v3-xsmall (Apache-2.0), as converted to
 * ONNX and quantized to int8 by the transformers.js project (Xenova/nli-deberta-v3-xsmall),
 * pinned to one revision. On the 108 hand-written test claims it was scored in
 * docs/RESEARCH_EVALUATION.md; on the validation claims it matched the full PyTorch model.
 */
export const NLI_MODEL = {
  repo: "Xenova/nli-deberta-v3-xsmall",
  revision: "2a4f614a701367a02d51389039afc998faeda637",
  label: "DeBERTa-v3-xsmall NLI (int8 ONNX)",
  files: ["config.json", "tokenizer.json", "tokenizer_config.json", "onnx/model_quantized.onnx"],
  bytes: 1038 + 8656551 + 384 + 87246587,
  // ONNX Runtime Web 1.31.1's ort-wasm-simd-threaded.wasm, served with the demo.
  runtimeBytes: 14848852,
  // Model files and runtime together, in megabytes (10^6 bytes), before any compression.
  downloadMB: 111,
} as const;

export type NliRequest = { id: number; kind: "load" } | { id: number; kind: "classify"; pairs: [string, string][] };

export type NliResponse =
  | { id: number; kind: "progress"; fraction: number; bytes: number }
  | { id: number; kind: "loaded"; backend: "wasm"; ms: number; bytes: number }
  | { id: number; kind: "result"; probabilities: NliProbabilities[]; ms: number }
  | { id: number; kind: "error"; message: string };
