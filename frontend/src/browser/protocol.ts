/** Messages between the page and the Web Worker that runs GroundedAI in Pyodide. */

export interface RuntimeManifest {
  pyodide: { version: string; index_url: string; python: string };
  packages: string[];
  wheels: string[];
  samples: string[];
}

export type ChatMessage = { role: "system" | "user" | "assistant"; content: string };

export interface PreparedQuery {
  generate: boolean;
  messages?: ChatMessage[];
  temperature?: number;
  max_tokens?: number;
  seed?: number;
}

export interface WireResponse {
  status: number;
  headers: [string, string][];
  body: Uint8Array;
}

export type WorkerRequest =
  | { id: number; kind: "init"; runtimeUrl: string }
  | {
      id: number;
      kind: "fetch";
      method: string;
      path: string;
      headers: [string, string][];
      body?: Uint8Array;
    }
  | { id: number; kind: "prepare"; body: string }
  | { id: number; kind: "answer"; body: string; reply?: string }
  | { id: number; kind: "select"; generator: "extractive" | "webllm"; model?: string };

export type WorkerMessage =
  | { kind: "progress"; stage: string }
  | { kind: "result"; id: number; result: unknown }
  | { kind: "failure"; id: number; error: string };
