/**
 * The optional language model of the online demo: Qwen2.5-1.5B-Instruct (4-bit, MLC build)
 * running on the visitor's GPU through WebLLM and WebGPU. The weights are downloaded once from
 * Hugging Face and cached by the browser. GroundedAI's prompt and checks stay in Python: the
 * model only turns the prompt messages into a reply.
 */
import { PAGE_MODEL } from "./gpu";
import type { ChatMessage } from "./protocol";

export interface GenerationResult {
  text: string;
  promptTokens: number | null;
  completionTokens: number | null;
  ms: number;
}

interface Engine {
  chat: {
    completions: {
      create(request: Record<string, unknown>): Promise<{
        choices: { message: { content: string | null } }[];
        usage?: { prompt_tokens?: number; completion_tokens?: number };
      }>;
    };
  };
  unload(): Promise<void>;
}

export type ProgressCallback = (fraction: number, text: string) => void;
export type EngineFactory = (onProgress: ProgressCallback) => Promise<Engine>;

/** WebLLM in its own worker; a failed attempt stops its worker. */
async function createWorkerEngine(onProgress: ProgressCallback): Promise<Engine> {
  const webllm = await import("@mlc-ai/web-llm");
  const worker = new Worker(new URL("./llmWorker.ts", import.meta.url), { type: "module" });
  try {
    return (await webllm.CreateWebWorkerMLCEngine(worker, PAGE_MODEL.id, {
      initProgressCallback: (report) => onProgress(report.progress, report.text),
    })) as unknown as Engine;
  } catch (error) {
    worker.terminate();
    throw error;
  }
}

/** A download that broke off (slow or unstable connection), as opposed to e.g. missing WebGPU. */
export function isInterruptedDownload(error: unknown): boolean {
  const text = error instanceof Error ? `${error.name} ${error.message}` : String(error);
  return /network|fetch|Cache\.add|aborted|timed? ?out/i.test(text);
}

export class PageModel {
  private engine: Promise<Engine> | null = null;

  constructor(
    private readonly createEngine: EngineFactory = createWorkerEngine,
    private readonly attempts = 3,
  ) {}

  /**
   * Downloads (or reads from the browser cache) and compiles the model. A download that breaks
   * off is resumed: weight files already stored in the browser cache are not fetched again.
   */
  load(onProgress: ProgressCallback): Promise<Engine> {
    this.engine ??= (async () => {
      for (let attempt = 1; ; attempt += 1) {
        try {
          return await this.createEngine(onProgress);
        } catch (error) {
          if (attempt >= this.attempts || !isInterruptedDownload(error)) throw error;
          onProgress(0, `The download was interrupted; resuming (attempt ${attempt + 1} of ${this.attempts})`);
        }
      }
    })();
    this.engine.catch(() => {
      this.engine = null;
    });
    return this.engine;
  }

  get loaded(): boolean {
    return this.engine !== null;
  }

  async generate(
    messages: ChatMessage[],
    options: { temperature?: number; max_tokens?: number; seed?: number },
  ): Promise<GenerationResult> {
    if (!this.engine) throw new Error("The language model is not loaded.");
    const engine = await this.engine;
    const started = performance.now();
    const reply = await engine.chat.completions.create({
      messages,
      temperature: options.temperature ?? 0,
      max_tokens: options.max_tokens ?? 384,
      seed: options.seed ?? 0,
      stream: false,
    });
    return {
      text: reply.choices[0]?.message.content ?? "",
      promptTokens: reply.usage?.prompt_tokens ?? null,
      completionTokens: reply.usage?.completion_tokens ?? null,
      ms: performance.now() - started,
    };
  }
}
