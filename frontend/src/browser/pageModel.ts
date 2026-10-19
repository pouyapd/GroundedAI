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

export class PageModel {
  private engine: Promise<Engine> | null = null;

  /** Downloads (or reads from the browser cache) and compiles the model. */
  load(onProgress: (fraction: number, text: string) => void): Promise<Engine> {
    this.engine ??= (async () => {
      const webllm = await import("@mlc-ai/web-llm");
      const worker = new Worker(new URL("./llmWorker.ts", import.meta.url), { type: "module" });
      return (await webllm.CreateWebWorkerMLCEngine(worker, PAGE_MODEL.id, {
        initProgressCallback: (report) => onProgress(report.progress, report.text),
      })) as unknown as Engine;
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
