import { describe, expect, it, vi } from "vitest";

import { isInterruptedDownload, PageModel, type EngineFactory } from "../browser/pageModel";

type Engine = Awaited<ReturnType<EngineFactory>>;

const engine = {
  chat: {
    completions: {
      create: vi.fn().mockResolvedValue({
        choices: [{ message: { content: "It costs 14 euros [S1]." } }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      }),
    },
  },
  unload: vi.fn(),
} as unknown as Engine;

const cacheError = new Error("Failed to execute 'add' on 'Cache': Cache.add() encountered a network error");
cacheError.name = "NetworkError";

describe("PageModel", () => {
  it("resumes a download that broke off", async () => {
    const factory = vi.fn<EngineFactory>().mockRejectedValueOnce(cacheError).mockResolvedValueOnce(engine);
    const progress = vi.fn();
    const model = new PageModel(factory);
    await model.load(progress);
    expect(factory).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenCalledWith(0, expect.stringContaining("resuming (attempt 2 of 3)"));
    const reply = await model.generate([{ role: "user", content: "q" }], {});
    expect(reply.text).toBe("It costs 14 euros [S1].");
  });

  it("gives up after the last attempt and can be asked again", async () => {
    const factory = vi.fn<EngineFactory>().mockRejectedValue(cacheError);
    const model = new PageModel(factory, 2);
    await expect(model.load(vi.fn())).rejects.toThrow("network error");
    expect(factory).toHaveBeenCalledTimes(2);
    await Promise.resolve();
    expect(model.loaded).toBe(false);
  });

  it("does not retry errors that a retry cannot fix", async () => {
    const factory = vi.fn<EngineFactory>().mockRejectedValue(new Error("WebGPU device was lost"));
    const model = new PageModel(factory);
    await expect(model.load(vi.fn())).rejects.toThrow("WebGPU");
    expect(factory).toHaveBeenCalledTimes(1);
  });

  it("recognises interrupted downloads", () => {
    expect(isInterruptedDownload(cacheError)).toBe(true);
    expect(isInterruptedDownload(new TypeError("Failed to fetch"))).toBe(true);
    expect(isInterruptedDownload(new Error("No WebGPU adapter is available."))).toBe(false);
  });
});
