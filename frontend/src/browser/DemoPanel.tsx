import { useEffect, useState } from "react";

import { useApp } from "../context";
import type { BrowserApi, Generator, RuntimeState } from "./BrowserApi";
import { checkWebGpu, PAGE_MODEL, type GpuSupport } from "./gpu";

/** The live demo's controls: runtime start-up, generator choice and reset. */
export function DemoPanel({ api }: { api: BrowserApi }) {
  const { refreshStatus, resetDemo } = useApp();
  const [runtime, setRuntime] = useState<RuntimeState>({ phase: "starting", stage: "Starting the demo" });
  const [gpu, setGpu] = useState<GpuSupport | null>(null);
  const [generator, setGenerator] = useState<Generator>(api.generator);
  const [modelProgress, setModelProgress] = useState<{ fraction: number; text: string } | null>(null);
  const [modelError, setModelError] = useState<string | null>(null);
  const [resetting, setResetting] = useState(false);

  useEffect(() => api.subscribe(setRuntime), [api]);
  useEffect(() => {
    let cancelled = false;
    void checkWebGpu().then((support) => {
      if (!cancelled) setGpu(support);
    });
    return () => {
      cancelled = true;
    };
  }, []);

  async function choose(next: Generator) {
    setModelError(null);
    if (next === "webllm" && !api.pageModel.loaded) {
      setModelProgress({ fraction: 0, text: "Preparing the download" });
      try {
        await api.pageModel.load((fraction, text) => setModelProgress({ fraction, text }));
      } catch (error) {
        setModelProgress(null);
        setModelError(error instanceof Error ? error.message : String(error));
        return;
      }
      setModelProgress(null);
    }
    await api.selectGenerator(next);
    setGenerator(next);
    refreshStatus();
  }

  async function reset() {
    setResetting(true);
    try {
      await api.reset();
    } finally {
      setResetting(false);
      resetDemo();
      refreshStatus();
    }
  }

  const loadingModel = modelProgress !== null;
  const starting = runtime.phase === "starting";

  return (
    <section className="demo-panel" aria-labelledby="demo-title">
      <div className="demo-head">
        <div>
          <p className="eyebrow">Live demo</p>
          <h1 id="demo-title" className="demo-title">
            Ask a PDF. Every statement comes back with its quote and page.
          </h1>
          <p className="demo-lead">
            This page runs the complete GroundedAI backend in your browser: upload checks, PDF extraction, hybrid
            retrieval, the prompt and the claim check, as Python compiled to WebAssembly. Two sample PDFs are
            loaded. Your questions and any file you upload stay in this tab.
          </p>
        </div>
        <button type="button" className="btn btn-small" onClick={() => void reset()} disabled={starting || resetting}>
          {resetting ? <span className="spinner" /> : null} Reset demo
        </button>
      </div>

      {starting ? (
        <div className="demo-status" role="status" aria-live="polite">
          <span className="spinner" /> {runtime.stage}… <span className="muted">(first visit: about 15–40 s)</span>
        </div>
      ) : null}
      {runtime.phase === "failed" ? (
        <div className="demo-status demo-status-error" role="alert">
          The demo could not start: {runtime.error}. It needs a current browser with WebAssembly and access to
          cdn.jsdelivr.net.{" "}
          <button type="button" className="btn-link" onClick={() => void reset()}>
            Try again
          </button>
        </div>
      ) : null}

      <fieldset className="generator-choice" disabled={starting || loadingModel}>
        <legend>Answer generator</legend>
        <label className={`option-card${generator === "extractive" ? " is-selected" : ""}`}>
          <input
            type="radio"
            name="generator"
            checked={generator === "extractive"}
            onChange={() => void choose("extractive")}
          />
          <span>
            <strong>Extractive generator</strong>
            <span className="option-meta">Instant · deterministic · not a language model</span>
            <span className="option-text">
              Copies the best-matching sentences from the passages. Good for literal questions; it refuses
              paraphrased ones.
            </span>
          </span>
        </label>
        <label
          className={`option-card${generator === "webllm" ? " is-selected" : ""}${gpu && !gpu.available ? " is-disabled" : ""}`}
        >
          <input
            type="radio"
            name="generator"
            checked={generator === "webllm"}
            disabled={!gpu?.available}
            onChange={() => void choose("webllm")}
          />
          <span>
            <strong>{PAGE_MODEL.label} on your GPU</strong>
            <span className="option-meta">
              WebGPU · one-time download of about {PAGE_MODEL.downloadMB} MB, then cached
            </span>
            <span className="option-text">
              A real language model, run in this tab with WebLLM. Its answers go through the same claim check.
              {gpu && !gpu.available ? ` Not available here: ${gpu.reason}` : ""}
            </span>
          </span>
        </label>
      </fieldset>

      {modelProgress ? (
        <div className="model-progress" role="status" aria-live="polite">
          <div className="progress">
            <div className="progress-bar" style={{ width: `${Math.round(modelProgress.fraction * 100)}%` }} />
          </div>
          <span className="small secondary">
            {Math.round(modelProgress.fraction * 100)}% · {modelProgress.text}
          </span>
        </div>
      ) : null}
      {modelError ? (
        <div className="demo-status demo-status-error" role="alert">
          The language model could not be loaded: {modelError} The extractive generator still works.
        </div>
      ) : null}
    </section>
  );
}
