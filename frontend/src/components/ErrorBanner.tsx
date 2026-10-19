import { ApiError } from "../api/client";
import { Banner } from "./Banner";

const TITLES: Record<string, string> = {
  network_error: "The backend is not reachable",
  model_unavailable: "The language model is not available",
  model_timeout: "The model took too long to answer",
  server_busy: "The server is busy",
  rate_limited: "Too many requests",
  unauthorized: "API key required",
  duplicate_document: "This document is already indexed",
  demo_not_recorded: "Not part of the recorded demo",
  demo_read_only: "Read-only static demo",
};

export function ErrorBanner({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const apiError = error instanceof ApiError ? error : null;
  const code = apiError?.code ?? "unknown";
  const title = TITLES[code] ?? "Something went wrong";
  const tone = code.startsWith("demo_") ? "info" : code === "rate_limited" || code === "server_busy" ? "warning" : "critical";
  const retryAfter = apiError?.details["retry_after_s"];
  return (
    <Banner
      tone={tone}
      title={title}
      action={
        onRetry ? (
          <button type="button" className="btn btn-small" onClick={onRetry}>
            Retry
          </button>
        ) : undefined
      }
    >
      <p>{apiError?.message ?? (error instanceof Error ? error.message : "Unexpected error.")}</p>
      {typeof retryAfter === "number" ? <p>Try again in {retryAfter} s.</p> : null}
      {code === "network_error" ? (
        <p>
          Start the API with <code>groundedai serve</code> (or <code>docker compose up</code>) and reload this page.
        </p>
      ) : null}
      {apiError?.requestId ? <p className="muted">Request id: {apiError.requestId}</p> : null}
    </Banner>
  );
}
