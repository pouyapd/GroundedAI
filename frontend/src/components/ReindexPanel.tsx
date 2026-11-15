import { useEffect, useRef, useState } from "react";

import type { Api } from "../api/client";
import type { IndexDifference, ReindexJob, ReindexStatus } from "../api/types";
import { Banner } from "./Banner";
import { ErrorBanner } from "./ErrorBanner";

const POLL_MS = 1500;

export function differenceText(difference: IndexDifference): string {
  return `${difference.setting} ${difference.indexed} → ${difference.current}`;
}

function plural(count: number, word: string): string {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

/**
 * Ready documents indexed with older settings (for example text clean-up v1 before v2 became
 * the default), and the bulk re-indexing job with its progress. Each document stays searchable
 * with its previous index until its new one is written; one whose re-indexing fails keeps it.
 */
export function ReindexPanel({
  api,
  reloadToken,
  readOnly,
  onFinished,
}: {
  api: Api;
  reloadToken: number;
  readOnly: boolean;
  onFinished: () => void;
}) {
  const [status, setStatus] = useState<ReindexStatus | null>(null);
  const [startError, setStartError] = useState<unknown>(null);
  const [starting, setStarting] = useState(false);
  const [poll, setPoll] = useState(0);
  // The finished job is reported only if this page saw it run (not one from an earlier visit).
  const [watchedJob, setWatchedJob] = useState<string | null>(null);
  const running = status?.job?.state === "running";
  const wasRunning = useRef(false);

  useEffect(() => {
    let cancelled = false;
    api
      .reindexStatus()
      .then((result) => {
        if (cancelled) return;
        setStatus(result);
        if (result.job?.state === "running") setWatchedJob(result.job.id);
      })
      .catch(() => {
        // A server without this endpoint (older versions) simply shows no panel.
        if (!cancelled) setStatus(null);
      });
    return () => {
      cancelled = true;
    };
  }, [api, reloadToken, poll]);

  useEffect(() => {
    if (!running) return;
    const timer = window.setInterval(() => setPoll((value) => value + 1), POLL_MS);
    return () => window.clearInterval(timer);
  }, [running]);

  useEffect(() => {
    if (wasRunning.current && !running) onFinished(); // refresh the document list
    wasRunning.current = running;
  }, [running, onFinished]);

  async function start() {
    setStarting(true);
    setStartError(null);
    try {
      const result = await api.startReindex();
      if (result.job) setWatchedJob(result.job.id);
      setStatus(result);
      setPoll((value) => value + 1);
    } catch (error) {
      setStartError(error);
    } finally {
      setStarting(false);
    }
  }

  if (!status) return null;
  const job = status.job;
  const seen = job && job.id === watchedJob ? job : null;
  const reindexable = status.outdated - status.outdated_without_source;

  if (running && job) {
    const fraction = job.total > 0 ? job.done / job.total : 0;
    return (
      <Banner tone="info" title={`Re-indexing: ${job.done} of ${plural(job.total, "document")} done`}>
        <div className="progress" style={{ margin: "6px 0" }} aria-hidden="true">
          <div className="progress-bar" style={{ width: `${Math.round(fraction * 100)}%` }} />
        </div>
        {job.processing_title ? <p style={{ margin: 0 }}>Now: {job.processing_title}.</p> : null}
        <p style={{ margin: 0 }}>
          Each document stays searchable with its previous index until its new one is ready.
        </p>
      </Banner>
    );
  }

  if (status.outdated === 0) {
    if (!seen) return null;
    return (
      <Banner tone="info" title="All documents are indexed with the current settings">
        <JobSummary job={seen} />
      </Banner>
    );
  }

  return (
    <>
      <Banner
        tone="warning"
        title={`${plural(status.outdated, "document")} indexed with older settings`}
        action={
          !readOnly && reindexable > 0 ? (
            <button type="button" className="btn btn-small btn-primary" onClick={() => void start()} disabled={starting}>
              {starting ? <span className="spinner" /> : null} Re-index {plural(reindexable, "document")}
            </button>
          ) : null
        }
      >
        <ul style={{ margin: "4px 0", paddingLeft: 18 }}>
          {status.differences.map((difference) => (
            <li key={differenceText(difference)}>
              {differenceText(difference)} ({plural(difference.documents, "document")})
            </li>
          ))}
        </ul>
        <p style={{ margin: 0 }}>
          Their passages were prepared with those settings. Re-indexing rebuilds them with the current ones; each
          document stays searchable meanwhile, and keeps its previous index if re-indexing it fails.
        </p>
        {status.outdated_without_source > 0 ? (
          <p style={{ margin: "4px 0 0" }}>
            {plural(status.outdated_without_source, "document")} cannot be re-indexed because the original file was not
            kept; upload {status.outdated_without_source === 1 ? "it" : "them"} again.
          </p>
        ) : null}
        {seen ? <JobSummary job={seen} /> : null}
      </Banner>
      {startError ? <ErrorBanner error={startError} /> : null}
    </>
  );
}

function JobSummary({ job }: { job: ReindexJob }) {
  return (
    <div style={{ marginTop: 4 }}>
      <p style={{ margin: 0 }}>
        Last run: {job.reindexed} re-indexed
        {job.already_current > 0 ? `, ${job.already_current} already current` : ""}
        {job.skipped.length > 0 ? `, ${job.skipped.length} skipped` : ""}
        {job.failed.length > 0 ? `, ${job.failed.length} failed (they keep their previous index)` : ""}.
      </p>
      {[...job.failed, ...job.skipped].length > 0 ? (
        <ul style={{ margin: "4px 0 0", paddingLeft: 18 }}>
          {[...job.failed, ...job.skipped].map((outcome) => (
            <li key={outcome.document_id}>
              {outcome.title || outcome.document_id}: {outcome.detail}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
