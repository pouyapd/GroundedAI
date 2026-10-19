import { useCallback, useEffect, useRef, useState, type DragEvent, type FormEvent } from "react";

import { ApiError } from "../api/client";
import type { ChunkOut, DocumentOut, Page } from "../api/types";
import { DocumentStatusBadge } from "../components/Badge";
import { Banner, EmptyState, Loading } from "../components/Banner";
import { ErrorBanner } from "../components/ErrorBanner";
import { useApp } from "../context";
import { formatBytes, formatDate, pageRange } from "../lib/format";

const PAGE_SIZE = 20;

export function validateFile(file: File, allowed: string[], maxBytes: number): string | null {
  const extension = file.name.includes(".") ? `.${file.name.split(".").pop()?.toLowerCase()}` : "";
  if (!allowed.includes(extension)) return `Unsupported file type. Allowed: ${allowed.join(", ")}.`;
  if (file.size === 0) return "The file is empty.";
  if (file.size > maxBytes) return `The file is larger than ${formatBytes(maxBytes)}.`;
  return null;
}

export function DocumentsPage() {
  const { api, status, askAbout } = useApp();
  const readOnly = api.mode === "static-demo";
  const [page, setPage] = useState<Page<DocumentOut> | null>(null);
  const [offset, setOffset] = useState(0);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  const [uploadError, setUploadError] = useState<unknown>(null);
  const [validation, setValidation] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [openChunks, setOpenChunks] = useState<{ id: string; chunks: ChunkOut[] } | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const allowed = status?.limits.allowed_extensions ?? [".pdf", ".txt", ".md"];
  const maxBytes = status?.limits.max_upload_bytes ?? 10 * 1024 * 1024;

  const [reloadToken, setReloadToken] = useState(0);
  const load = useCallback(() => setReloadToken((token) => token + 1), []);

  useEffect(() => {
    let cancelled = false;
    api
      .listDocuments(offset, PAGE_SIZE)
      .then((result) => {
        if (cancelled) return;
        setPage(result);
        setLoadError(null);
      })
      .catch((error: unknown) => {
        if (!cancelled) setLoadError(error);
      });
    return () => {
      cancelled = true;
    };
  }, [api, offset, reloadToken]);

  const busy = page?.items.some((doc) => doc.status === "pending" || doc.status === "processing") ?? false;
  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(load, 2000);
    return () => window.clearInterval(timer);
  }, [busy, load]);

  function choose(selected: File | null) {
    setUploadError(null);
    setFile(selected);
    setValidation(selected ? validateFile(selected, allowed, maxBytes) : null);
  }

  async function onUpload(event: FormEvent) {
    event.preventDefault();
    if (!file || validation) return;
    setUploading(true);
    setUploadError(null);
    try {
      await api.uploadDocument(file, title.trim() || undefined);
      setFile(null);
      setTitle("");
      if (inputRef.current) inputRef.current.value = "";
      setOffset(0);
      load();
    } catch (error) {
      setUploadError(error);
    } finally {
      setUploading(false);
    }
  }

  async function act(action: () => Promise<unknown>) {
    try {
      await action();
      load();
    } catch (error) {
      setLoadError(error);
    }
  }

  async function toggleChunks(documentId: string) {
    if (openChunks?.id === documentId) {
      setOpenChunks(null);
      return;
    }
    try {
      const chunks = await api.listChunks(documentId);
      setOpenChunks({ id: documentId, chunks: chunks.items });
    } catch (error) {
      setLoadError(error);
    }
  }

  function onDrop(event: DragEvent) {
    event.preventDefault();
    setDragging(false);
    choose(event.dataTransfer.files[0] ?? null);
  }

  return (
    <>
      <div>
        <h1 className="page-title">Documents</h1>
        <p className="page-intro">
          Upload PDF, plain-text or Markdown files. They are parsed, split into chunks with page numbers, embedded and
          indexed for lexical and vector search.
          {api.mode === "browser"
            ? " In this demo everything happens in your browser: files are processed in this tab, are not uploaded anywhere, and are gone when you close it or reset the demo."
            : ""}
        </p>
      </div>

      {readOnly ? (
        <Banner tone="info" title="Uploads are disabled in the static demo">
          The list below shows the sample corpus that was indexed when the demo was recorded.
        </Banner>
      ) : (
        <form className="card" onSubmit={onUpload} aria-label="Upload a document">
          <div
            className={dragging ? "dropzone active" : "dropzone"}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
          >
            <p style={{ margin: "0 0 8px" }}>Drag a file here, or</p>
            <label className="btn" htmlFor="file-input">
              Choose a file
            </label>
            <input
              id="file-input"
              ref={inputRef}
              className="sr-only"
              type="file"
              accept={allowed.join(",")}
              onChange={(event) => choose(event.target.files?.[0] ?? null)}
            />
            <p className="small muted" style={{ margin: "8px 0 0" }}>
              {allowed.join(", ")} · up to {formatBytes(maxBytes)}
            </p>
          </div>
          {file ? (
            <div className="controls-row" style={{ marginTop: 12 }}>
              <div className="field" style={{ flex: 1 }}>
                <span className="small secondary">Selected</span>
                <span>
                  {file.name} <span className="muted small">({formatBytes(file.size)})</span>
                </span>
              </div>
              <div className="field" style={{ flex: 1 }}>
                <label htmlFor="doc-title">Title (optional)</label>
                <input id="doc-title" type="text" value={title} maxLength={300} onChange={(e) => setTitle(e.target.value)} />
              </div>
              <button type="submit" className="btn btn-primary" disabled={uploading || Boolean(validation)}>
                {uploading ? <span className="spinner" /> : null} Upload
              </button>
            </div>
          ) : null}
          {validation ? (
            <p role="alert" className="small" style={{ color: "var(--critical-ink)", marginBottom: 0 }}>
              {validation}
            </p>
          ) : null}
          {uploadError ? (
            <div style={{ marginTop: 12 }}>
              <ErrorBanner error={uploadError} />
              {uploadError instanceof ApiError && uploadError.code === "duplicate_document" ? (
                <p className="small muted">Existing document id: {String(uploadError.details["document_id"] ?? "")}</p>
              ) : null}
            </div>
          ) : null}
        </form>
      )}

      {loadError ? <ErrorBanner error={loadError} onRetry={load} /> : null}

      <section className="card" aria-labelledby="docs-heading">
        <div className="card-header">
          <h2 id="docs-heading">Indexed documents</h2>
          <span className="small muted">
            {page ? `${page.total} total` : ""} {busy ? "· updating" : ""}
          </span>
        </div>
        {!page && !loadError ? <Loading /> : null}
        {page && page.total === 0 ? (
          <EmptyState title="No documents yet">
            <p className="small">
              Upload a file above (two small sample PDFs are in the repository&apos;s <code>examples/</code> folder), or
              index the sample corpus from a terminal:
              <br />
              <code>groundedai ingest evaluation/datasets/groundedai-synthetic-v1/corpus/*</code>
            </p>
          </EmptyState>
        ) : null}
        {page && page.items.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Title</th>
                  <th>Status</th>
                  <th className="num">Pages</th>
                  <th className="num">Chunks</th>
                  <th className="num">Size</th>
                  <th>Added</th>
                  <th>
                    <span className="sr-only">Actions</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {page.items.map((doc) => (
                  <DocumentRow
                    key={doc.id}
                    doc={doc}
                    readOnly={readOnly}
                    chunks={openChunks?.id === doc.id ? openChunks.chunks : null}
                    onToggleChunks={() => void toggleChunks(doc.id)}
                    onAsk={() => askAbout(doc.id)}
                    onReingest={() => void act(() => api.reingestDocument(doc.id))}
                    onDelete={() => {
                      if (window.confirm(`Delete "${doc.title}" and its chunks?`)) void act(() => api.deleteDocument(doc.id));
                    }}
                  />
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
        {page && page.total > PAGE_SIZE ? (
          <div className="controls-row" style={{ marginTop: 12, justifyContent: "flex-end" }}>
            <button type="button" className="btn btn-small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}>
              Previous
            </button>
            <span className="small muted">
              {offset + 1}–{Math.min(offset + PAGE_SIZE, page.total)} of {page.total}
            </span>
            <button type="button" className="btn btn-small" disabled={offset + PAGE_SIZE >= page.total} onClick={() => setOffset(offset + PAGE_SIZE)}>
              Next
            </button>
          </div>
        ) : null}
      </section>
    </>
  );
}

function DocumentRow({
  doc,
  readOnly,
  chunks,
  onToggleChunks,
  onAsk,
  onReingest,
  onDelete,
}: {
  doc: DocumentOut;
  readOnly: boolean;
  chunks: ChunkOut[] | null;
  onToggleChunks: () => void;
  onAsk: () => void;
  onReingest: () => void;
  onDelete: () => void;
}) {
  const settled = doc.status === "ready" || doc.status === "failed";
  return (
    <>
      <tr>
        <td>
          <div style={{ fontWeight: 600 }}>{doc.title}</div>
          <div className="small muted">{doc.filename}</div>
          {doc.status === "failed" && doc.error_message ? (
            <div className="small" style={{ color: "var(--critical-ink)" }}>
              {doc.error_message}
            </div>
          ) : null}
        </td>
        <td>
          <DocumentStatusBadge status={doc.status} />
        </td>
        <td className="num">{doc.page_count ?? "–"}</td>
        <td className="num">{doc.chunk_count}</td>
        <td className="num">{formatBytes(doc.size_bytes)}</td>
        <td className="small">{formatDate(doc.created_at)}</td>
        <td style={{ whiteSpace: "nowrap" }}>
          {!readOnly ? (
            <>
              <button
                type="button"
                className="btn btn-small btn-primary"
                onClick={onAsk}
                disabled={doc.status !== "ready"}
                aria-label={`Ask about ${doc.title}`}
              >
                Ask
              </button>{" "}
              <button type="button" className="btn btn-small" onClick={onToggleChunks} disabled={doc.status !== "ready"}>
                {chunks ? "Hide chunks" : "Chunks"}
              </button>{" "}
              <button
                type="button"
                className="btn btn-small"
                onClick={onReingest}
                disabled={!settled || doc.source_stored === false}
                title={doc.source_stored === false ? "The file was not kept; upload it again to retry." : undefined}
              >
                Re-ingest
              </button>{" "}
              <button type="button" className="btn btn-small btn-danger" onClick={onDelete} disabled={doc.status === "processing"}>
                Delete
              </button>
            </>
          ) : null}
        </td>
      </tr>
      {chunks ? (
        <tr>
          <td colSpan={7}>
            <div className="source-list">
              {chunks.map((chunk) => (
                <div key={chunk.id} className="source">
                  <div className="source-head">
                    <span className="source-label">#{chunk.chunk_index}</span>
                    <span className="small muted">{pageRange(chunk.page_start, chunk.page_end)}</span>
                  </div>
                  <pre className="passage">{chunk.text}</pre>
                </div>
              ))}
            </div>
          </td>
        </tr>
      ) : null}
    </>
  );
}
