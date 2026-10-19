import { useEffect, useRef, useState, type FormEvent } from "react";

import { findRecorded, findRecording, StaticDemoApi, type RecordedAnswers } from "../api/demo";
import type { AnswerResponse, DocumentOut, RetrievalMode } from "../api/types";
import { AnswerView } from "../components/AnswerView";
import { ErrorBanner } from "../components/ErrorBanner";
import { Loading } from "../components/Banner";
import type { BrowserApi } from "../browser/BrowserApi";
import { DemoPanel } from "../browser/DemoPanel";
import { useApp } from "../context";

export const MODES: { value: RetrievalMode; label: string; description: string }[] = [
  {
    value: "hybrid",
    label: "Hybrid",
    description: "BM25 keyword search and embedding search, combined with reciprocal rank fusion.",
  },
  { value: "lexical", label: "Lexical", description: "BM25 keyword search only (no embeddings)." },
  { value: "dense", label: "Dense", description: "Embedding similarity search only (no keyword matching)." },
];

const TOP_K_VALUES = [3, 5, 8, 10];

/** Example questions, offered only when the bundled sample corpus has been indexed. */
const SAMPLE_CORPUS_FILE = /^(aldervale|solvane)-/;
const LIVE_EXAMPLES = [
  "What does error code E07 mean on the SX-5 inverter?",
  "What is the maximum hotel cost per night that Aldervale reimburses?",
  "What are the payment terms for Aldervale suppliers?",
  "How many vacation days do contractors receive at Aldervale?",
];

/** Online demo: questions about the two sample PDFs, each asked of one document. */
const BROWSER_EXAMPLES: { question: string; file: string; label: string }[] = [
  { question: "How much does a single adult ticket to Mirrow cost?", file: "harbourline-ferry-guide.pdf", label: "Ferry guide" },
  { question: "How much does it cost to take a bicycle?", file: "harbourline-ferry-guide.pdf", label: "Ferry guide" },
  { question: "Do young children have to pay for the ferry?", file: "harbourline-ferry-guide.pdf", label: "Ferry guide" },
  { question: "How many items can members borrow?", file: "harbourline-ferry-guide.pdf", label: "Ferry guide" },
  { question: "How many items can members borrow?", file: "orchard-library-rules.pdf", label: "Library rules" },
];

/** The settings a displayed answer was produced with. */
interface AskedWith {
  question: string;
  mode: RetrievalMode;
  topK: number;
  diagnostics: boolean;
  /** "all", or the id of the one document searched. */
  scope: string;
  scopeLabel: string;
}

export function modeLabel(mode: RetrievalMode): string {
  return MODES.find((item) => item.value === mode)?.label ?? mode;
}

export function AskPage({ active = true }: { active?: boolean }) {
  const { api, status, askScope, setAskScope: setScope } = useApp();
  const browserApi = api.mode === "browser" ? (api as BrowserApi) : null;
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [diagnostics, setDiagnostics] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [response, setResponse] = useState<AnswerResponse | null>(null);
  const [asked, setAsked] = useState<AskedWith | null>(null);
  const [recorded, setRecorded] = useState<RecordedAnswers | null>(null);
  const [readyDocs, setReadyDocs] = useState<DocumentOut[] | null>(null);
  const latestRequest = useRef(0);

  const isDemo = api instanceof StaticDemoApi;

  useEffect(() => {
    if (api instanceof StaticDemoApi) {
      api.recordedAnswers().then(setRecorded).catch(setError);
    }
  }, [api]);

  // Refresh the searchable documents whenever the page is shown (uploads happen on another tab).
  useEffect(() => {
    if (isDemo || !active) return;
    let cancelled = false;
    api
      .listDocuments(0, 100)
      .then((page) => {
        if (!cancelled) setReadyDocs(page.items.filter((doc) => doc.status === "ready"));
      })
      .catch(() => {
        // The list only fills the document selector; questions still work without it.
        if (!cancelled) setReadyDocs(null);
      });
    return () => {
      cancelled = true;
    };
  }, [api, active, isDemo]);

  // A selected document that was deleted (or is being re-ingested) falls back to all documents.
  const scopeMissing = askScope !== "all" && readyDocs !== null && !readyDocs.some((doc) => doc.id === askScope);
  const scope = scopeMissing ? "all" : askScope;

  const maxChars = status?.limits.max_question_chars ?? 1000;
  const trimmed = question.trim();
  const tooShort = trimmed.length < 3;
  const tooLong = trimmed.length > maxChars;
  const topKValues = isDemo && recorded ? recorded.top_k_values : TOP_K_VALUES;
  const noDocuments = !isDemo && readyDocs !== null && readyDocs.length === 0;
  const scopeLabel =
    scope === "all"
      ? readyDocs
        ? `all ${readyDocs.length} ready document${readyDocs.length === 1 ? "" : "s"}`
        : "all documents"
      : `“${readyDocs?.find((doc) => doc.id === scope)?.title ?? scope}”`;

  async function ask(text: string, scopeOverride?: { id: string; label: string }) {
    const askedScope = scopeOverride?.id ?? scope;
    const settings: AskedWith = {
      question: text,
      mode,
      topK,
      diagnostics,
      scope: askedScope,
      scopeLabel: scopeOverride ? `“${scopeOverride.label}”` : scopeLabel,
    };
    // Only the newest request may update the page, so a slow earlier answer never replaces it.
    const requestId = ++latestRequest.current;
    setLoading(true);
    setError(null);
    setResponse(null);
    setAsked(settings);
    try {
      const result = await api.query({
        question: text,
        retrieval_mode: mode,
        top_k: topK,
        include_diagnostics: diagnostics,
        ...(askedScope === "all" ? {} : { document_ids: [askedScope] }),
      });
      if (requestId === latestRequest.current) setResponse(result);
    } catch (caught) {
      if (requestId === latestRequest.current) setError(caught);
    } finally {
      if (requestId === latestRequest.current) setLoading(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!tooShort && !tooLong && !loading && !noDocuments) void ask(trimmed);
  }

  const sampleCorpusIndexed = readyDocs?.some((doc) => SAMPLE_CORPUS_FILE.test(doc.filename)) ?? false;
  const examples = isDemo
    ? (recorded?.items.map((item) => item.question) ?? [])
    : sampleCorpusIndexed && scope === "all"
      ? LIVE_EXAMPLES
      : [];
  const settingsChanged =
    asked !== null &&
    !loading &&
    (asked.mode !== mode || asked.topK !== topK || asked.diagnostics !== diagnostics || asked.scope !== scope);
  const recordedItem = isDemo && recorded && asked ? findRecorded(recorded, asked.question) : undefined;
  const recording = recordedItem && asked ? findRecording(recordedItem, asked.mode, asked.topK) : undefined;

  return (
    <>
      {browserApi ? (
        <DemoPanel api={browserApi} />
      ) : (
        <div>
          <h1 className="page-title">Ask your documents</h1>
          <p className="page-intro">
            Upload a PDF, ask a question, and get an answer built only from passages of your documents. Each statement
            is checked against those passages and shown with its quote and page, and the system says so when the
            evidence is not enough.
          </p>
        </div>
      )}

      {noDocuments ? (
        <div className="notice" role="status">
          <span>No document is ready to search yet. Upload a PDF, text or Markdown file first.</span>
          <a className="btn btn-small btn-primary" href="#documents">
            Go to Documents
          </a>
        </div>
      ) : null}

      <form className="card" onSubmit={onSubmit} aria-label="Ask a question">
        <div className="field">
          <label htmlFor="question">Question</label>
          <textarea
            id="question"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder={isDemo ? "Pick one of the recorded questions below" : "e.g. What is the hotel limit per night?"}
            maxLength={maxChars + 50}
            onKeyDown={(event) => {
              if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) onSubmit(event);
            }}
          />
          <span className="small muted">
            {tooLong ? `Too long: ${trimmed.length}/${maxChars} characters` : "Ctrl+Enter to send"}
          </span>
        </div>
        <div className="controls-row" style={{ marginTop: 12 }}>
          {!isDemo && readyDocs && readyDocs.length > 0 ? (
            <div className="field scope-field">
              <label htmlFor="scope">Search in</label>
              <select id="scope" value={scope} onChange={(event) => setScope(event.target.value)}>
                <option value="all">All ready documents ({readyDocs.length})</option>
                {readyDocs.map((doc) => (
                  <option key={doc.id} value={doc.id}>
                    {doc.title}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          <div className="field">
            <span className="small secondary" id="mode-label">
              Retrieval
            </span>
            <div className="segmented" role="group" aria-labelledby="mode-label">
              {MODES.map((item) => (
                <button
                  key={item.value}
                  type="button"
                  aria-pressed={mode === item.value}
                  title={item.description}
                  onClick={() => setMode(item.value)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <div className="field" style={{ minWidth: 100 }}>
            <label htmlFor="topk">Passages</label>
            <select id="topk" value={topK} onChange={(event) => setTopK(Number(event.target.value))}>
              {topKValues.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
          <label className="checkbox">
            <input type="checkbox" checked={diagnostics} onChange={(event) => setDiagnostics(event.target.checked)} />
            Retrieval diagnostics
          </label>
          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading || tooShort || tooLong || noDocuments}
            style={{ marginLeft: "auto" }}
          >
            {loading ? <span className="spinner" /> : null} Ask question
          </button>
        </div>
        <p className="small secondary" style={{ margin: "8px 0 0" }} id="mode-description">
          {MODES.find((item) => item.value === mode)?.description} Up to {topK} passages go to the{" "}
          {status?.generation.is_llm ? "language model" : "answer generator"}.
          {mode !== "lexical" && status?.embedding.provider === "hashing"
            ? ` ${browserApi ? "The demo" : "This server"} uses the hashing embedder, which matches words, not meaning.`
            : ""}
          {isDemo
            ? " Static demo: each combination of mode and passage count replays a recording made with exactly that setting."
            : ""}
        </p>
        {browserApi && readyDocs ? (
          <div style={{ marginTop: 16 }}>
            <p className="small secondary" style={{ margin: "0 0 6px" }}>
              Try a question about the sample PDFs (each one searches the document shown)
            </p>
            <div className="chips">
              {BROWSER_EXAMPLES.map((example) => {
                const doc = readyDocs.find((item) => item.filename === example.file);
                if (!doc) return null;
                return (
                  <button
                    key={`${example.file}|${example.question}`}
                    type="button"
                    className="chip chip-example"
                    disabled={loading}
                    onClick={() => {
                      setQuestion(example.question);
                      setScope(doc.id);
                      void ask(example.question, { id: doc.id, label: doc.title });
                    }}
                  >
                    <span className="chip-doc">{example.label}</span> {example.question}
                  </button>
                );
              })}
            </div>
          </div>
        ) : null}
        {examples.length > 0 ? (
          <div style={{ marginTop: 16 }}>
            <p className="small secondary" style={{ margin: "0 0 6px" }}>
              {isDemo ? "Recorded questions" : "Example questions for the bundled sample corpus"}
            </p>
            <div className="chips">
              {examples.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="chip"
                  disabled={loading}
                  onClick={() => {
                    setQuestion(example);
                    void ask(example);
                  }}
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </form>

      {scopeMissing ? (
        <div className="notice" role="status">
          <span>The selected document is no longer ready, so all documents are searched.</span>
        </div>
      ) : null}
      {settingsChanged && asked ? (
        <div className="notice" role="status">
          <span>
            The answer below used {modeLabel(asked.mode)} retrieval with {asked.topK} passages
            {asked.diagnostics ? "" : ", without diagnostics"}
            {isDemo ? "" : ` and searched ${asked.scopeLabel}`}. Your settings have changed.
          </span>
          <button type="button" className="btn btn-small" onClick={() => void ask(asked.question)}>
            Ask again with {modeLabel(mode)}, {topK} passages
          </button>
        </div>
      ) : null}
      {loading ? (
        <Loading
          label={
            browserApi?.generator === "webllm"
              ? "Retrieving passages, then generating on your GPU (often 10–60 s)"
              : "Retrieving passages and generating an answer"
          }
        />
      ) : null}
      {error ? <ErrorBanner error={error} /> : null}
      {response && asked && !isDemo ? (
        <p className="small secondary" style={{ margin: 0 }}>
          Searched {asked.scopeLabel}.
        </p>
      ) : null}
      {response && asked ? (
        <AnswerView
          response={response}
          askedWith={{ mode: asked.mode, topK: asked.topK }}
          recordedNote={isDemo && recorded ? `${recorded.note} Recorded ${recorded.recorded_at}.` : undefined}
          expected={recordedItem?.expected}
          keyFactRecall={recording?.key_fact_recall ?? null}
        />
      ) : null}
    </>
  );
}
