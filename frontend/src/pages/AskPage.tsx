import { useEffect, useState, type FormEvent } from "react";

import { findRecorded, StaticDemoApi, type RecordedAnswers } from "../api/demo";
import type { AnswerResponse, RetrievalMode } from "../api/types";
import { AnswerView } from "../components/AnswerView";
import { ErrorBanner } from "../components/ErrorBanner";
import { Loading } from "../components/Banner";
import { useApp } from "../context";

const MODES: { value: RetrievalMode; label: string }[] = [
  { value: "hybrid", label: "Hybrid" },
  { value: "lexical", label: "Lexical" },
  { value: "dense", label: "Dense" },
];

const LIVE_EXAMPLES = [
  "What does error code E07 mean on the SX-5 inverter?",
  "What is the maximum hotel cost per night that Aldervale reimburses?",
  "What are the payment terms for Aldervale suppliers?",
  "How many vacation days do contractors receive at Aldervale?",
];

export function AskPage() {
  const { api, status } = useApp();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<RetrievalMode>("hybrid");
  const [topK, setTopK] = useState(5);
  const [diagnostics, setDiagnostics] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [response, setResponse] = useState<AnswerResponse | null>(null);
  const [asked, setAsked] = useState("");
  const [recorded, setRecorded] = useState<RecordedAnswers | null>(null);

  const isDemo = api instanceof StaticDemoApi;
  useEffect(() => {
    if (api instanceof StaticDemoApi) {
      api.recordedAnswers().then(setRecorded).catch(setError);
    }
  }, [api]);

  const maxChars = status?.limits.max_question_chars ?? 1000;
  const trimmed = question.trim();
  const tooShort = trimmed.length < 3;
  const tooLong = trimmed.length > maxChars;

  async function ask(text: string) {
    setLoading(true);
    setError(null);
    setResponse(null);
    setAsked(text);
    try {
      setResponse(
        await api.query({ question: text, retrieval_mode: mode, top_k: topK, include_diagnostics: diagnostics }),
      );
    } catch (caught) {
      setError(caught);
    } finally {
      setLoading(false);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
    if (!tooShort && !tooLong) void ask(trimmed);
  }

  const examples = isDemo ? (recorded?.items.map((item) => item.question) ?? []) : LIVE_EXAMPLES;

  return (
    <>
      <div>
        <h1 className="page-title">Ask your documents</h1>
        <p className="page-intro">
          Answers are built only from retrieved passages. Every citation is checked against the passages the model
          received, and the system says so when the evidence is not enough.
        </p>
      </div>

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
                  onClick={() => setMode(item.value)}
                  disabled={isDemo}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
          <div className="field" style={{ minWidth: 100 }}>
            <label htmlFor="topk">Passages</label>
            <select id="topk" value={topK} onChange={(event) => setTopK(Number(event.target.value))} disabled={isDemo}>
              {[3, 5, 8, 10].map((value) => (
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
          <button type="submit" className="btn btn-primary" disabled={loading || tooShort || tooLong} style={{ marginLeft: "auto" }}>
            {loading ? <span className="spinner" /> : null} Ask question
          </button>
        </div>
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

      {loading ? <Loading label="Retrieving passages and generating an answer" /> : null}
      {error ? <ErrorBanner error={error} /> : null}
      {response ? (
        <AnswerView
          response={response}
          recordedNote={isDemo && recorded ? `${recorded.note} Recorded ${recorded.recorded_at}.` : undefined}
          expected={isDemo && recorded ? findRecorded(recorded, asked)?.expected : undefined}
        />
      ) : null}
    </>
  );
}
