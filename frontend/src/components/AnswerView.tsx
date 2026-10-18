import { useMemo, useState } from "react";

import { compareWithExpected, type RecordedExpectation } from "../api/demo";
import type { AnswerResponse, RetrievalMode } from "../api/types";
import { formatMs } from "../lib/format";
import { AnswerStatusBadge, Badge, SupportBadge, VerificationBadges } from "./Badge";
import { Banner } from "./Banner";
import { CitedText } from "./CitedText";
import { ClaimCheck } from "./ClaimCheck";
import { Disagreements } from "./Disagreements";
import { SourceCard } from "./SourceCard";

const MODE_NAMES: Record<RetrievalMode, string> = { hybrid: "Hybrid", lexical: "Lexical (BM25)", dense: "Dense" };
const GROUNDING_TEXT = {
  grounded: "every sentence cited",
  partially_grounded: "some sentences cited",
  ungrounded: "no valid citations",
} as const;

function ExpectedResult({
  expected,
  response,
  keyFactRecall,
}: {
  expected: RecordedExpectation;
  response: AnswerResponse;
  keyFactRecall: number | null;
}) {
  const { matches, summary } = compareWithExpected(expected, response.status);
  return (
    <div className="expected" aria-label="Expected result from the evaluation dataset">
      <div className="expected-header">
        <strong>Expected by the evaluation dataset</strong>
        {matches === true ? <Badge tone="good">Behaviour matches</Badge> : null}
        {matches === false ? <Badge tone="critical">Behaviour does not match</Badge> : null}
        {matches === null ? <Badge tone="neutral">Not judged here</Badge> : null}
      </div>
      <p>{summary}</p>
      {expected.reference_answer ? (
        <p>
          <span className="secondary">Reference answer:</span> {expected.reference_answer}
        </p>
      ) : null}
      {expected.notes ? (
        <p>
          <span className="secondary">Dataset note:</span> {expected.notes}
        </p>
      ) : null}
      {keyFactRecall !== null ? (
        <p>
          <span className="secondary">Key facts from the dataset found in the answer:</span>{" "}
          {Math.round(keyFactRecall * 100)}%
        </p>
      ) : null}
      <p className="small muted">
        Question {expected.question_id}, category {expected.category.replaceAll("_", " ")}.
        {response.status === "answered" ? " A valid citation does not make an answer correct." : null}
      </p>
    </div>
  );
}

export function AnswerView({
  response,
  recordedNote,
  expected,
  keyFactRecall = null,
  askedWith,
}: {
  response: AnswerResponse;
  recordedNote?: string;
  expected?: RecordedExpectation;
  keyFactRecall?: number | null;
  /** The retrieval settings the question was sent with. */
  askedWith?: { mode: RetrievalMode; topK: number };
}) {
  const [highlighted, setHighlighted] = useState<string | null>(null);
  const validLabels = useMemo(() => new Set(response.citation_check.valid), [response]);
  const claims = response.claims ?? [];
  const conflicts = response.conflicts ?? [];
  const verification = response.verification ?? null;
  const isEvidence = (source: AnswerResponse["sources"][number]) =>
    source.cited || (source.supports_claims?.length ?? 0) > 0;
  // Disagreements are shown in their own panel; do not repeat their explanation as a warning.
  const warnings = response.warnings.filter((warning) => !conflicts.some((item) => item.explanation === warning));
  const cited = response.sources.filter(isEvidence);
  const uncited = response.sources.filter((source) => !isEvidence(source));
  const showCite = (label: string) => {
    setHighlighted(label);
    document.getElementById(`source-${label}`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  return (
    <section className="card" aria-labelledby="answer-heading">
      <div className="card-header">
        <h2 id="answer-heading">Answer</h2>
        {response.status === "answered" && verification ? (
          <VerificationBadges verification={verification} />
        ) : (
          <AnswerStatusBadge status={response.status} grounding={response.grounding} />
        )}
      </div>

      {recordedNote ? (
        <div style={{ marginBottom: 12 }}>
          <Banner tone="info" title="Recorded answer (static demo)">
            {recordedNote}
          </Banner>
        </div>
      ) : null}

      {response.status === "answered" && response.answer ? (
        <CitedText text={response.answer} validLabels={validLabels} onCite={showCite} />
      ) : null}
      {response.status === "insufficient_evidence" ? (
        <p className="answer-text">
          The indexed documents do not contain enough evidence to answer this question.
          {response.abstention_reason ? <span className="secondary"> ({response.abstention_reason})</span> : null}
        </p>
      ) : null}
      {response.status === "invalid_output" ? (
        <p className="answer-text">The model returned no usable answer. The retrieved passages are listed below.</p>
      ) : null}
      {response.status === "insufficient_evidence" && response.unverified_answer ? (
        <details style={{ marginTop: 10 }}>
          <summary>Show the withheld answer (none of its statements was found in the sources)</summary>
          <p className="small secondary">{response.unverified_answer}</p>
        </details>
      ) : null}

      {conflicts.length > 0 ? <Disagreements conflicts={conflicts} onCite={showCite} /> : null}

      {expected ? <ExpectedResult expected={expected} response={response} keyFactRecall={keyFactRecall} /> : null}

      <div className="meta-row" style={{ marginTop: 12 }}>
        {askedWith ? (
          <span>
            {MODE_NAMES[askedWith.mode]} retrieval · {response.sources.length} of up to {askedWith.topK} passages used
          </span>
        ) : null}
        {response.provider.is_llm ? (
          <span>
            Generated by <strong>{response.provider.name}</strong> · <code>{response.provider.model}</code>
          </span>
        ) : (
          <Badge tone="info">Demo generator: extractive sentence selection, not an LLM</Badge>
        )}
        {response.status === "answered" && verification && response.grounding ? (
          <span title="Strict check of the model's own [S#] citations, as in earlier reports">
            model citations: {GROUNDING_TEXT[response.grounding]}
          </span>
        ) : null}
        <span>retrieval {formatMs(response.timings.retrieval_ms)}</span>
        <span>generation {formatMs(response.timings.generation_ms)}</span>
        <span>total {formatMs(response.timings.total_ms)}</span>
        {response.usage.completion_tokens !== null ? (
          <span>
            tokens {response.usage.prompt_tokens ?? "?"} in / {response.usage.completion_tokens} out
          </span>
        ) : null}
      </div>

      {warnings.length > 0 ? (
        <div style={{ marginTop: 14 }}>
          <Banner tone="warning" title="Check before relying on this answer">
            <ul className="warnings">
              {warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
            </ul>
          </Banner>
        </div>
      ) : null}

      {verification && claims.length > 0 ? (
        <ClaimCheck claims={claims} verification={verification} onCite={showCite} />
      ) : null}

      {response.sentences.length > 0 ? (
        <details style={{ marginTop: 16 }}>
          <summary>
            {verification ? "Strict [S#] check, as in earlier reports" : "Sentence-level check"} (
            {response.citation_check.sentences_cited} of {response.citation_check.sentences_total} sentences cited)
          </summary>
          <p className="small muted">
            Citation validity is exact: every label must match a provided source. Support is a word and number overlap
            check against the cited passage; it flags invented numbers but cannot confirm meaning.
          </p>
          <ul className="sentence-list">
            {response.sentences.map((sentence, index) => (
              <li key={index}>
                <SupportBadge support={sentence.support} />
                <span>
                  {sentence.text}{" "}
                  {sentence.citations.length ? <span className="muted small">[{sentence.citations.join(", ")}]</span> : null}
                  {sentence.unsupported_numbers.length ? (
                    <span className="small" style={{ color: "var(--critical-ink)" }}>
                      {" "}
                      numbers not in the source: {sentence.unsupported_numbers.join(", ")}
                    </span>
                  ) : null}
                </span>
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      {cited.length > 0 ? (
        <div style={{ marginTop: 20 }}>
          <h3>{verification ? "Passages used as evidence" : "Cited sources"}</h3>
          <div className="source-list">
            {cited.map((source) => (
              <SourceCard key={source.label} source={source} highlighted={highlighted === source.label} />
            ))}
          </div>
        </div>
      ) : null}

      {uncited.length > 0 ? (
        <details style={{ marginTop: 16 }} open={cited.length === 0}>
          <summary>
            {verification ? "Other retrieved passages" : "Retrieved but not cited"} ({uncited.length})
          </summary>
          <div className="source-list" style={{ marginTop: 10 }}>
            {uncited.map((source) => (
              <SourceCard key={source.label} source={source} highlighted={highlighted === source.label} />
            ))}
          </div>
        </details>
      ) : null}

      {response.retrieval ? (
        <details style={{ marginTop: 16 }}>
          <summary>Retrieval diagnostics</summary>
          <div className="table-wrap" style={{ marginTop: 8 }}>
            <table>
              <tbody>
                <tr>
                  <th>Mode</th>
                  <td>
                    {response.retrieval.mode}
                    {response.retrieval.fusion ? ` (${response.retrieval.fusion} fusion)` : ""}
                  </td>
                </tr>
                <tr>
                  <th>Embedding model</th>
                  <td>
                    <code>{response.retrieval.embedding_model ?? "–"}</code>
                  </td>
                </tr>
                <tr>
                  <th>Reranker</th>
                  <td>{response.retrieval.reranker ?? "none"}</td>
                </tr>
                <tr>
                  <th>Candidates</th>
                  <td>
                    lexical {response.retrieval.lexical_candidates}, dense {response.retrieval.dense_candidates}, fused{" "}
                    {response.retrieval.fused_candidates}, duplicates removed {response.retrieval.duplicates_removed}
                  </td>
                </tr>
                <tr>
                  <th>Timings</th>
                  <td>
                    {Object.entries(response.retrieval.timings_ms)
                      .map(([name, value]) => `${name.replace(/_ms$/, "")} ${formatMs(value)}`)
                      .join(" · ")}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </section>
  );
}
