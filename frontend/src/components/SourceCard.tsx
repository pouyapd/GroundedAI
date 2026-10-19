import { useState } from "react";

import type { SourceOut } from "../api/types";
import { formatScore, pageRange } from "../lib/format";
import { Badge } from "./Badge";

export function SourceCard({ source, highlighted }: { source: SourceOut; highlighted: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const scores = source.scores;
  return (
    <article id={`source-${source.label}`} className={highlighted ? "source highlight" : "source"}>
      <div className="source-head">
        <span className="source-label">{source.label}</span>
        <span className="source-title">{source.document_title}</span>
        {source.page_start !== null ? (
          <span className="small muted">{pageRange(source.page_start, source.page_end)}</span>
        ) : null}
        {source.injection_flags.length > 0 ? (
          <Badge tone="warning">Contains instruction-like text</Badge>
        ) : null}
        {source.truncated ? <Badge tone="neutral">Shortened</Badge> : null}
      </div>
      <blockquote className="excerpt">{source.excerpt}</blockquote>
      {source.injection_flags.length > 0 ? (
        <p className="small muted" style={{ margin: "6px 0 0" }}>
          Detected patterns: {source.injection_flags.join(", ")}. The passage was given to the model as data only.
        </p>
      ) : null}
      <div className="scores" aria-label="Retrieval scores">
        <span>lexical rank {scores.lexical_rank ?? "–"}</span>
        <span>dense rank {scores.dense_rank ?? "–"}</span>
        <span>fused {formatScore(scores.fused, 4)}</span>
        {scores.rerank !== null ? <span>rerank {formatScore(scores.rerank, 2)}</span> : null}
      </div>
      <button type="button" className="btn-link small" onClick={() => setExpanded((value) => !value)} aria-expanded={expanded}>
        {expanded ? "Hide full passage" : "Show full passage"}
      </button>
      {expanded ? <pre className="passage">{source.text}</pre> : null}
    </article>
  );
}
