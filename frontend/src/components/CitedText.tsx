import { Fragment } from "react";

const CITATION = /\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]/gi;

export interface CitationPart {
  kind: "text" | "citation";
  value: string;
}

/** Split answer text into plain text and individual citation labels (S1, S2, ...). */
export function splitCitations(text: string): CitationPart[] {
  const parts: CitationPart[] = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    const index = match.index ?? 0;
    if (index > last) parts.push({ kind: "text", value: text.slice(last, index) });
    for (const label of (match[1] ?? "").split(/\s*[,;]\s*/)) {
      parts.push({ kind: "citation", value: label.toUpperCase() });
    }
    last = index + match[0].length;
  }
  if (last < text.length) parts.push({ kind: "text", value: text.slice(last) });
  return parts;
}

export function CitedText({
  text,
  validLabels,
  onCite,
}: {
  text: string;
  validLabels: ReadonlySet<string>;
  onCite: (label: string) => void;
}) {
  return (
    <p className="answer-text">
      {splitCitations(text).map((part, index) => {
        if (part.kind === "text") return <Fragment key={index}>{part.value}</Fragment>;
        if (!validLabels.has(part.value)) {
          return (
            <span
              key={index}
              className="cite cite-invalid"
              title={`${part.value} was not among the sources given to the model`}
            >
              {part.value}
            </span>
          );
        }
        return (
          <button
            key={index}
            type="button"
            className="cite"
            onClick={() => onCite(part.value)}
            aria-label={`Show source ${part.value}`}
          >
            {part.value}
          </button>
        );
      })}
    </p>
  );
}
