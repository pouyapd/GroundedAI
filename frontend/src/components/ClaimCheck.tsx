import type { ClaimOut, EvidenceOut, VerificationOut } from "../api/types";
import { pageRange } from "../lib/format";
import { ClaimStatusBadge } from "./Badge";

/** One passage sentence quoted as evidence, with its source label, title and page. */
export function EvidenceQuote({
  label,
  title,
  pageStart,
  pageEnd,
  page,
  quote,
  onCite,
}: {
  label: string;
  title: string;
  pageStart: number | null;
  pageEnd: number | null;
  /** The quote's own page, when known; otherwise the passage's page range is shown. */
  page?: number | null;
  quote: string;
  onCite: (label: string) => void;
}) {
  return (
    <blockquote className="evidence">
      <span className="evidence-source">
        <button type="button" className="cite" onClick={() => onCite(label)} aria-label={`Show source ${label}`}>
          {label}
        </button>{" "}
        {title}
        {page != null ? ` · p. ${page}` : pageStart !== null ? ` · ${pageRange(pageStart, pageEnd)}` : ""}
      </span>
      <span className="evidence-quote">“{quote}”</span>
    </blockquote>
  );
}

function evidenceLabels(evidence: EvidenceOut[]): string {
  return [...new Set(evidence.map((item) => item.label))].join(", ");
}

export function ClaimCheck({
  claims,
  verification,
  onCite,
}: {
  claims: ClaimOut[];
  verification: VerificationOut;
  onCite: (label: string) => void;
}) {
  const found = verification.supported + verification.wrong_source + verification.added_source;
  return (
    <details open style={{ marginTop: 16 }}>
      <summary>
        Claim check ({found} of {verification.claims} statements found in the retrieved passages)
      </summary>
      <p className="small muted">
        {verification.method}
        {verification.references_variant > 0
          ? ` The model wrote ${verification.references_variant} citation(s) in another form than [S#] (for example [Source 1]); they are read here but do not count as valid citations.`
          : ""}
      </p>
      <ol className="claim-list">
        {claims.map((claim, index) => (
          <li key={index} className="claim">
            <div className="claim-head">
              <ClaimStatusBadge status={claim.status} />
              <span>{claim.text}</span>
            </div>
            {claim.status === "wrong_source" ? (
              <p className="small secondary claim-note">
                The model cited {claim.citations.join(", ")}; the statement is in {evidenceLabels(claim.evidence)}.
              </p>
            ) : null}
            {claim.status === "added_source" ? (
              <p className="small secondary claim-note">
                The model gave no citation; the passage below was matched by GroundedAI.
              </p>
            ) : null}
            {claim.missing_figures.length > 0 ? (
              <p className="small claim-note" style={{ color: "var(--critical-ink)" }}>
                Not stated in any retrieved passage: {claim.missing_figures.join(", ")}
              </p>
            ) : null}
            {claim.flags.includes("instruction_text") ? (
              <p className="small claim-note" style={{ color: "var(--critical-ink)" }}>
                Repeats instruction-like text from a source; not treated as a fact.
              </p>
            ) : null}
            {claim.status === "partial" && claim.evidence.length > 0 ? (
              <p className="small secondary claim-note">Closest passage (not enough overlap to confirm):</p>
            ) : null}
            {claim.evidence.map((item, evidenceIndex) => (
              <EvidenceQuote
                key={evidenceIndex}
                label={item.label}
                title={item.document_title}
                pageStart={item.page_start}
                pageEnd={item.page_end}
                page={item.page}
                quote={item.quote}
                onCite={onCite}
              />
            ))}
          </li>
        ))}
      </ol>
    </details>
  );
}
