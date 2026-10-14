import type { ConflictOut } from "../api/types";
import { Badge } from "./Badge";
import { EvidenceQuote } from "./ClaimCheck";

/** Retrieved passages that disagree. Both sides are shown; nothing is resolved. */
export function Disagreements({ conflicts, onCite }: { conflicts: ConflictOut[]; onCite: (label: string) => void }) {
  return (
    <section className="disagreements" aria-label="Retrieved passages that disagree">
      {conflicts.map((conflict, index) => (
        <div className="disagreement" key={index}>
          <div className="claim-head">
            <Badge tone="warning">
              {conflict.kind === "conflict" ? "Sources disagree" : "Different values for different dates"}
            </Badge>
          </div>
          <p className="small">{conflict.explanation}</p>
          <div className="disagreement-sides">
            {conflict.statements.map((statement) => (
              <div key={statement.label}>
                <EvidenceQuote
                  label={statement.label}
                  title={statement.document_title}
                  pageStart={statement.page_start}
                  pageEnd={statement.page_end}
                  quote={statement.quote}
                  onCite={onCite}
                />
                <p className="small secondary claim-note">
                  {statement.denies_requirement && statement.values.length === 0
                    ? "Says there is no such requirement."
                    : `Value: ${statement.values.join(", ")}`}
                  {statement.dates.length > 0 ? ` · Dates: ${statement.dates.join("; ")}` : ""}
                  {conflict.answer_reports.includes(statement.label) ? " · The answer reports this value." : ""}
                </p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </section>
  );
}
