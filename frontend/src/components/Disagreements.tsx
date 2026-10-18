import type { ConflictOut, ConflictStatementOut } from "../api/types";
import { Badge } from "./Badge";
import { EvidenceQuote } from "./ClaimCheck";

/** What one side states: its values, or whether it says something is (not) required. */
function statedValue(statement: ConflictStatementOut): string {
  if (statement.values.length > 0) return statement.values.join(", ");
  if (statement.denies_requirement) return "not required";
  if (statement.affirms_requirement) return "required";
  return "no value stated";
}

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
                  page={statement.page}
                  quote={statement.quote}
                  onCite={onCite}
                />
                <p className="small secondary claim-note">
                  {`Value: ${statedValue(statement)}`}
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
