import type { AnswerResponse } from "../api/types";

type Row = { property: string; result: string; detail?: string };

/**
 * The separate properties of an answer, each with what was actually checked. None of them is
 * a judgement of truth, and they are never merged into one status: a statement can match the
 * retrieved text and still not answer the question, and a valid citation can point to a
 * passage that does not support the sentence.
 */
export function checkRows(response: AnswerResponse): Row[] {
  const ran = new Set(response.checks_run ?? []);
  const answered = response.status === "answered";
  const verification = response.verification ?? null;
  const claims = response.claims ?? [];
  const rows: Row[] = [
    {
      property: "Retrieval",
      result:
        response.sources.length > 0
          ? `${response.sources.length} passage${response.sources.length === 1 ? "" : "s"} retrieved`
          : "no passage retrieved",
      detail: "Ranked by word and vector search; retrieval does not show that a passage answers the question.",
    },
  ];
  if (answered && ran.has("citation_labels")) {
    const invalid = response.citation_check.invalid;
    rows.push({
      property: "Citation labels",
      result:
        invalid.length > 0
          ? `${invalid.length} label(s) name a source that was not provided: ${invalid.join(", ")}`
          : response.citation_check.valid.length > 0
            ? "every label names a provided source"
            : "no source labels in the answer",
      detail: "Checks that each [S#] exists, not that the passage supports the sentence.",
    });
  }
  if (answered && verification && ran.has("lexical_claims")) {
    const matched = verification.supported + verification.wrong_source + verification.added_source;
    rows.push({
      property: "Lexical match",
      result: `${matched} of ${verification.claims} statements matched retrieved text`,
      detail: "Words and figures are compared, not meaning.",
    });
  }
  if (answered && ran.has("added_details")) {
    const added = claims.filter((claim) => (claim.added_details ?? []).length > 0);
    rows.push({
      property: "Added details",
      result:
        added.length > 0
          ? `${added.length} statement(s) add words that no passage states`
          : "none found",
      detail: "Two or more consecutive words of a statement that appear in no retrieved passage.",
    });
  } else if (answered) {
    rows.push({
      property: "Added details",
      result: "not checked (optional rule, off by default)",
      detail:
        "The added-detail rule runs only when the server sets GROUNDEDAI_CLAIM_DETAIL_CHECK=on; it did not help on the hand-written test claims.",
    });
  }
  if (ran.has("relevance") && response.relevance) {
    const missing = response.relevance.unmatched;
    rows.push({
      property: "Question relevance",
      result:
        missing.length > 0
          ? `the passages do not mention: ${missing.join(", ")}`
          : "every asked word appears in a retrieved passage",
      detail: "Lexical: a passage can mention every asked word and still be about something else.",
    });
  }
  const meaning = response.meaning;
  if (answered && meaning && ran.has("meaning_nli")) {
    const count = (kind: string) => meaning.claims.filter((claim) => claim.meaning === kind).length;
    const contradicted = count("contradicted");
    rows.push({
      property: "Meaning (entailment)",
      result:
        `${count("entailed")} of ${meaning.claims.length} statements entailed by a passage` +
        (contradicted > 0 ? `; ${contradicted} appear contradicted` : ""),
      detail: `${meaning.model}, ${meaning.backend.toUpperCase()} in this tab, ${meaning.pairs} comparisons in ${meaning.total_ms} ms. An NLI model's judgement, not proof.`,
    });
  } else {
    rows.push({
      property: "Meaning (entailment)",
      result: "not checked in this answer",
      detail: "No semantic verifier ran; nothing here says that the statements mean what the sources mean.",
    });
  }
  rows.push({
    property: "Answer correctness",
    result: "not assessed",
    detail: "It needs a reference answer; the Evaluation page reports it for the evaluation questions.",
  });
  return rows;
}

export function CheckSummary({ response }: { response: AnswerResponse }) {
  if (!response.checks_run || response.checks_run.length === 0) return null;
  return (
    <details className="check-summary" style={{ marginTop: 14 }}>
      <summary>Checks that ran on this answer</summary>
      <table className="table" aria-label="Checks that ran">
        <tbody>
          {checkRows(response).map((row) => (
            <tr key={row.property}>
              <th scope="row">{row.property}</th>
              <td>
                {row.result}
                {row.detail ? <div className="small muted">{row.detail}</div> : null}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </details>
  );
}
