import type { AnswerResponse, MeaningCheck } from "../api/types";
import type { Classify } from "../browser/nliModel";
import { judge, nearestPremises, windowsOf, type Premise } from "./premises";

/**
 * Run the meaning check on an answer's statements: each statement against the six sentence
 * windows of the retrieved passages that share the most words with it. Adds `meaning` to the
 * answer and "meaning_nli" to `checks_run`. The lexical checks are not changed; the outcome is
 * a second, separate opinion.
 */
export async function checkMeaning(
  response: AnswerResponse,
  classify: Classify,
  model: { model: string; backend: "wasm" },
): Promise<AnswerResponse> {
  const claims = response.claims ?? [];
  if (response.status !== "answered" || claims.length === 0) return response;
  const started = performance.now();
  const windows = windowsOf(response.sources.map((source) => ({ label: source.label, text: source.text })));
  const chosen: Premise[][] = claims.map((claim) => nearestPremises(claim.text, windows));
  const pairs: [string, string][] = [];
  chosen.forEach((premises, index) => {
    for (const premise of premises) pairs.push([premise.text, claims[index]?.text ?? ""]);
  });
  const { probabilities, ms } = await classify(pairs);
  let offset = 0;
  const results: MeaningCheck["claims"] = chosen.map((premises, index) => {
    const own = premises.map((premise, i) => ({ premise, probabilities: probabilities[offset + i] ?? { entailment: 0, neutral: 1, contradiction: 0 } }));
    offset += premises.length;
    const verdict = judge(own);
    return {
      index,
      meaning: verdict.meaning,
      source: verdict.premise?.label ?? null,
      premise: verdict.premise?.text ?? null,
      probability: Math.round(verdict.probability * 1000) / 1000,
    };
  });
  const meaning: MeaningCheck = {
    model: model.model,
    backend: model.backend,
    pairs: pairs.length,
    model_ms: Math.round(ms),
    total_ms: Math.round(performance.now() - started),
    claims: results,
  };
  return { ...response, meaning, checks_run: [...(response.checks_run ?? []), "meaning_nli"] };
}
