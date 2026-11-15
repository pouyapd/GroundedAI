/**
 * The meaning check's rules, without the model: which passage text each statement is compared
 * with, and how the model's labels become one outcome per statement.
 *
 * The same rules were validated in Python (experiments/verifier/answer_level.py with
 * `--premise window --max-windows 6`): NLI models are trained on short premises, so a statement
 * is compared with windows of one or two consecutive sentences of the retrieved passages, and
 * only with the six windows that share the most words with it.
 */

export type NliLabel = "entailment" | "neutral" | "contradiction";
export type NliProbabilities = Record<NliLabel, number>;
export type Meaning = "entailed" | "contradicted" | "not_entailed";

export const WINDOWS_PER_STATEMENT = 6;

const STOPWORDS = new Set(
  "a an and are as at be been by can do does for from has have how in is it its may not of on or that the their there these this to was were what when where which who will with".split(
    " ",
  ),
);

export interface Premise {
  label: string; // the passage, as S#
  text: string;
}

function words(text: string): Set<string> {
  return new Set(
    text
      .toLowerCase()
      .split(/[^\p{L}\p{N}]+/u)
      .filter((word) => word.length > 1 && !STOPWORDS.has(word)),
  );
}

/** Sentences of a passage: split after ".", "!" or "?" and at line breaks. */
export function splitSentences(text: string): string[] {
  return text
    .split(/\n+/)
    .flatMap((line) => line.split(/(?<=[.!?])\s+(?=[\p{Lu}\p{N}"“(])/u))
    .map((sentence) => sentence.replace(/\s+/g, " ").trim())
    .filter((sentence) => sentence.length > 0);
}

/** Every sentence and every pair of consecutive sentences of every passage. */
export function windowsOf(passages: { label: string; text: string }[]): Premise[] {
  const out: Premise[] = [];
  for (const passage of passages) {
    const sentences = splitSentences(passage.text);
    sentences.forEach((sentence) => out.push({ label: passage.label, text: sentence }));
    for (let i = 0; i + 1 < sentences.length; i += 1) {
      out.push({ label: passage.label, text: `${sentences[i]} ${sentences[i + 1]}` });
    }
  }
  return out;
}

/** The `keep` premises sharing the most words with the statement; ties keep passage order. */
export function nearestPremises(statement: string, premises: Premise[], keep = WINDOWS_PER_STATEMENT): Premise[] {
  if (premises.length <= keep) return premises;
  const asked = words(statement);
  const scored = premises.map((premise, index) => {
    let shared = 0;
    for (const word of words(premise.text)) if (asked.has(word)) shared += 1;
    return { index, shared };
  });
  scored.sort((a, b) => b.shared - a.shared || a.index - b.index);
  return scored
    .slice(0, keep)
    .map((item) => item.index)
    .sort((a, b) => a - b)
    .map((index) => premises[index] as Premise);
}

export function topLabel(probabilities: NliProbabilities): NliLabel {
  let best: NliLabel = "neutral";
  for (const label of ["entailment", "neutral", "contradiction"] as const) {
    if (probabilities[label] > probabilities[best]) best = label;
  }
  return best;
}

/**
 * One outcome per statement: entailed when a window entails it; contradicted when no window
 * entails it and one contradicts it; otherwise not entailed.
 */
export function judge(results: { premise: Premise; probabilities: NliProbabilities }[]): {
  meaning: Meaning;
  premise: Premise | null;
  probability: number;
} {
  let best: { premise: Premise; probability: number } | null = null;
  for (const wanted of ["entailment", "contradiction"] as const) {
    for (const result of results) {
      if (topLabel(result.probabilities) !== wanted) continue;
      const probability = result.probabilities[wanted];
      if (!best || probability > best.probability) best = { premise: result.premise, probability };
    }
    if (best) return { meaning: wanted === "entailment" ? "entailed" : "contradicted", ...best };
  }
  return { meaning: "not_entailed", premise: null, probability: 0 };
}
