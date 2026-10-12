import { ApiError, type Api } from "./client";
import type {
  AnswerResponse,
  AnswerStatus,
  ChunkOut,
  DocumentOut,
  EvaluationRun,
  OperationalMetrics,
  Page,
  QueryRequest,
  Report,
  ReportListItem,
  SystemStatus,
} from "./types";

/** What the evaluation dataset expects for a recorded question (copied from questions.jsonl). */
export interface RecordedExpectation {
  question_id: string;
  category: string;
  behavior: "answer" | "abstain" | "flag_conflict" | "answer_or_clarify";
  reference_answer: string | null;
  notes: string | null;
  /** Share of the dataset's key facts found in the recorded answer; null when it did not answer. */
  key_fact_recall: number | null;
}

export interface RecordedItem {
  question: string;
  response: AnswerResponse;
  expected?: RecordedExpectation;
}

/** Recorded answers shipped with the static demo (produced by scripts/export_demo_data.py). */
export interface RecordedAnswers {
  recorded_at: string;
  note: string;
  items: RecordedItem[];
}

export function findRecorded(recorded: RecordedAnswers, question: string): RecordedItem | undefined {
  const wanted = question.trim().toLowerCase();
  return recorded.items.find((item) => item.question.trim().toLowerCase() === wanted);
}

const STATUS_WORDS: Record<AnswerStatus, string> = {
  answered: "It answered",
  insufficient_evidence: "It refused for lack of evidence",
  invalid_output: "It returned no usable answer",
};

/**
 * Compares a recorded reply with the dataset's expected behaviour, using the same rule as the
 * answer evaluation. `matches` is null when the rule needs more than the status: a conflict
 * question counts only if the answer cites both conflicting sources, which is not checked here.
 */
export function compareWithExpected(
  expected: RecordedExpectation,
  status: AnswerStatus,
): { matches: boolean | null; summary: string } {
  const did = STATUS_WORDS[status];
  switch (expected.behavior) {
    case "answer":
      return status === "answered"
        ? { matches: true, summary: "It answered, as the dataset expects. Compare the wording with the reference answer." }
        : { matches: false, summary: `${did}, but the dataset expects an answer.` };
    case "abstain":
      return status === "insufficient_evidence"
        ? { matches: true, summary: "It refused, as the dataset expects." }
        : { matches: false, summary: `${did}, but the dataset expects a refusal.` };
    case "flag_conflict":
      return status === "answered"
        ? {
            matches: null,
            summary: "It answered. The dataset expects the answer to say that the sources disagree; compare with the reference.",
          }
        : { matches: false, summary: `${did}, but the dataset expects an answer that points out the conflict.` };
    default:
      return { matches: null, summary: `${did}. The dataset accepts an answer or a clarifying question.` };
  }
}

export interface DemoManifest {
  exported_at: string;
  note: string;
}

const READ_ONLY = "This is a static demo with no backend. Run GroundedAI locally to upload documents.";

/**
 * Serves the static demo from JSON files under `public/demo/`. Nothing here is
 * generated at runtime: every answer was recorded from a real local run and is
 * replayed as-is, and the UI labels it as a recording.
 */
export class StaticDemoApi implements Api {
  readonly mode = "static-demo" as const;
  private readonly root: string;
  private answers: Promise<RecordedAnswers> | null = null;

  constructor(root = `${import.meta.env.BASE_URL}demo`) {
    this.root = root.replace(/\/$/, "");
  }

  private async load<T>(file: string): Promise<T> {
    const response = await fetch(`${this.root}/${file}`);
    if (!response.ok) throw new ApiError(response.status, "demo_data_missing", `Demo file ${file} is missing.`);
    return (await response.json()) as T;
  }

  recordedAnswers(): Promise<RecordedAnswers> {
    this.answers ??= this.load<RecordedAnswers>("answers.json");
    return this.answers;
  }

  manifest() {
    return this.load<DemoManifest>("manifest.json");
  }

  systemStatus() {
    return this.load<SystemStatus>("status.json");
  }
  listDocuments(offset: number, limit: number) {
    return this.load<Page<DocumentOut>>("documents.json").then((page) => ({
      ...page,
      items: page.items.slice(offset, offset + limit),
      offset,
      limit,
    }));
  }
  async listChunks(): Promise<Page<ChunkOut>> {
    throw new ApiError(400, "demo_read_only", "Chunk inspection needs the live backend.");
  }
  async uploadDocument(): Promise<DocumentOut> {
    throw new ApiError(400, "demo_read_only", READ_ONLY);
  }
  async deleteDocument(): Promise<void> {
    throw new ApiError(400, "demo_read_only", READ_ONLY);
  }
  async reingestDocument(): Promise<DocumentOut> {
    throw new ApiError(400, "demo_read_only", READ_ONLY);
  }
  async query(request: QueryRequest): Promise<AnswerResponse> {
    const match = findRecorded(await this.recordedAnswers(), request.question);
    if (!match) {
      throw new ApiError(
        404,
        "demo_not_recorded",
        "The static demo can only replay the recorded example questions. Run GroundedAI locally to ask your own.",
      );
    }
    return match.response;
  }
  metrics() {
    return this.load<OperationalMetrics>("metrics.json");
  }
  async listReports() {
    const index = await this.load<{ items: ReportListItem[] }>("reports/index.json");
    return index.items;
  }
  getReport(reportId: string) {
    return this.load<Report>(`reports/${encodeURIComponent(reportId)}.json`);
  }
  async startEvaluationRun(): Promise<EvaluationRun> {
    throw new ApiError(400, "demo_read_only", "Evaluation runs need the live backend.");
  }
  async getEvaluationRun(): Promise<EvaluationRun> {
    throw new ApiError(400, "demo_read_only", "Evaluation runs need the live backend.");
  }
}
