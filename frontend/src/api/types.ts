// Mirrors the backend Pydantic schemas (backend/groundedai/schemas).

export type DocumentStatus = "pending" | "processing" | "ready" | "failed";

export interface DocumentOut {
  id: string;
  title: string;
  filename: string;
  media_type: string;
  size_bytes: number;
  sha256: string;
  status: DocumentStatus;
  error_code: string | null;
  error_message: string | null;
  page_count: number | null;
  char_count: number | null;
  chunk_count: number;
  embedding_model: string | null;
  attempts: number;
  created_at: string;
  updated_at: string;
}

export interface Page<T> {
  items: T[];
  total: number;
  offset: number;
  limit: number;
}

export interface ChunkOut {
  id: string;
  chunk_index: number;
  text: string;
  page_start: number | null;
  page_end: number | null;
}

export type RetrievalMode = "lexical" | "dense" | "hybrid";

export interface QueryRequest {
  question: string;
  top_k?: number;
  document_ids?: string[];
  retrieval_mode?: RetrievalMode;
  rerank?: boolean;
  include_diagnostics?: boolean;
}

export interface ChunkScores {
  lexical: number | null;
  lexical_rank: number | null;
  dense: number | null;
  dense_rank: number | null;
  fused: number | null;
  rerank: number | null;
}

export interface SourceOut {
  label: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_start: number | null;
  page_end: number | null;
  excerpt: string;
  text: string;
  cited: boolean;
  truncated: boolean;
  injection_flags: string[];
  scores: ChunkScores;
  /** Indices into `claims` of the claims this passage is evidence for (absent before 2026-10-14). */
  supports_claims?: number[];
}

export type SupportLabel = "supported" | "partial" | "unsupported" | "uncited";

export interface SentenceOut {
  text: string;
  citations: string[];
  support: SupportLabel;
  token_recall: number | null;
  unsupported_numbers: string[];
}

export interface RetrievalDiagnostics {
  mode: string;
  fusion: string | null;
  reranker: string | null;
  embedding_model: string | null;
  lexical_candidates: number;
  dense_candidates: number;
  fused_candidates: number;
  duplicates_removed: number;
  timings_ms: Record<string, number>;
  warnings: string[];
}

export type AnswerStatus = "answered" | "insufficient_evidence" | "invalid_output";
export type Grounding = "grounded" | "partially_grounded" | "ungrounded";

export type ClaimStatus = "supported" | "wrong_source" | "added_source" | "partial" | "unsupported";
export type Verdict = "verified" | "partially_verified" | "unverified" | "not_applicable";

export interface EvidenceOut {
  label: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_start: number | null;
  page_end: number | null;
  quote: string;
  quote_start: number;
  quote_end: number;
}

export interface ClaimOut {
  text: string;
  citations: string[];
  status: ClaimStatus;
  evidence: EvidenceOut[];
  missing_figures: string[];
  word_overlap: number;
  flags: string[];
  /** claims-v2 only. */
  citation_status?: "correct" | "wrong" | "partly_wrong" | "invalid" | "unconfirmed" | "none" | null;
  invalid_citations?: string[];
}

export type EvidenceVerdict = "all_found" | "some_uncertain" | "some_not_found" | "none_found" | "not_applicable";
export type CitationVerdict = "all_correct" | "some_wrong" | "some_unconfirmed" | "none_given" | "not_applicable";
export type AddressesQuestion = "yes" | "no" | "declined" | "not_checked";

export interface VerificationOut {
  verdict: Verdict;
  claims: number;
  supported: number;
  wrong_source: number;
  added_source: number;
  partial: number;
  unsupported: number;
  notes: number;
  references_standard: number;
  references_variant: number;
  method: string;
  /** Absent in answers recorded before claims-v2. */
  version?: string;
  evidence?: EvidenceVerdict | null;
  citations?: CitationVerdict | null;
  addresses_question?: AddressesQuestion | null;
  expected_answer_type?: string | null;
}

export interface ConflictStatementOut {
  label: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  page_start: number | null;
  page_end: number | null;
  quote: string;
  quote_start: number;
  quote_end: number;
  values: string[];
  denies_requirement: boolean;
  /** Absent in recordings made before 2026-10-14's fix. */
  affirms_requirement?: boolean;
  dates: string[];
}

export interface ConflictOut {
  kind: "conflict" | "different_dates";
  statements: ConflictStatementOut[];
  explanation: string;
  answer_reports: string[];
}

export interface AnswerResponse {
  status: AnswerStatus;
  answer: string | null;
  abstention_reason: string | null;
  grounding: Grounding | null;
  sources: SourceOut[];
  sentences: SentenceOut[];
  citation_check: {
    cited: string[];
    valid: string[];
    invalid: string[];
    all_valid: boolean;
    sentences_total: number;
    sentences_cited: number;
  };
  warnings: string[];
  provider: { name: string; model: string; is_llm: boolean };
  timings: { retrieval_ms: number | null; generation_ms: number | null; total_ms: number };
  usage: {
    prompt_tokens: number | null;
    completion_tokens: number | null;
    total_tokens: number | null;
    estimated_cost_usd: number | null;
    cost_note: string;
  };
  retrieval: RetrievalDiagnostics | null;
  // Claim and disagreement checks (2026-10-14); absent in older recordings.
  claims?: ClaimOut[];
  verification?: VerificationOut | null;
  conflicts?: ConflictOut[];
  unverified_answer?: string | null;
}

export interface ComponentStatus {
  provider: string;
  model: string | null;
  is_llm: boolean | null;
  available: boolean | null;
  detail: string;
}

export interface SystemStatus {
  version: string;
  environment: string;
  demo_mode: boolean;
  auth_required: boolean;
  database_backend: string;
  embedding: ComponentStatus;
  generation: ComponentStatus;
  reranker: ComponentStatus;
  retrieval_defaults: Record<string, string | number>;
  limits: {
    max_upload_bytes: number;
    allowed_extensions: string[];
    max_pdf_pages: number;
    max_question_chars: number;
    rate_limit_query_per_minute: number | null;
  };
}

export interface LatencySummary {
  n: number;
  mean: number | null;
  p50: number | null;
  p95: number | null;
  max: number | null;
}

export interface OperationalMetrics {
  uptime_s: number;
  documents_by_status: Record<string, number>;
  documents_total: number;
  chunks_total: number;
  queries_total: number;
  queries_window_hours: number;
  queries_in_window: number;
  queries_by_status: Record<string, number>;
  abstention_rate: number | null;
  citation_valid_rate: number | null;
  latency_ms: Record<string, LatencySummary>;
  hourly: { hour: string; queries: number; answered: number; abstained: number; failed: number }[];
  http_requests_total: number;
  http_server_errors_total: number;
}

export interface ReportListItem {
  id: string;
  kind: "retrieval" | "generation" | string;
  title: string;
  generated_at: string | null;
  dataset: string | null;
  n_configurations: number;
  prompt_version?: string | null;
}

export interface MetricValue {
  mean: number;
  ci95: [number, number] | null;
  n: number;
}

export interface Environment {
  os?: string;
  python?: string;
  cpu?: string;
  logical_cpus?: number;
  ram_gb?: number | null;
  power_source?: string;
}

export interface RetrievalConfigResult {
  name: string;
  description: string;
  mode: string;
  embedding_model: string | null;
  fusion: string | null;
  reranker: string | null;
  metrics: Record<string, MetricValue>;
  latency_ms: LatencySummary;
}

export interface RetrievalReport {
  report_id: string;
  kind: "retrieval";
  title: string;
  generated_at: string;
  measured: boolean;
  dataset: {
    name: string;
    version: string;
    fingerprint: string;
    documents: number;
    questions_evaluated: number;
    chunks: number;
  };
  settings: {
    k_values: number[];
    baseline: string;
    backend: string;
    chunking: { chunk_size: number; chunk_overlap: number };
    bootstrap: { samples: number; seed: number };
  };
  environment: Environment;
  configurations: RetrievalConfigResult[];
  comparisons: {
    configuration: string;
    baseline: string;
    metric: string;
    mean_diff: number;
    ci95_low: number;
    ci95_high: number;
    distinguishable: boolean;
  }[];
  limitations: string[];
}

export interface GenerationSummary {
  n: number;
  answered: number;
  abstained: number;
  errors: number;
  failure_rate: number | null;
  behavior_accuracy: number | null;
  abstention_recall: number | null;
  abstention_precision: number | null;
  citation_validity: number | null;
  grounded_rate: number | null;
  evidence_coverage: number | null;
  key_fact_recall: number | null;
  injection_resistance: number | null;
  latency_ms: LatencySummary;
  // Scoring version 3 (2026-10-13); absent in older reports.
  answer_rate_answerable?: number | null;
  incorrect_refusal_rate?: number | null;
  citation_format_compliance?: number | null;
  evidence_support_rate?: number | null;
  // Scoring version 4 (2026-10-14); absent in older reports.
  claims?: number;
  claim_match_rate?: number | null;
  cited_claim_precision?: number | null;
  wrong_source_rate?: number | null;
  unsupported_claim_rate?: number | null;
  answers_verified_rate?: number | null;
  key_fact_attribution?: number | null;
  conflict_surfaced_rate?: number | null;
  false_conflict_rate?: number | null;
}

export interface GenerationReport {
  report_id: string;
  kind: "generation";
  title: string;
  generated_at: string;
  measured: boolean;
  /** Absent in reports written before 2026-10-13, which all used answer-v1. */
  prompt_version?: string;
  run_config?: { prompt_fingerprint?: string; conflict_check?: string; claim_check?: string };
  /** Present when recorded model outputs were replayed through later checks (no model run). */
  revalidation?: {
    source_report_id: string;
    source_generated_at: string;
    model_run: false;
    model_input_changed: string[];
    note: string;
  };
  dataset: { name: string; version: string; questions_evaluated: number };
  provider: { name: string; model: string; is_llm: boolean; revision?: string | null; runtime?: string };
  retrieval: { mode: string; top_k: number; embedding_model: string; reranker: string | null };
  environment: Environment;
  summary: GenerationSummary;
  by_category: Record<string, GenerationSummary>;
  limitations: string[];
}

export type Report = RetrievalReport | GenerationReport;

export interface EvaluationRun {
  id: string;
  kind: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at: string;
  finished_at: string | null;
  summary: Record<string, Record<string, number>> | null;
  error_message: string | null;
  report?: RetrievalReport | null;
}
