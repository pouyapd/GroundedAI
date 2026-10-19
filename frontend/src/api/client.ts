import type {
  AnswerResponse,
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

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId: string | null;
  readonly details: Record<string, unknown>;

  constructor(
    status: number,
    code: string,
    message: string,
    requestId: string | null = null,
    details: Record<string, unknown> = {},
  ) {
    super(message);
    this.status = status;
    this.code = code;
    this.requestId = requestId;
    this.details = details;
  }
}

export interface Api {
  readonly mode: "live" | "static-demo";
  systemStatus(): Promise<SystemStatus>;
  listDocuments(offset: number, limit: number): Promise<Page<DocumentOut>>;
  listChunks(documentId: string): Promise<Page<ChunkOut>>;
  uploadDocument(file: File, title?: string): Promise<DocumentOut>;
  deleteDocument(documentId: string): Promise<void>;
  reingestDocument(documentId: string): Promise<DocumentOut>;
  query(request: QueryRequest): Promise<AnswerResponse>;
  metrics(): Promise<OperationalMetrics>;
  listReports(): Promise<ReportListItem[]>;
  getReport(reportId: string): Promise<Report>;
  startEvaluationRun(): Promise<EvaluationRun>;
  getEvaluationRun(runId: string): Promise<EvaluationRun>;
}

const API_KEY_STORAGE = "groundedai.apiKey";

export function getStoredApiKey(): string {
  try {
    return sessionStorage.getItem(API_KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

export function storeApiKey(key: string): void {
  try {
    if (key) sessionStorage.setItem(API_KEY_STORAGE, key);
    else sessionStorage.removeItem(API_KEY_STORAGE);
  } catch {
    // Storage can be unavailable (private mode); the key then lasts for this page only.
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  const requestId = response.headers.get("x-request-id");
  try {
    const body = (await response.json()) as {
      error?: { code?: string; message?: string; details?: Record<string, unknown> };
    };
    if (body.error) {
      return new ApiError(
        response.status,
        body.error.code ?? "http_error",
        body.error.message ?? response.statusText,
        requestId,
        body.error.details ?? {},
      );
    }
  } catch {
    // Not JSON (for example a proxy error page).
  }
  return new ApiError(response.status, "http_error", `Request failed (${response.status}).`, requestId);
}

export class LiveApi implements Api {
  readonly mode = "live" as const;
  private readonly base: string;
  private readonly fetchImpl: typeof fetch;

  constructor(baseUrl = "", fetchImpl: typeof fetch = (...args) => fetch(...args)) {
    this.base = `${baseUrl.replace(/\/$/, "")}/api/v1`;
    this.fetchImpl = fetchImpl;
  }

  private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers);
    const key = getStoredApiKey();
    if (key) headers.set("X-API-Key", key);
    if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
    let response: Response;
    try {
      response = await this.fetchImpl(`${this.base}${path}`, { ...init, headers });
    } catch {
      throw new ApiError(0, "network_error", "The GroundedAI API is not reachable.");
    }
    if (!response.ok) throw await toApiError(response);
    if (response.status === 204) return undefined as T;
    return (await response.json()) as T;
  }

  systemStatus() {
    return this.request<SystemStatus>("/system/status");
  }
  listDocuments(offset: number, limit: number) {
    return this.request<Page<DocumentOut>>(`/documents?offset=${offset}&limit=${limit}`);
  }
  listChunks(documentId: string) {
    return this.request<Page<ChunkOut>>(`/documents/${documentId}/chunks?limit=50`);
  }
  uploadDocument(file: File, title?: string) {
    const form = new FormData();
    form.append("file", file);
    if (title) form.append("title", title);
    return this.request<DocumentOut>("/documents", { method: "POST", body: form });
  }
  deleteDocument(documentId: string) {
    return this.request<void>(`/documents/${documentId}`, { method: "DELETE" });
  }
  reingestDocument(documentId: string) {
    return this.request<DocumentOut>(`/documents/${documentId}/reingest`, { method: "POST" });
  }
  query(request: QueryRequest) {
    return this.request<AnswerResponse>("/query", { method: "POST", body: JSON.stringify(request) });
  }
  metrics() {
    return this.request<OperationalMetrics>("/metrics");
  }
  async listReports() {
    const body = await this.request<{ items: ReportListItem[] }>("/evaluations/reports");
    return body.items;
  }
  getReport(reportId: string) {
    return this.request<Report>(`/evaluations/reports/${encodeURIComponent(reportId)}`);
  }
  startEvaluationRun() {
    return this.request<EvaluationRun>("/evaluations/runs", { method: "POST", body: "{}" });
  }
  getEvaluationRun(runId: string) {
    return this.request<EvaluationRun>(`/evaluations/runs/${runId}`);
  }
}
