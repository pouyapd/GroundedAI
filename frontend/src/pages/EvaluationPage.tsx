import { useCallback, useEffect, useState } from "react";

import type { EvaluationRun, GenerationReport, Report, ReportListItem, RetrievalReport } from "../api/types";
import { Badge } from "../components/Badge";
import { Banner, EmptyState, Loading } from "../components/Banner";
import { HBarChart, RangeDotChart } from "../components/charts";
import { ErrorBanner } from "../components/ErrorBanner";
import { useApp } from "../context";
import { formatDate, formatMs, formatPercent } from "../lib/format";

type Selected = { source: "published"; id: string } | { source: "run"; run: EvaluationRun };

export function EvaluationPage() {
  const { api } = useApp();
  const [reports, setReports] = useState<ReportListItem[] | null>(null);
  const [selected, setSelected] = useState<Selected | null>(null);
  const [fetched, setFetched] = useState<{ id: string; report: Report } | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [run, setRun] = useState<EvaluationRun | null>(null);
  const [runError, setRunError] = useState<unknown>(null);

  useEffect(() => {
    api
      .listReports()
      .then((items) => {
        setReports(items);
        const first = items.find((item) => item.kind === "retrieval") ?? items[0];
        if (first) setSelected({ source: "published", id: first.id });
      })
      .catch(setError);
  }, [api]);

  const selectedId = selected?.source === "published" ? selected.id : null;
  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    api
      .getReport(selectedId)
      .then((value) => {
        if (!cancelled) setFetched({ id: selectedId, report: value });
      })
      .catch((caught: unknown) => {
        if (!cancelled) setError(caught);
      });
    return () => {
      cancelled = true;
    };
  }, [api, selectedId]);

  const report: Report | null =
    selected?.source === "run"
      ? (selected.run.report ?? null)
      : fetched && fetched.id === selectedId
        ? fetched.report
        : null;

  const pollRun = useCallback(
    async (runId: string) => {
      for (let attempt = 0; attempt < 300; attempt += 1) {
        const current = await api.getEvaluationRun(runId);
        setRun(current);
        if (current.status === "completed") {
          setSelected({ source: "run", run: current });
          return;
        }
        if (current.status === "failed") return;
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
      }
    },
    [api],
  );

  async function startRun() {
    setRunError(null);
    try {
      const created = await api.startEvaluationRun();
      setRun(created);
      await pollRun(created.id);
    } catch (caught) {
      setRunError(caught);
    }
  }

  const running = run !== null && (run.status === "queued" || run.status === "running");

  return (
    <>
      <div>
        <h1 className="page-title">Evaluation</h1>
        <p className="page-intro">
          Measured results on a version-controlled synthetic dataset. Every number on this page comes from a benchmark
          run; sample sizes, confidence intervals, hardware and timestamps are shown with it.
        </p>
      </div>

      <section className="card">
        <div className="controls-row">
          <div className="field" style={{ flex: 1, minWidth: 240 }}>
            <label htmlFor="report-select">Report</label>
            <select
              id="report-select"
              value={selected?.source === "published" ? selected.id : ""}
              onChange={(event) => setSelected({ source: "published", id: event.target.value })}
              disabled={!reports || reports.length === 0}
            >
              {selected?.source === "run" ? <option value="">Run on this instance ({selected.run.id})</option> : null}
              {(reports ?? []).map((item) => (
                <option key={item.id} value={item.id}>
                  {reportOptionLabel(item)}
                </option>
              ))}
            </select>
          </div>
          {api.mode === "live" ? (
            <button type="button" className="btn" onClick={() => void startRun()} disabled={running}>
              {running ? <span className="spinner" /> : null} Evaluate this instance
            </button>
          ) : null}
        </div>
        {api.mode === "live" ? (
          <p className="small muted" style={{ marginBottom: 0 }}>
            “Evaluate this instance” runs the retrieval benchmark with the server's current embedding and retrieval
            settings on the bundled dataset, in a temporary index. Your documents are not touched.
          </p>
        ) : null}
        {run && run.status === "failed" ? (
          <div style={{ marginTop: 12 }}>
            <Banner tone="critical" title="Evaluation run failed">
              {run.error_message}
            </Banner>
          </div>
        ) : null}
        {runError ? (
          <div style={{ marginTop: 12 }}>
            <ErrorBanner error={runError} />
          </div>
        ) : null}
      </section>

      {error ? <ErrorBanner error={error} /> : null}
      {reports && reports.length === 0 && selected === null ? (
        <section className="card">
          <EmptyState title="No published reports">
            <p className="small">
              Run <code>groundedai eval retrieval --config evaluation/configs/retrieval_offline.json</code> and copy the
              report into <code>evaluation/reports/published/</code>.
            </p>
          </EmptyState>
        </section>
      ) : null}
      {selected && !report && !error ? <Loading label="Loading report" /> : null}
      {report?.kind === "retrieval" ? (
        <RetrievalReportView report={report} origin={selected?.source === "run" ? "run" : "published"} />
      ) : null}
      {report?.kind === "generation" ? <GenerationReportView report={report} /> : null}
    </>
  );
}

function ReportHeader({ report, origin }: { report: Report; origin: "run" | "published" }) {
  const env = report.environment;
  return (
    <section className="card">
      <div className="card-header">
        <h2>{report.title}</h2>
        {report.measured ? <Badge tone="good">Measured</Badge> : <Badge tone="warning">Sample data</Badge>}
      </div>
      <div className="meta-row">
        <span>Generated {formatDate(report.generated_at)}</span>
        <span>
          Dataset {report.dataset.name} v{report.dataset.version}
        </span>
        <span>{origin === "run" ? "Run on this instance" : "Published report"}</span>
      </div>
      <p className="small muted" style={{ marginBottom: 0 }}>
        Hardware: {env.cpu ?? "unknown CPU"} ({env.logical_cpus ?? "?"} logical CPUs), {env.ram_gb ?? "?"} GB RAM, no GPU,
        power source {env.power_source ?? "unknown"}; Python {env.python} on {env.os}.
      </p>
    </section>
  );
}

function RetrievalReportView({ report, origin }: { report: RetrievalReport; origin: "run" | "published" }) {
  const [showTable, setShowTable] = useState(false);
  const k = report.settings.k_values.includes(5) ? 5 : Math.max(...report.settings.k_values);
  const metrics = [
    { key: `recall@${k}`, title: `Recall@${k}` },
    { key: "mrr", title: "MRR" },
    { key: `ndcg@${k}`, title: `nDCG@${k}` },
  ];
  const n = report.dataset.questions_evaluated;

  return (
    <>
      <ReportHeader report={report} origin={origin} />
      <section className="card">
        <div className="card-header">
          <h2>Retrieval quality by configuration</h2>
          <button type="button" className="btn btn-small" onClick={() => setShowTable((value) => !value)}>
            {showTable ? "Show charts" : "Show table"}
          </button>
        </div>
        <p className="small secondary" style={{ marginTop: 0 }}>
          Mean over {n} questions with evidence; whiskers show 95% bootstrap intervals (
          {report.settings.bootstrap.samples} resamples, seed {report.settings.bootstrap.seed}). Corpus:{" "}
          {report.dataset.documents} documents, {report.dataset.chunks} chunks of {report.settings.chunking.chunk_size}{" "}
          characters. Backend: {report.settings.backend}.
        </p>
        {showTable ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Configuration</th>
                  {Object.keys(report.configurations[0]?.metrics ?? {}).map((name) => (
                    <th key={name} className="num">
                      {name}
                    </th>
                  ))}
                  <th className="num">p50</th>
                  <th className="num">p95</th>
                </tr>
              </thead>
              <tbody>
                {report.configurations.map((config) => (
                  <tr key={config.name}>
                    <td>
                      <div style={{ fontWeight: 600 }}>{config.name}</div>
                      <div className="small muted">{config.description}</div>
                    </td>
                    {Object.entries(config.metrics).map(([name, value]) => (
                      <td key={name} className="num">
                        {value.mean.toFixed(3)}
                        {value.ci95 ? (
                          <div className="small muted">
                            [{value.ci95[0].toFixed(2)}, {value.ci95[1].toFixed(2)}]
                          </div>
                        ) : null}
                      </td>
                    ))}
                    <td className="num">{formatMs(config.latency_ms.p50)}</td>
                    <td className="num">{formatMs(config.latency_ms.p95)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="small-multiples">
            {metrics.map((metric) => (
              <HBarChart
                key={metric.key}
                title={metric.title}
                data={report.configurations.map((config) => ({
                  label: config.name,
                  value: config.metrics[metric.key]?.mean ?? 0,
                  low: config.metrics[metric.key]?.ci95?.[0] ?? null,
                  high: config.metrics[metric.key]?.ci95?.[1] ?? null,
                }))}
              />
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <h2>Retrieval latency per query</h2>
        <p className="small secondary" style={{ marginTop: 0 }}>
          Wall-clock time per query in one process, including query embedding and reranking ({n} queries per
          configuration).
        </p>
        <RangeDotChart
          title="Latency (ms)"
          data={report.configurations.map((config) => ({
            label: config.name,
            p50: config.latency_ms.p50 ?? 0,
            p95: config.latency_ms.p95 ?? 0,
          }))}
        />
      </section>

      <section className="card">
        <h2>Paired comparison with the baseline ({report.settings.baseline})</h2>
        <p className="small secondary" style={{ marginTop: 0 }}>
          Mean per-question difference with a paired 95% bootstrap interval. “No” means this dataset cannot separate the
          configuration from the baseline.
        </p>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Configuration</th>
                <th>Metric</th>
                <th className="num">Difference</th>
                <th className="num">95% interval</th>
                <th>Distinguishable</th>
              </tr>
            </thead>
            <tbody>
              {report.comparisons.map((row) => (
                <tr key={`${row.configuration}-${row.metric}`}>
                  <td>{row.configuration}</td>
                  <td>{row.metric}</td>
                  <td className="num">{row.mean_diff >= 0 ? "+" : ""}{row.mean_diff.toFixed(3)}</td>
                  <td className="num">
                    [{row.ci95_low.toFixed(3)}, {row.ci95_high.toFixed(3)}]
                  </td>
                  <td>{row.distinguishable ? <Badge tone="info">Yes</Badge> : <Badge tone="neutral">No</Badge>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <Limitations items={report.limitations} />
    </>
  );
}

const SUMMARY_FIELDS: { key: keyof GenerationReport["summary"]; label: string }[] = [
  { key: "behavior_accuracy", label: "Correct behaviour" },
  { key: "abstention_recall", label: "Correct refusals" },
  { key: "citation_validity", label: "Citation validity" },
  { key: "evidence_coverage", label: "Evidence coverage" },
  { key: "key_fact_recall", label: "Key-fact recall" },
  { key: "injection_resistance", label: "Injection resistance" },
];

// Shown only for reports that contain them (scoring version 3, from 2026-10-13).
const V3_FIELDS: { key: keyof GenerationReport["summary"]; label: string }[] = [
  { key: "answer_rate_answerable", label: "Answerable questions answered" },
  { key: "incorrect_refusal_rate", label: "Wrong refusals (answerable)" },
  { key: "citation_format_compliance", label: "Citation format followed" },
  { key: "evidence_support_rate", label: "Cited sentences supported" },
];

const V4_FIELDS: { key: keyof GenerationReport["summary"]; label: string }[] = [
  { key: "claim_match_rate", label: "Statements found in a passage" },
  { key: "cited_claim_precision", label: "Model citations pointing to the right passage" },
  { key: "answers_verified_rate", label: "Answers with every statement found" },
  { key: "key_fact_attribution", label: "Key facts traced to the gold evidence" },
  { key: "conflict_surfaced_rate", label: "Conflict questions: both sides shown" },
  { key: "false_conflict_rate", label: "Other questions: false disagreement" },
];

/** Report titles before 2026-10-13 do not name the prompt, so two runs could look identical. */
export function reportOptionLabel(item: ReportListItem): string {
  const prompt = item.prompt_version && !item.title.includes(item.prompt_version) ? ` · ${item.prompt_version}` : "";
  return `${item.title}${prompt} · ${formatDate(item.generated_at)}`;
}

export function promptLabel(report: Pick<GenerationReport, "prompt_version" | "run_config">): string {
  const version = report.prompt_version ?? "answer-v1";
  const fingerprint = report.run_config?.prompt_fingerprint;
  return fingerprint ? `${version} (${fingerprint})` : version;
}

/** Replies that were neither answers, abstentions nor errors (e.g. citation markers only). */
export function invalidCount(item: { n: number; answered: number; abstained: number; errors: number }): number {
  return Math.max(0, item.n - item.answered - item.abstained - item.errors);
}

function GenerationReportView({ report }: { report: GenerationReport }) {
  const summary = report.summary;
  return (
    <>
      <ReportHeader report={report} origin="published" />
      {report.revalidation ? (
        <Banner tone="info" title="Recorded outputs re-checked: the model was not run">
          The raw outputs recorded in report {report.revalidation.source_report_id} were passed through the current
          checks. Latency and token counts are those of the original run.
          {report.revalidation.model_input_changed.length > 0
            ? ` For ${report.revalidation.model_input_changed.join(", ")} the current pipeline would send the model a different note, so a new run could answer differently.`
            : ""}
        </Banner>
      ) : null}
      {!report.provider.is_llm ? (
        <Banner tone="info" title="Baseline, not a language model">
          These results describe the deterministic extractive demo generator. They show how the pipeline and the checks
          behave; they are not LLM quality numbers.
        </Banner>
      ) : null}
      <section className="card">
        <div className="card-header">
          <h2>Answer quality</h2>
          <span className="small muted">
            {report.provider.name} · {report.provider.model}
            {report.provider.revision ? ` (${report.provider.revision})` : ""} · prompt {promptLabel(report)}
          </span>
        </div>
        <p className="small secondary" style={{ marginTop: 0 }}>
          {summary.n} questions: {summary.answered} answered, {summary.abstained} abstained, {invalidCount(summary)}{" "}
          invalid output, {summary.errors} errors. Latency p50 {formatMs(summary.latency_ms.p50)}, p95{" "}
          {formatMs(summary.latency_ms.p95)}.
        </p>
        <div className="stats">
          {SUMMARY_FIELDS.map((field) => (
            <div className="stat" key={field.key}>
              <div className="stat-label">{field.label}</div>
              <div className="stat-value">{formatPercent(summary[field.key] as number | null)}</div>
            </div>
          ))}
        </div>
        {V3_FIELDS.some((field) => summary[field.key] !== undefined) ? (
          <div className="stats" style={{ marginTop: 12 }}>
            {V3_FIELDS.map((field) => (
              <div className="stat" key={field.key}>
                <div className="stat-label">{field.label}</div>
                <div className="stat-value">{formatPercent((summary[field.key] ?? null) as number | null)}</div>
              </div>
            ))}
          </div>
        ) : null}
        {V4_FIELDS.some((field) => summary[field.key] !== undefined) ? (
          <>
            <h3 style={{ marginTop: 16 }}>Claim and disagreement checks</h3>
            <p className="small secondary" style={{ marginTop: 0 }}>
              Every statement of an answer compared with every retrieved passage (words and figures, not meaning);
              key facts traced to the dataset's evidence; disagreeing passages reported on the two conflict
              questions and wrongly reported elsewhere.
            </p>
            <div className="stats">
              {V4_FIELDS.map((field) => (
                <div className="stat" key={field.key}>
                  <div className="stat-label">{field.label}</div>
                  <div className="stat-value">{formatPercent((summary[field.key] ?? null) as number | null)}</div>
                </div>
              ))}
            </div>
          </>
        ) : null}
      </section>
      <section className="card">
        <h2>By question category</h2>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th className="num">n</th>
                <th className="num">Answered</th>
                <th className="num">Abstained</th>
                <th className="num">Invalid</th>
                <th className="num">Correct behaviour</th>
                <th className="num">Evidence coverage</th>
                <th className="num">Key facts</th>
                <th className="num">Injection resisted</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(report.by_category).map(([category, item]) => (
                <tr key={category}>
                  <td>{category.replaceAll("_", " ")}</td>
                  <td className="num">{item.n}</td>
                  <td className="num">{item.answered}</td>
                  <td className="num">{item.abstained}</td>
                  <td className="num">{invalidCount(item)}</td>
                  <td className="num">{formatPercent(item.behavior_accuracy)}</td>
                  <td className="num">{formatPercent(item.evidence_coverage)}</td>
                  <td className="num">{formatPercent(item.key_fact_recall)}</td>
                  <td className="num">{formatPercent(item.injection_resistance)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
      <Limitations items={report.limitations} />
    </>
  );
}

function Limitations({ items }: { items: string[] }) {
  return (
    <section className="card">
      <h2>Limitations of this report</h2>
      <ul className="warnings">
        {items.map((item) => (
          <li key={item}>{item}</li>
        ))}
      </ul>
    </section>
  );
}
