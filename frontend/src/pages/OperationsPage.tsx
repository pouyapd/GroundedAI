import { useCallback, useEffect, useState } from "react";

import type { ComponentStatus, OperationalMetrics } from "../api/types";
import { Badge } from "../components/Badge";
import { Banner, Loading } from "../components/Banner";
import { ColumnChart } from "../components/charts";
import { ErrorBanner } from "../components/ErrorBanner";
import { useApp } from "../context";
import { formatDuration, formatMs, formatPercent } from "../lib/format";

export function OperationsPage() {
  const { api, status } = useApp();
  const [metrics, setMetrics] = useState<OperationalMetrics | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [showTable, setShowTable] = useState(false);

  const load = useCallback(() => {
    api
      .metrics()
      .then((value) => {
        setMetrics(value);
        setError(null);
      })
      .catch(setError);
  }, [api]);

  useEffect(() => {
    load();
    if (api.mode !== "live") return;
    const timer = window.setInterval(load, 15_000);
    return () => window.clearInterval(timer);
  }, [api, load]);

  return (
    <>
      <div className="card-header" style={{ marginBottom: 0 }}>
        <div>
          <h1 className="page-title">Operations</h1>
          <p className="page-intro">
            Live counters from this server: index size, question outcomes and latency. Questions are logged as outcomes
            and timings only; the question text is not stored.
          </p>
        </div>
        {api.mode === "live" ? (
          <button type="button" className="btn btn-small" onClick={load}>
            Refresh
          </button>
        ) : null}
      </div>

      {api.mode === "static-demo" ? (
        <Banner tone="info" title="Snapshot, not live">
          These values were recorded when the static demo was exported. A running server shows live values here.
        </Banner>
      ) : null}
      {error ? <ErrorBanner error={error} onRetry={load} /> : null}
      {!metrics && !error ? <Loading /> : null}

      {metrics ? (
        <>
          <div className="stats">
            <Stat label="Documents ready" value={`${metrics.documents_by_status.ready ?? 0}`} note={`${metrics.documents_total} total, ${metrics.documents_by_status.failed ?? 0} failed`} />
            <Stat label="Indexed chunks" value={metrics.chunks_total.toLocaleString()} />
            <Stat
              label={`Questions, last ${metrics.queries_window_hours} h`}
              value={metrics.queries_in_window.toLocaleString()}
              note={`${metrics.queries_total.toLocaleString()} since the database was created`}
            />
            <Stat label="Abstention rate" value={formatPercent(metrics.abstention_rate)} note="answered + abstained questions" />
            <Stat label="Answers with valid citations" value={formatPercent(metrics.citation_valid_rate)} />
            <Stat
              label="Server errors (5xx)"
              value={metrics.http_server_errors_total.toLocaleString()}
              note={`of ${metrics.http_requests_total.toLocaleString()} requests · up ${formatDuration(metrics.uptime_s)}`}
            />
          </div>

          <section className="card">
            <div className="card-header">
              <h2>Questions per hour</h2>
              <button type="button" className="btn btn-small" onClick={() => setShowTable((value) => !value)}>
                {showTable ? "Show chart" : "Show table"}
              </button>
            </div>
            {metrics.queries_in_window === 0 ? (
              <p className="secondary small">No questions in this window yet.</p>
            ) : showTable ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Hour (UTC)</th>
                      <th className="num">Questions</th>
                      <th className="num">Answered</th>
                      <th className="num">Abstained</th>
                      <th className="num">Failed</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.hourly
                      .filter((bucket) => bucket.queries > 0)
                      .map((bucket) => (
                        <tr key={bucket.hour}>
                          <td>{bucket.hour.slice(0, 16).replace("T", " ")}</td>
                          <td className="num">{bucket.queries}</td>
                          <td className="num">{bucket.answered}</td>
                          <td className="num">{bucket.abstained}</td>
                          <td className="num">{bucket.failed}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <ColumnChart
                title="Questions per hour"
                data={metrics.hourly.map((bucket) => ({
                  label: `${bucket.hour.slice(11, 13)}:00`,
                  value: bucket.queries,
                  detail: `${bucket.queries} questions · ${bucket.answered} answered · ${bucket.abstained} abstained · ${bucket.failed} failed`,
                }))}
              />
            )}
          </section>

          <section className="card">
            <h2>Latency of answered questions</h2>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Stage</th>
                    <th className="num">n</th>
                    <th className="num">mean</th>
                    <th className="num">p50</th>
                    <th className="num">p95</th>
                    <th className="num">max</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(metrics.latency_ms).map(([stage, summary]) => (
                    <tr key={stage}>
                      <td>{stage}</td>
                      <td className="num">{summary.n}</td>
                      <td className="num">{formatMs(summary.mean)}</td>
                      <td className="num">{formatMs(summary.p50)}</td>
                      <td className="num">{formatMs(summary.p95)}</td>
                      <td className="num">{formatMs(summary.max)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      ) : null}

      {status ? (
        <section className="card">
          <h2>Model providers</h2>
          <div className="grid-2">
            <Provider title="Generation" component={status.generation} />
            <Provider title="Embeddings" component={status.embedding} />
            <Provider title="Reranker" component={status.reranker} />
            <div>
              <h3>Storage</h3>
              <p className="small secondary" style={{ margin: 0 }}>
                {status.database_backend === "postgresql"
                  ? "PostgreSQL with full-text search and pgvector"
                  : "SQLite with FTS5 full-text search and exact vector search"}
              </p>
            </div>
          </div>
        </section>
      ) : null}
    </>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {note ? <div className="stat-note">{note}</div> : null}
    </div>
  );
}

function Provider({ title, component }: { title: string; component: ComponentStatus }) {
  const badge =
    component.available === null ? (
      <Badge tone="neutral">Off</Badge>
    ) : component.available ? (
      <Badge tone="good">Available</Badge>
    ) : (
      <Badge tone="critical">Unavailable</Badge>
    );
  return (
    <div>
      <div className="card-header" style={{ marginBottom: 4 }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        {badge}
      </div>
      <p className="small" style={{ margin: 0 }}>
        <code>{component.provider}</code>
        {component.model ? (
          <>
            {" "}
            · <code>{component.model}</code>
          </>
        ) : null}
      </p>
      <p className="small secondary" style={{ margin: "4px 0 0" }}>
        {component.detail}
      </p>
    </div>
  );
}
