import { useEffect, useRef, useState, type ReactNode } from "react";

/*
 * Small hand-written SVG charts (no chart library).
 * Mark specs: bars <= 20px thick with a 4px rounded data end, hairline solid grid,
 * text in text colours (never the series colour), per-mark hover tooltips,
 * and a table view available next to every chart.
 */

interface TooltipState {
  x: number;
  y: number;
  content: ReactNode;
}

/**
 * Width of the chart's container. Charts lay themselves out for the space they have
 * (narrower label column on phones) instead of scaling the whole drawing, so text stays
 * at its real size.
 */
function useContainerWidth(fallback: number) {
  const ref = useRef<HTMLElement>(null);
  const [width, setWidth] = useState(fallback);
  useEffect(() => {
    const element = ref.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver((entries) => {
      const measured = Math.floor(entries[0]?.contentRect.width ?? 0);
      if (measured > 0) setWidth(measured);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);
  return [ref, width] as const;
}

function truncate(label: string, max: number): string {
  return label.length > max ? `${label.slice(0, max - 1)}…` : label;
}

function useTooltip() {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const node = tooltip ? (
    <div className="tooltip" role="tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
      {tooltip.content}
    </div>
  ) : null;
  return { show: setTooltip, hide: () => setTooltip(null), node };
}

/** Rectangle with a rounded right end and a square left end (anchored to the baseline). */
function barPath(x: number, y: number, width: number, height: number, radius = 4): string {
  const r = Math.min(radius, width, height / 2);
  if (width <= 0) return "";
  return [
    `M${x},${y}`,
    `H${x + width - r}`,
    `Q${x + width},${y} ${x + width},${y + r}`,
    `V${y + height - r}`,
    `Q${x + width},${y + height} ${x + width - r},${y + height}`,
    `H${x}`,
    "Z",
  ].join(" ");
}

export interface HBarDatum {
  label: string;
  value: number;
  low?: number | null;
  high?: number | null;
  emphasis?: boolean;
}

/** Horizontal bars on a 0–1 scale with optional confidence-interval whiskers. */
export function HBarChart({
  data,
  title,
  format = (value: number) => value.toFixed(3),
}: {
  data: HBarDatum[];
  title: string;
  format?: (value: number) => string;
}) {
  const { show, hide, node } = useTooltip();
  const [ref, available] = useContainerWidth(538);
  const width = Math.max(280, Math.min(available, 560));
  const compact = width < 440;
  const labelWidth = compact ? 120 : 190;
  const valueWidth = 48;
  const plotWidth = width - labelWidth - valueWidth;
  const rowHeight = 30;
  const barHeight = 16;
  const top = 8;
  const height = top + data.length * rowHeight + 22;
  const x = (value: number) => labelWidth + Math.max(0, Math.min(1, value)) * plotWidth;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <figure ref={ref} className="chart" style={{ margin: 0 }}>
      <figcaption className="small" style={{ fontWeight: 600, marginBottom: 6 }}>
        {title}
      </figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={title}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="grid-line" x1={x(tick)} x2={x(tick)} y1={top} y2={height - 20} />
            <text className="tick" x={x(tick)} y={height - 6} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        <line className="baseline" x1={labelWidth} x2={labelWidth} y1={top} y2={height - 20} />
        {data.map((datum, index) => {
          const y = top + index * rowHeight + (rowHeight - barHeight) / 2;
          const barWidth = x(datum.value) - labelWidth;
          const hasCi = datum.low !== null && datum.low !== undefined && datum.high !== null && datum.high !== undefined;
          const tooltip = (
            <>
              <strong>{datum.label}</strong>
              <br />
              {format(datum.value)}
              {hasCi ? ` (95% CI ${format(datum.low as number)}–${format(datum.high as number)})` : ""}
            </>
          );
          return (
            <g
              key={datum.label}
              onMouseMove={(event) => {
                const bounds = (event.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                show({ x: event.clientX - bounds.left, y: event.clientY - bounds.top, content: tooltip });
              }}
              onMouseLeave={hide}
            >
              <rect x={0} y={y - 6} width={width} height={barHeight + 12} fill="transparent" />
              <text x={labelWidth - 10} y={y + barHeight / 2 + 4} textAnchor="end" fontSize={12}>
                {truncate(datum.label, compact ? 17 : 28)}
              </text>
              <path
                d={barPath(labelWidth, y, barWidth, barHeight)}
                fill="var(--series-1)"
                opacity={datum.emphasis === false ? 0.45 : 1}
              />
              {hasCi ? (
                <g stroke="var(--text-2)" strokeWidth={1.5}>
                  <line x1={x(datum.low as number)} x2={x(datum.high as number)} y1={y + barHeight / 2} y2={y + barHeight / 2} />
                  <line x1={x(datum.low as number)} x2={x(datum.low as number)} y1={y + 3} y2={y + barHeight - 3} />
                  <line x1={x(datum.high as number)} x2={x(datum.high as number)} y1={y + 3} y2={y + barHeight - 3} />
                </g>
              ) : null}
              <text x={labelWidth + plotWidth + 6} y={y + barHeight / 2 + 4} fontSize={12} fontWeight={600}>
                {format(datum.value)}
              </text>
            </g>
          );
        })}
      </svg>
      {node}
    </figure>
  );
}

export interface RangeDatum {
  label: string;
  p50: number;
  p95: number;
}

/** p50 and p95 latency per configuration as two dots joined by a line, on one axis. */
export function RangeDotChart({ data, title, unit = "ms" }: { data: RangeDatum[]; title: string; unit?: string }) {
  const { show, hide, node } = useTooltip();
  const [ref, available] = useContainerWidth(550);
  const width = Math.max(280, Math.min(available, 600));
  const compact = width < 440;
  const labelWidth = compact ? 120 : 190;
  const plotWidth = width - labelWidth - 20;
  const rowHeight = 28;
  const top = 8;
  const height = top + data.length * rowHeight + 22;
  const max = Math.max(1, ...data.map((d) => d.p95));
  const niceMax = niceCeiling(max);
  const x = (value: number) => labelWidth + (value / niceMax) * plotWidth;
  const ticks = [0, niceMax / 4, niceMax / 2, (3 * niceMax) / 4, niceMax];

  return (
    <figure ref={ref} className="chart" style={{ margin: 0 }}>
      <figcaption className="small" style={{ fontWeight: 600, marginBottom: 6 }}>
        {title}
      </figcaption>
      <div className="legend" style={{ marginBottom: 4 }}>
        <span>
          <span className="legend-key" style={{ background: "var(--series-1)" }} />
          median (p50)
        </span>
        <span>
          <span className="legend-key" style={{ background: "var(--series-2)" }} />
          95th percentile (p95)
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={title}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="grid-line" x1={x(tick)} x2={x(tick)} y1={top} y2={height - 20} />
            <text className="tick" x={x(tick)} y={height - 6} textAnchor="middle">
              {formatTick(tick)}
            </text>
          </g>
        ))}
        {data.map((datum, index) => {
          const y = top + index * rowHeight + rowHeight / 2;
          return (
            <g
              key={datum.label}
              onMouseMove={(event) => {
                const bounds = (event.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                show({
                  x: event.clientX - bounds.left,
                  y: event.clientY - bounds.top,
                  content: (
                    <>
                      <strong>{datum.label}</strong>
                      <br />
                      p50 {datum.p50.toFixed(1)} {unit} · p95 {datum.p95.toFixed(1)} {unit}
                    </>
                  ),
                });
              }}
              onMouseLeave={hide}
            >
              <rect x={0} y={y - rowHeight / 2} width={width} height={rowHeight} fill="transparent" />
              <text x={labelWidth - 10} y={y + 4} textAnchor="end" fontSize={12}>
                {truncate(datum.label, compact ? 17 : 28)}
              </text>
              <line x1={x(datum.p50)} x2={x(datum.p95)} y1={y} y2={y} stroke="var(--axis)" strokeWidth={2} />
              <circle cx={x(datum.p50)} cy={y} r={5} fill="var(--series-1)" stroke="var(--surface)" strokeWidth={2} />
              <circle cx={x(datum.p95)} cy={y} r={5} fill="var(--series-2)" stroke="var(--surface)" strokeWidth={2} />
            </g>
          );
        })}
      </svg>
      {node}
    </figure>
  );
}

export interface ColumnDatum {
  label: string;
  value: number;
  detail?: ReactNode;
}

/** Vertical columns (one series), e.g. queries per hour. */
export function ColumnChart({ data, title }: { data: ColumnDatum[]; title: string }) {
  const { show, hide, node } = useTooltip();
  const [ref, available] = useContainerWidth(720);
  const width = Math.max(280, Math.min(available, 1000));
  const height = 180;
  const left = 32;
  const bottom = 22;
  const top = 8;
  const plotHeight = height - top - bottom;
  const slot = (width - left) / Math.max(1, data.length);
  const barWidth = Math.min(20, slot - 2);
  const max = niceCeiling(Math.max(1, ...data.map((d) => d.value)));
  const y = (value: number) => top + plotHeight - (value / max) * plotHeight;
  const ticks = [0, max / 2, max];

  return (
    <figure ref={ref} className="chart" style={{ margin: 0 }}>
      <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height} role="img" aria-label={title}>
        {ticks.map((tick) => (
          <g key={tick}>
            <line className="grid-line" x1={left} x2={width} y1={y(tick)} y2={y(tick)} />
            <text className="tick" x={left - 6} y={y(tick) + 4} textAnchor="end">
              {formatTick(tick)}
            </text>
          </g>
        ))}
        <line className="baseline" x1={left} x2={width} y1={y(0)} y2={y(0)} />
        {data.map((datum, index) => {
          const cx = left + slot * index + slot / 2;
          const barTop = y(datum.value);
          const barHeight = y(0) - barTop;
          const r = Math.min(4, barHeight / 2, barWidth / 2);
          const x0 = cx - barWidth / 2;
          const path =
            barHeight > 0
              ? `M${x0},${y(0)} V${barTop + r} Q${x0},${barTop} ${x0 + r},${barTop} H${x0 + barWidth - r} Q${x0 + barWidth},${barTop} ${x0 + barWidth},${barTop + r} V${y(0)} Z`
              : "";
          return (
            <g
              key={datum.label}
              onMouseMove={(event) => {
                const bounds = (event.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
                show({
                  x: event.clientX - bounds.left,
                  y: event.clientY - bounds.top,
                  content: (
                    <>
                      <strong>{datum.label}</strong>
                      <br />
                      {datum.detail ?? datum.value}
                    </>
                  ),
                });
              }}
              onMouseLeave={hide}
            >
              <rect x={cx - slot / 2} y={top} width={slot} height={plotHeight} fill="transparent" />
              {path ? <path d={path} fill="var(--series-1)" /> : null}
              {index % 4 === 0 ? (
                <text className="tick" x={cx} y={height - 6} textAnchor="middle">
                  {datum.label}
                </text>
              ) : null}
            </g>
          );
        })}
      </svg>
      {node}
    </figure>
  );
}

function niceCeiling(value: number): number {
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * base) return step * base;
  }
  return 10 * base;
}

function formatTick(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 1)}K`;
  if (Number.isInteger(value)) return String(value);
  return value.toFixed(1);
}
