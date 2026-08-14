import { useEffect, useState } from "react";
import { api } from "../api";
import type { DistributionData, ExperimentResultsResponse } from "../types";

const PERCENTILES = [50, 90, 95, 99];

function percentileValues(series: DistributionData["adaptive"]): { label: string; value: number }[] {
  const out: { label: string; value: number }[] = [];
  const cdf = series.cdf;
  const edges = series.edges;
  for (const p of PERCENTILES) {
    const q = p / 100;
    let idx = 0;
    for (let i = 0; i < cdf.length; i++) {
      if (cdf[i] >= q) {
        idx = i;
        break;
      }
    }
    out.push({ label: `P${p}`, value: edges[idx] ?? 0 });
  }
  return out;
}

function histRows(data: DistributionData) {
  const rows: { bin: string; low: number; adaptive: number; immediate: number }[] = [];
  const { edges, bin_counts: ad } = data.adaptive;
  const { bin_counts: im } = data.immediate;
  for (let i = 0; i < edges.length - 1; i++) {
    rows.push({ bin: `${edges[i].toFixed(0)}`, low: edges[i], adaptive: ad[i] ?? 0, immediate: im[i] ?? 0 });
  }
  return rows;
}

function cellColor(v: number): string {
  if (v < -0.05) return "#22c55e";
  if (v > 0.05) return "#ef4444";
  return "#94a3b8";
}

function SeriesLegend() {
  return (
    <div className="dist-legend" role="list" aria-label="Series legend">
      <div className="dist-legend-item" role="listitem">
        <span className="dist-swatch adaptive" aria-hidden="true" />
        <span>Adaptive</span>
      </div>
      <div className="dist-legend-item" role="listitem">
        <span className="dist-swatch immediate" aria-hidden="true" />
        <span>Immediate</span>
      </div>
    </div>
  );
}

function HistogramSVG({ data }: { data: DistributionData }) {
  const rows = histRows(data);
  const W = 520;
  const H = 190;
  const PAD_L = 40;
  const PAD_R = 8;
  const PAD_T = 10;
  const PAD_B = 22;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const maxCount = Math.max(1, ...rows.map((r) => Math.max(r.adaptive, r.immediate)));
  const bw = innerW / rows.length;
  const topY = (v: number) => PAD_T + innerH - (v / maxCount) * innerH;
  const gridTicks = [0, 0.25, 0.5, 0.75, 1].map((q) => Math.round(maxCount * q));
  const xLabelEvery = Math.max(1, Math.ceil(rows.length / 10));
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Delivery-time histogram. Blue bars = adaptive policy, orange bars = immediate policy.">
      {gridTicks.map((t) => (
        <g key={t}>
          <line x1={PAD_L} y1={topY(t)} x2={W - PAD_R} y2={topY(t)} stroke="rgba(255,255,255,0.06)" />
          <text x={PAD_L - 4} y={topY(t) + 3} textAnchor="end" fontSize={9} fill="#94a3b8">
            {t}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const x = PAD_L + i * bw;
        return (
          <g key={i}>
            <rect x={x} y={topY(r.adaptive)} width={bw / 2} height={PAD_T + innerH - topY(r.adaptive)} fill="#3b82f6" opacity={0.8} />
            <rect x={x + bw / 2} y={topY(r.immediate)} width={bw / 2} height={PAD_T + innerH - topY(r.immediate)} fill="#f97316" opacity={0.6} />
            {i % xLabelEvery === 0 && (
              <text x={x + bw / 2} y={H - 6} textAnchor="middle" fontSize={9} fill="#94a3b8">
                {r.bin}
              </text>
            )}
          </g>
        );
      })}
      <text x={PAD_L} y={PAD_T - 2} fontSize={9} fill="#94a3b8">
        orders / bin
      </text>
    </svg>
  );
}

function CdfSVG({ data }: { data: DistributionData }) {
  const a = data.adaptive;
  const b = data.immediate;
  const n = Math.min(a.cdf.length, b.cdf.length);
  const W = 520;
  const H = 190;
  const PAD_L = 40;
  const PAD_R = 8;
  const PAD_T = 10;
  const PAD_B = 22;
  const innerW = W - PAD_L - PAD_R;
  const innerH = H - PAD_T - PAD_B;
  const x = (i: number) => PAD_L + (n > 1 ? (i / (n - 1)) * innerW : innerW / 2);
  const y = (v: number) => PAD_T + innerH - (v / 100) * innerH;
  const gridTicks = [0, 25, 50, 75, 100];
  const ptsA = a.cdf.slice(0, n).map((v, i) => `${x(i)},${y(v * 100)}`).join(" ");
  const ptsB = b.cdf.slice(0, n).map((v, i) => `${x(i)},${y(v * 100)}`).join(" ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} role="img" aria-label="Delivery-time CDF. Blue line = adaptive policy, orange line = immediate policy.">
      {gridTicks.map((t) => (
        <g key={t}>
          <line x1={PAD_L} y1={y(t)} x2={W - PAD_R} y2={y(t)} stroke="rgba(255,255,255,0.06)" />
          <text x={PAD_L - 4} y={y(t) + 3} textAnchor="end" fontSize={9} fill="#94a3b8">
            {t}%
          </text>
        </g>
      ))}
      <polyline points={ptsB} fill="none" stroke="#f97316" strokeWidth={2} opacity={0.8}>
        <title>Immediate</title>
      </polyline>
      <polyline points={ptsA} fill="none" stroke="#3b82f6" strokeWidth={2}>
        <title>Adaptive</title>
      </polyline>
      <text x={PAD_L} y={H - 6} fontSize={9} fill="#94a3b8">
        minutes
      </text>
    </svg>
  );
}

export function DeliveryDistributionPanel() {
  const [results, setResults] = useState<ExperimentResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);

  async function refresh() {
    try {
      const res = await api.experimentResults();
      setResults(res);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }

  // Load results on mount
  useEffect(() => {
    let mounted = true;
    async function fetchResults() {
      try {
        const res = await api.experimentResults();
        if (mounted) {
          setResults(res);
          setError(null);
        }
      } catch (e) {
        if (mounted) setError(String(e));
      } finally {
        if (mounted) setLoading(false);
      }
    }
    fetchResults();
    return () => {
      mounted = false;
    };
  }, []);

  // Poll for experiment status (same pattern as ExperimentPanel); refetch results
  // when a run completes so new distribution data shows up without a page reload.
  useEffect(() => {
    let mounted = true;
    async function pollStatus() {
      try {
        const status = await api.experimentStatus();
        if (!mounted) return;
        setRunning(status.running);
        if (!status.running && status.progress.status === "completed") {
          await refresh();
        }
        if (mounted && status.running) {
          setTimeout(pollStatus, 1000);
        }
      } catch {
        // Ignore errors
      }
    }
    pollStatus();
    return () => {
      mounted = false;
    };
  }, []);

  if (loading) return <div className="panel">Loading distribution data...</div>;
  if (error) return <div className="panel error">Error: {error}</div>;

  const distributions = results?.distributions;
  const scenarios = results?.scenarios;
  const multiScenario = Boolean(results?.multi_scenario);

  const hasData = distributions ? Object.keys(distributions).length > 0 : false;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Delivery-Time Distributions</h3>
        <div className="head-actions">
          {running && <div className="progress-badge">Running experiment…</div>}
          <span>{multiScenario ? "Across scenarios" : "Pooled over all paired runs"}</span>
          <button className="btn secondary" onClick={refresh} disabled={running}>
            Refresh
          </button>
        </div>
      </div>

      {!hasData ? (
        <div className="no-results">
          <p>No distribution data captured yet.</p>
          <p>Run an experiment (single or multi-scenario) to see delivery-time distribution shapes, CDFs, and percentile comparisons.</p>
        </div>
      ) : (
        <div className="panel-body">
          <SeriesLegend />

          {Object.entries(distributions ?? {}).map(([key, data]) => {
            const label = data.scenario || key;
            return (
              <div className="exp-scenario-block" key={key}>
                <h4>{label}</h4>
                <p className="exp-note">
                  {data.num_paired_runs.toLocaleString()} paired runs · {data.adaptive.total_orders.toLocaleString()} orders · adaptive avg{" "}
                  {data.adaptive.avg_delivery_min.toFixed(2)} min vs immediate avg {data.immediate.avg_delivery_min.toFixed(2)} min
                </p>

                <div className="dist-grid">
                  <div className="dist-card">
                    <h5>Delivery-time histogram</h5>
                    <HistogramSVG data={data} />
                  </div>
                  <div className="dist-card">
                    <h5>CDF (delivered by time)</h5>
                    <CdfSVG data={data} />
                  </div>
                </div>

                <div className="dist-card">
                  <h5>Percentile comparison (adaptive vs immediate)</h5>
                  <PercentileBars data={data} />
                </div>
              </div>
            );
          })}

          {scenarios && Object.keys(scenarios).length > 0 && (
            <div className="exp-scenario-block">
              <h4>Scenario × Delivery-Tail Heatmap</h4>
              <p className="exp-note">Adaptive − Immediate, mean diff in minutes. Green = adaptive faster, red = immediate faster.</p>
              <table className="exp-table">
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Avg</th>
                    <th>P50</th>
                    <th>P90</th>
                    <th>P95</th>
                    <th>P99</th>
                    <th>Max</th>
                    <th>Late count</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(scenarios).map(([name, sc]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td style={{ color: cellColor(sc.avg_delivery_min_diff_mean) }}>{sc.avg_delivery_min_diff_mean.toFixed(3)}</td>
                      <td style={{ color: cellColor(sc.p50_delivery_min_diff_mean) }}>{sc.p50_delivery_min_diff_mean.toFixed(3)}</td>
                      <td style={{ color: cellColor(sc.p90_delivery_min_diff_mean) }}>{sc.p90_delivery_min_diff_mean.toFixed(3)}</td>
                      <td style={{ color: cellColor(sc.p95_delivery_min_diff_mean) }}>{sc.p95_delivery_min_diff_mean.toFixed(3)}</td>
                      <td style={{ color: cellColor(sc.p99_delivery_min_diff_mean) }}>{sc.p99_delivery_min_diff_mean.toFixed(3)}</td>
                      <td style={{ color: cellColor(sc.max_delivery_min_diff_mean) }}>{sc.max_delivery_min_diff_mean.toFixed(3)}</td>
                      <td style={{ color: cellColor(sc.late_count_diff_mean) }}>{sc.late_count_diff_mean.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function PercentileBars({ data }: { data: DistributionData }) {
  const adaptive = percentileValues(data.adaptive);
  const immediate = percentileValues(data.immediate);
  return (
    <div className="rc-bars">
      {adaptive.map((a, i) => {
        const im = immediate[i]?.value ?? 0;
        const winner = a.value <= im ? "adaptive" : "immediate";
        const span = Math.max(1, a.value, im);
        return (
          <div className="bar-row" key={a.label}>
            <div className="bar-label">{a.label}</div>
            <div className="bar">
              <div className="bar-fill" style={{ width: `${(a.value / span) * 100}%`, background: "#3b82f6" }} />
              <div className="bar-fill" style={{ width: `${(im / span) * 100}%`, background: "#f97316", opacity: 0.75 }} />
            </div>
            <div className="bar-value">
              {a.value.toFixed(1)} / {im.toFixed(1)} min · {winner}
            </div>
          </div>
        );
      })}
    </div>
  );
}
