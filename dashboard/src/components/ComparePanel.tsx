import type { Metrics } from "../types";
import type { CompareState } from "../useCompare";

const ROWS: { key: keyof Metrics; label: string; fmt: (v: number) => string; lowerBetter: boolean }[] = [
  { key: "avg_delivery_min", label: "Avg delivery time", fmt: (v) => v.toFixed(1) + " min", lowerBetter: true },
  { key: "on_time_rate", label: "On-time rate", fmt: (v) => (v * 100).toFixed(1) + "%", lowerBetter: false },
  { key: "p95_delivery_min", label: "P95 delivery time", fmt: (v) => v.toFixed(1) + " min", lowerBetter: true },
];

function winner(a: number, b: number, lowerBetter: boolean): "baseline" | "optimized" | "tie" {
  if (a === b) return "tie";
  return lowerBetter ? (a < b ? "baseline" : "optimized") : (a > b ? "baseline" : "optimized");
}

export function ComparePanel({ state, seed, days }: { state: CompareState; seed: number; days: number }) {
  const { baseline, optimized, running, error } = state;

  return (
    <div className="panel-body">
      {error && <div className="error-banner">{error}</div>}
      {running && <div className="empty">Running both policies to completion&hellip;</div>}
      {!baseline && !optimized && !running && (
        <div className="empty">
          Press <strong>Run compare</strong> to run both policies on the same seed and see which wins.
        </div>
      )}
      {baseline && optimized && (
        <>
          <div className="compare-grid">
            {ROWS.map((row) => {
              const b = baseline[row.key];
              const o = optimized[row.key];
              const w = winner(b, o, row.lowerBetter);
              return (
                <div className="compare-row" key={row.key}>
                  <div className="compare-label">{row.label}</div>
                  <div className="compare-values">
                    <div className={`compare-val baseline ${w === "baseline" ? "win" : ""}`}>
                      <span className="compare-tag">Baseline</span>
                      <span className="compare-num">{row.fmt(b)}</span>
                    </div>
                    <div className="compare-vs">vs</div>
                    <div className={`compare-val optimized ${w === "optimized" ? "win" : ""}`}>
                      <span className="compare-tag">Optimized</span>
                      <span className="compare-num">{row.fmt(o)}</span>
                    </div>
                  </div>
                  <div className={`compare-winner ${w}`}>
                    {w === "tie" ? "tie" : w === "baseline" ? "baseline better" : "optimized better"}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="compare-context">
            Seed {seed} · {days}-day simulation
          </div>
        </>
      )}
    </div>
  );
}