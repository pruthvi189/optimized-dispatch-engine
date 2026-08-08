import type { Metrics } from "../types";
import type { CompareState } from "../useCompare";

const ROWS: { key: keyof Metrics; label: string; fmt: (v: number) => string; lowerBetter: boolean }[] = [
  { key: "cost_score", label: "cost score", fmt: (v) => v.toFixed(0), lowerBetter: true },
  { key: "avg_rider_wait_kitchen_min", label: "rider kitchen wait (min)", fmt: (v) => v.toFixed(2), lowerBetter: true },
  { key: "avg_order_wait_min", label: "order wait (min)", fmt: (v) => v.toFixed(2), lowerBetter: true },
  { key: "on_time_rate", label: "on-time %", fmt: (v) => (v * 100).toFixed(1), lowerBetter: false },
  { key: "avg_delivery_min", label: "avg delivery (min)", fmt: (v) => v.toFixed(1), lowerBetter: true },
];

export function ComparePanel({ state }: { state: CompareState }) {
  const { immediate, adaptive, running, error } = state;

  return (
    <div className="panel-body">
      {error && <div className="error-banner">{error}</div>}
      {running && <div className="empty">running both policies to completion…</div>}
      {!immediate && !adaptive && !running && (
        <div className="empty">
          Press "Run compare" to run both policies on the same seed and see which wins on the headline KPIs.
        </div>
      )}
      {immediate && adaptive && (
        <div className="compare-bar">
          {ROWS.map((row) => {
            const a = immediate[row.key];
            const b = adaptive[row.key];
            const better = row.lowerBetter
              ? a > b
                ? "adaptive"
                : a === b
                  ? "tie"
                  : "immediate"
              : b > a
                ? "adaptive"
                : b === a
                  ? "tie"
                  : "immediate";
            const total = Math.max(a, b, 0.001);
            return (
              <div className="crow" key={row.key}>
                <div className="clabel">
                  <span className="l">{row.label}</span>
                  <span className={`win ${better === "tie" ? "tie" : better}`}>
                    {better === "tie" ? "tie" : `${better} better`}
                  </span>
                </div>
                <div className="track">
                  <div className="seg immediate" title={`immediate: ${row.fmt(a)}`} style={{ width: `${(a / total) * 100}%` }} />
                  <div className="seg adaptive" title={`adaptive: ${row.fmt(b)}`} style={{ width: `${(b / total) * 100}%` }} />
                </div>
                <div className="vals">
                  <span className="im">im {row.fmt(a)}</span>
                  <span className="ad">ad {row.fmt(b)}</span>
                </div>
              </div>
            );
          })}
          <div className="compare-meta">
            immediate: cost {immediate.cost_score.toFixed(0)}, on-time {(immediate.on_time_rate * 100).toFixed(1)}% ·{" "}
            adaptive: cost {adaptive.cost_score.toFixed(0)}, on-time {(adaptive.on_time_rate * 100).toFixed(1)}%
          </div>
        </div>
      )}
    </div>
  );
}