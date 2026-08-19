import type { Metrics } from "../types";

function Card({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="kpi">
      <div className="label">{label}</div>
      <div className={`value ${tone ?? ""}`}>{value}</div>
    </div>
  );
}

export function KpiCards({ m }: { m: Metrics | null }) {
  if (!m || Object.keys(m).length === 0) {
    return (
      <div className="kpis">
        {["avg delivery", "P95 delivery", "on-time", "order wait"].map((l) => (
          <Card key={l} label={l} value="—" />
        ))}
      </div>
    );
  }
  const onTime = (m.on_time_rate * 100).toFixed(1) + "%";
  return (
    <div className="kpis">
      <Card label="avg delivery" value={`${m.avg_delivery_min.toFixed(1)}m`} />
      <Card label="P95 delivery" value={`${m.p95_delivery_min.toFixed(1)}m`} />
      <Card label="on-time" value={onTime} tone={m.on_time_rate >= 0.5 ? "good" : m.on_time_rate >= 0.15 ? "warn" : "bad"} />
      <Card label="order wait" value={`${m.avg_order_wait_min.toFixed(2)}m`} tone={m.avg_order_wait_min > 1 ? "warn" : ""} />
    </div>
  );
}