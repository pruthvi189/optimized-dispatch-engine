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
  // Backend sends `metrics: {}` on the first frame before the runner emits;
  // treat an empty object the same as null and show the skeleton.
  if (!m || Object.keys(m).length === 0) {
    return (
      <div className="kpis">
        {["placed", "completed", "cancelled", "on-time", "avg delivery", "order wait", "rider kitchen wait", "cost score"].map((l) => (
          <Card key={l} label={l} value="—" />
        ))}
      </div>
    );
  }
  const onTime = (m.on_time_rate * 100).toFixed(1) + "%";
  return (
    <div className="kpis">
      <Card label="placed" value={`${m.orders_placed}`} />
      <Card label="completed" value={`${m.orders_completed}`} tone="good" />
      <Card label="cancelled" value={`${m.orders_cancelled}`} tone={m.orders_cancelled ? "warn" : ""} />
      <Card label="on-time" value={onTime} tone={m.on_time_rate >= 0.5 ? "good" : m.on_time_rate >= 0.15 ? "warn" : "bad"} />
      <Card label="avg delivery" value={`${m.avg_delivery_min.toFixed(1)}m`} />
      <Card label="order wait" value={`${m.avg_order_wait_min.toFixed(2)}m`} tone={m.avg_order_wait_min > 1 ? "warn" : ""} />
      <Card label="rider kitchen wait" value={`${m.avg_rider_wait_kitchen_min.toFixed(2)}m`} tone={m.avg_rider_wait_kitchen_min > 1.5 ? "warn" : "good"} />
      <Card label="cost score" value={m.cost_score.toFixed(0)} />
    </div>
  );
}