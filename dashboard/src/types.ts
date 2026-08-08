// Mirror of the Phase 4 snapshot contract (api/snapshot.py).
// The dashboard consumes these shapes verbatim from /ws and /sim/status.

export interface KitchenOrder {
  id: number;
  status: string;
  items: number;
  complexity: string;
  placed_at: number;
  dispatch_at: number | null;
}

export interface KitchenState {
  id: number;
  queue_len: number;
  orders: KitchenOrder[];
}

export interface RiderState {
  id: number;
  status: string;
  busy_min: number;
  assigned_to: number | null;
}

export interface Decision {
  order_id: number | null;
  policy: string | null;
  dispatch_at: number | null;
  prep_mean: number | null;
  prep_low: number | null;
  prep_high: number | null;
  uncertainty: string | null;
  risk_buffer_min: number | null;
  travel_to_kitchen_min: number | null;
  rationale: string | null;
}

export interface EventRow {
  sim_time: number;
  event_type: string;
  order_id: number | null;
  rider_id: number | null;
}

export interface Metrics {
  orders_placed: number;
  orders_completed: number;
  orders_cancelled: number;
  on_time_rate: number;
  avg_delivery_min: number;
  avg_late_min: number;
  avg_order_wait_min: number;
  avg_rider_wait_kitchen_min: number;
  avg_rider_idle_min: number;
  cost_score: number;
}

export interface Snapshot {
  sim_time_min: number;
  scenario: string | null;
  policy: string | null;
  days: number | null;
  seed: number | null;
  total_minutes: number;
  speed: number | null;
  running: boolean;
  paused: boolean;
  finished: boolean;
  weather: string | null;
  traffic: string | null;
  kitchens: KitchenState[];
  riders: RiderState[];
  recent_decisions: Decision[];
  metrics: Metrics;
  events: EventRow[];
}

export interface RunnerConfig {
  scenario: string;
  seed: number;
  policy: string;
  days: number;
  speed: number | null;
  step_minutes: number;
}

export const SCENARIOS = ["normal", "lunch_rush", "rain", "low_staffing", "traffic_spike"];
export const POLICIES = ["adaptive", "immediate"];

export function fmtClock(min: number): string {
  const total = Math.max(0, Math.floor(min));
  const hh = String(Math.floor(total / 60) % 24).padStart(2, "0");
  const mm = String(total % 60).padStart(2, "0");
  const day = Math.floor(total / 1440);
  return day > 0 ? `${hh}:${mm} (day ${day + 1})` : `${hh}:${mm}`;
}
