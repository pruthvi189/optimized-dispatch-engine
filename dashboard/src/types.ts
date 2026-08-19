// Mirror of the Phase 4 snapshot contract (api/snapshot.py).
// The dashboard consumes these shapes verbatim from /ws and /sim/status.

export interface StageDurations {
  kitchen_queue: number;
  kitchen_prep: number;
  dispatch_delay: number;
  rider_to_kitchen: number;
  rider_wait: number;
  customer_travel: number;
}

export interface RootCauseAnalysis {
  order_id: number;
  is_late: boolean;
  delivery_time_min: number;
  promise_time_min: number;
  lateness_min: number;
  primary_root_cause: string;
  contributing_factors: string[];
  stage_durations: StageDurations;
}

export interface RootCauseAggregate {
  total_orders: number;
  late_orders: number;
  on_time_rate: number;
  root_cause_distribution: Record<string, number>;
  primary_cause_percentages: Record<string, number>;
  contributing_factor_counts: Record<string, number>;
  contributing_factor_percentages: Record<string, number>;
}

export interface RootCausesResponse {
  aggregate: RootCauseAggregate;
  late_orders: RootCauseAnalysis[];
  total_analyzed: number;
}

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

export interface KitchenEvaluation {
  kitchen_id: number;
  distance_km: number;
  queue_len: number;
  staff_level: number;
  delivery_est_min: number;
  score?: number;
  rider_id?: number;
  rider_to_kitchen_km?: number;
  kitchen_distance_km?: number;
  total_est_min?: number;
}

export interface DecisionInputs {
  evaluations?: KitchenEvaluation[];
  selected_kitchen_id?: number;
  selected_rider_id?: number;
  [key: string]: unknown;
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
  items: number | null;
  complexity: string | null;
  selected_kitchen_id: number | null;
  selected_rider_id: number | null;
  selected_kitchen_distance: number | null;
  inputs: DecisionInputs | null;
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
  p95_delivery_min: number;
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
export const ALL_POLICIES = ["immediate", "adaptive", "nearest_kitchen", "optimized_kitchen", "nearest_heuristic", "joint_optimizer"];
export const MAIN_POLICIES: { key: string; label: string; role: "baseline" | "optimized" }[] = [
  { key: "nearest_heuristic", label: "Baseline Dispatch", role: "baseline" },
  { key: "joint_optimizer", label: "Optimized Dispatch", role: "optimized" },
];
/** @deprecated Use MAIN_POLICIES on the main dashboard */
export const POLICIES = ALL_POLICIES;

// Experiment types
export interface ExperimentSummary {
  num_experiments: number;
  scenario: string;
  days: number;
  base_seed: number;
  adaptive_wins: number;
  immediate_wins: number;
  ties: number;
  on_time_pct_diff_mean: number;
  on_time_pct_diff_median: number;
  on_time_pct_diff_std: number;
  avg_delivery_min_diff_mean: number;
  avg_delivery_min_diff_median: number;
  avg_delivery_min_diff_std: number;
  p50_delivery_min_diff_mean: number;
  p90_delivery_min_diff_mean: number;
  p95_delivery_min_diff_mean: number;
  p99_delivery_min_diff_mean: number;
  max_delivery_min_diff_mean: number;
  late_count_diff_mean: number;
  avg_late_min_diff_mean: number;
  avg_late_min_diff_median: number;
  avg_late_min_diff_std: number;
  avg_order_wait_min_diff_mean: number;
  avg_rider_wait_kitchen_min_diff_mean: number;
  avg_rider_wait_kitchen_min_diff_median: number;
  cost_score_diff_mean: number;
  cost_score_diff_median: number;
  cost_score_diff_std: number;
  scenario_breakdown: Record<string, unknown>;
}

export interface DistributionSeries {
  bin_counts: number[];
  edges: number[];
  cdf: number[];
  total_orders: number;
  avg_delivery_min: number;
  percentiles?: { 50: number; 90: number; 95: number; 99: number };
}

export interface DistributionData {
  scenario: string;
  num_paired_runs: number;
  max_min: number;
  adaptive: DistributionSeries;
  immediate: DistributionSeries;
}

export interface ExperimentResultsResponse {
  summary: ExperimentSummary;
  num_results: number;
  multi_scenario?: boolean;
  scenarios?: Record<string, ExperimentSummary>;
  distributions?: Record<string, DistributionData>;
}

export interface ExperimentStatusResponse {
  running: boolean;
  progress: {
    current: number;
    total: number;
    status: string;
  };
}

export interface ExperimentRunRequest {
  num_experiments: number;
  base_seed: number;
  days: number;
  scenario: string;
  predictor_dir: string;
  out_dir: string;
  multi_scenario: boolean;
  experiments_per_scenario: number;
}

export function fmtClock(min: number): string {
  const total = Math.max(0, Math.floor(min));
  const hh = String(Math.floor(total / 60) % 24).padStart(2, "0");
  const mm = String(total % 60).padStart(2, "0");
  const day = Math.floor(total / 1440);
  return day > 0 ? `${hh}:${mm} (day ${day + 1})` : `${hh}:${mm}`;
}
