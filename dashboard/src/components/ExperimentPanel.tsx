import { useEffect, useState } from "react";
import { api } from "../api";
import type { ExperimentResultsResponse, ExperimentRunRequest, ExperimentStatusResponse } from "../types";

export function ExperimentPanel() {
  const [results, setResults] = useState<ExperimentResultsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [progress, setProgress] = useState<ExperimentStatusResponse["progress"]>({ current: 0, total: 0, status: "idle" });

  // Poll for experiment status
  useEffect(() => {
    let mounted = true;
    async function pollStatus() {
      try {
        const status = await api.experimentStatus();
        if (mounted) {
          setRunning(status.running);
          setProgress(status.progress);
        }
        if (mounted && status.running) {
          setTimeout(pollStatus, 1000);
        }
      } catch {
        // Ignore errors
      }
    }
    pollStatus();
    return () => { mounted = false; };
  }, []);

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
    return () => { mounted = false; };
  }, []);

  const handleRunExperiment = async () => {
    setError(null);
    try {
      const req: ExperimentRunRequest = {
        num_experiments: 1000,
        base_seed: 42,
        days: 1,
        scenario: "normal",
        predictor_dir: "artifacts",
        out_dir: "data/experiments",
        multi_scenario: false,
        experiments_per_scenario: 2000,
      };
      await api.runExperiment(req);
      setRunning(true);
      setProgress({ current: 0, total: req.num_experiments, status: "running" });
    } catch (e) {
      setError(String(e));
    }
  };

  const handleRunMultiScenario = async () => {
    setError(null);
    try {
      const req: ExperimentRunRequest = {
        num_experiments: 1000,
        base_seed: 42,
        days: 1,
        scenario: "normal",
        predictor_dir: "artifacts",
        out_dir: "data/experiments",
        multi_scenario: true,
        experiments_per_scenario: 500,
      };
      await api.runExperiment(req);
      setRunning(true);
      setProgress({ current: 0, total: req.experiments_per_scenario * 5, status: "running" });
    } catch (e) {
      setError(String(e));
    }
  };

  if (loading) return <div className="panel">Loading experiment results...</div>;
  if (error && !results) return <div className="panel error">Error: {error}</div>;

  const summary = results?.summary;

  // Helper to safely access summary properties
  const s = summary ?? {
    adaptive_wins: 0,
    immediate_wins: 0,
    ties: 0,
    num_experiments: 0,
    scenario: "—",
    days: 0,
    on_time_pct_diff_mean: 0,
    on_time_pct_diff_median: 0,
    on_time_pct_diff_std: 0,
    avg_delivery_min_diff_mean: 0,
    avg_delivery_min_diff_median: 0,
    avg_delivery_min_diff_std: 0,
    p50_delivery_min_diff_mean: 0,
    p90_delivery_min_diff_mean: 0,
    p95_delivery_min_diff_mean: 0,
    p99_delivery_min_diff_mean: 0,
    max_delivery_min_diff_mean: 0,
    late_count_diff_mean: 0,
    avg_late_min_diff_mean: 0,
    avg_late_min_diff_median: 0,
    avg_late_min_diff_std: 0,
    avg_order_wait_min_diff_mean: 0,
    avg_rider_wait_kitchen_min_diff_mean: 0,
    avg_rider_wait_kitchen_min_diff_median: 0,
    cost_score_diff_mean: 0,
    cost_score_diff_median: 0,
    cost_score_diff_std: 0,
  };

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Experiment Results</h3>
        <div className="head-actions">
          {running && (
            <div className="progress-badge">
              {progress.status === "running" ? (
                <>
                  Running: {progress.current}/{progress.total} ({Math.round(progress.current / Math.max(progress.total, 1) * 100)}%)
                </>
              ) : (
                progress.status
              )}
            </div>
          )}
          <button className="btn secondary" onClick={handleRunExperiment} disabled={running}>
            Run 1,000 Experiments
          </button>
          <button className="btn secondary" onClick={handleRunMultiScenario} disabled={running}>
            Run Multi-Scenario
          </button>
          <button className="btn primary" onClick={() => { setLoading(true); setTimeout(() => window.location.reload(), 100); }} disabled={running}>
            Refresh
          </button>
        </div>
      </div>

      {!results ? (
        <div className="no-results">
          <p>No experiment results found.</p>
          <p>Run an experiment to compare Adaptive vs Immediate dispatch across thousands of randomized environments.</p>
        </div>
      ) : (
        <div>
          {/* Win counts */}
          <div className="exp-summary-grid">
            <div className="exp-stat-card adaptive">
              <div className="exp-stat-label">Adaptive Wins</div>
              <div className="exp-stat-value">{s.adaptive_wins}</div>
            </div>
            <div className="exp-stat-card immediate">
              <div className="exp-stat-label">Immediate Wins</div>
              <div className="exp-stat-value">{s.immediate_wins}</div>
            </div>
            <div className="exp-stat-card tie">
              <div className="exp-stat-label">Ties</div>
              <div className="exp-stat-value">{s.ties}</div>
            </div>
            <div className="exp-stat-card">
              <div className="exp-stat-label">Adaptive Win Rate</div>
              <div className="exp-stat-value">{(s.adaptive_wins / Math.max(s.num_experiments, 1) * 100).toFixed(1)}%</div>
            </div>
            <div className="exp-stat-card">
              <div className="exp-stat-label">Total Experiments</div>
              <div className="exp-stat-value">{s.num_experiments}</div>
            </div>
            <div className="exp-stat-card">
              <div className="exp-stat-label">Scenario</div>
              <div className="exp-stat-value">{s.scenario} ({s.days} day)</div>
            </div>
          </div>

          {/* Multi-scenario breakdown */}
          {results.multi_scenario && results.scenarios && Object.keys(results.scenarios).length > 0 && (
            <div className="exp-scenario-block">
              <h4>Cross-Scenario Breakdown</h4>
              <p className="exp-note">Win rate per scenario (win = faster average delivery time, 0.1-min threshold).</p>
              <table className="exp-table">
                <thead>
                  <tr>
                    <th>Scenario</th>
                    <th>Adaptive Wins</th>
                    <th>Immediate Wins</th>
                    <th>Ties</th>
                    <th>Win Rate</th>
                    <th>Avg Delivery Diff (min)</th>
                    <th>P95 Delivery Diff (min)</th>
                    <th>On-Time Diff (pp)</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.scenarios).map(([name, sc]) => {
                    const wr = (sc.adaptive_wins / Math.max(sc.num_experiments, 1) * 100).toFixed(1);
                    return (
                      <tr key={name}>
                        <td>{name}</td>
                        <td>{sc.adaptive_wins}</td>
                        <td>{sc.immediate_wins}</td>
                        <td>{sc.ties}</td>
                        <td className={Number(wr) >= 50 ? "win-rate-good" : "win-rate-bad"}>{wr}%</td>
                        <td className={sc.avg_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{sc.avg_delivery_min_diff_mean.toFixed(3)}</td>
                        <td className={sc.p95_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{sc.p95_delivery_min_diff_mean.toFixed(3)}</td>
                        <td className={sc.on_time_pct_diff_mean >= 0 ? "positive" : "negative"}>{sc.on_time_pct_diff_mean >= 0 ? "+" : ""}{(sc.on_time_pct_diff_mean * 100).toFixed(2)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}

          {/* Metric comparison table */}
          <h4>Metric Comparison (Adaptive − Immediate)</h4>
          <p className="exp-note">Negative values = Adaptive better (lower is better). Positive on-time % = Adaptive higher on-time rate.</p>
          <table className="exp-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Mean Diff</th>
                <th>Median Diff</th>
                <th>Std Dev</th>
                <th>Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>On-time Rate (pp)</td>
                <td className={s.on_time_pct_diff_mean >= 0 ? "positive" : "negative"}>{(s.on_time_pct_diff_mean * 100).toFixed(2)}</td>
                <td className={s.on_time_pct_diff_median >= 0 ? "positive" : "negative"}>{(s.on_time_pct_diff_median * 100).toFixed(2)}</td>
                <td>{(s.on_time_pct_diff_std * 100).toFixed(2)}</td>
                <td>{s.on_time_pct_diff_mean >= 0 ? "Adaptive ↑ on-time" : "Immediate ↑ on-time"}</td>
              </tr>
              <tr>
                <td>Avg Delivery (min)</td>
                <td className={s.avg_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{s.avg_delivery_min_diff_mean.toFixed(3)}</td>
                <td className={s.avg_delivery_min_diff_median <= 0 ? "negative" : "positive"}>{s.avg_delivery_min_diff_median.toFixed(3)}</td>
                <td>{s.avg_delivery_min_diff_std.toFixed(3)}</td>
                <td>{s.avg_delivery_min_diff_mean <= 0 ? "Adaptive faster" : "Immediate faster"}</td>
              </tr>
              <tr>
                <td>P50 Delivery (min)</td>
                <td className={s.p50_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{s.p50_delivery_min_diff_mean.toFixed(3)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.p50_delivery_min_diff_mean <= 0 ? "Adaptive faster" : "Immediate faster"}</td>
              </tr>
              <tr>
                <td>P90 Delivery (min)</td>
                <td className={s.p90_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{s.p90_delivery_min_diff_mean.toFixed(3)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.p90_delivery_min_diff_mean <= 0 ? "Adaptive faster" : "Immediate faster"}</td>
              </tr>
              <tr>
                <td>P95 Delivery (min)</td>
                <td className={s.p95_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{s.p95_delivery_min_diff_mean.toFixed(3)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.p95_delivery_min_diff_mean <= 0 ? "Adaptive faster" : "Immediate faster"}</td>
              </tr>
              <tr>
                <td>P99 Delivery (min)</td>
                <td className={s.p99_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{s.p99_delivery_min_diff_mean.toFixed(3)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.p99_delivery_min_diff_mean <= 0 ? "Adaptive faster" : "Immediate faster"}</td>
              </tr>
              <tr>
                <td>Max Delivery (min)</td>
                <td className={s.max_delivery_min_diff_mean <= 0 ? "negative" : "positive"}>{s.max_delivery_min_diff_mean.toFixed(3)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.max_delivery_min_diff_mean <= 0 ? "Adaptive faster" : "Immediate faster"}</td>
              </tr>
              <tr>
                <td>Late Orders (count)</td>
                <td className={s.late_count_diff_mean <= 0 ? "negative" : "positive"}>{s.late_count_diff_mean.toFixed(2)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.late_count_diff_mean <= 0 ? "Adaptive fewer late" : "Immediate fewer late"}</td>
              </tr>
              <tr>
                <td>Avg Lateness (min)</td>
                <td className={s.avg_late_min_diff_mean <= 0 ? "negative" : "positive"}>{s.avg_late_min_diff_mean.toFixed(3)}</td>
                <td className={s.avg_late_min_diff_median <= 0 ? "negative" : "positive"}>{s.avg_late_min_diff_median.toFixed(3)}</td>
                <td>{s.avg_late_min_diff_std.toFixed(3)}</td>
                <td>{s.avg_late_min_diff_mean <= 0 ? "Adaptive less late" : "Immediate less late"}</td>
              </tr>
              <tr>
                <td>Avg Order Wait (min)</td>
                <td className={s.avg_order_wait_min_diff_mean <= 0 ? "negative" : "positive"}>{s.avg_order_wait_min_diff_mean.toFixed(3)}</td>
                <td>—</td>
                <td>—</td>
                <td>{s.avg_order_wait_min_diff_mean <= 0 ? "Adaptive less wait" : "Immediate less wait"}</td>
              </tr>
              <tr>
                <td>Avg Rider Kitchen Wait (min)</td>
                <td className={s.avg_rider_wait_kitchen_min_diff_mean <= 0 ? "negative" : "positive"}>{s.avg_rider_wait_kitchen_min_diff_mean.toFixed(3)}</td>
                <td className={s.avg_rider_wait_kitchen_min_diff_median <= 0 ? "negative" : "positive"}>{s.avg_rider_wait_kitchen_min_diff_median.toFixed(3)}</td>
                <td>—</td>
                <td>{s.avg_rider_wait_kitchen_min_diff_mean <= 0 ? "Adaptive less wait" : "Immediate less wait"}</td>
              </tr>
              <tr>
                <td>Cost Score</td>
                <td className={s.cost_score_diff_mean <= 0 ? "negative" : "positive"}>{s.cost_score_diff_mean.toFixed(1)}</td>
                <td className={s.cost_score_diff_median <= 0 ? "negative" : "positive"}>{s.cost_score_diff_median.toFixed(1)}</td>
                <td>{s.cost_score_diff_std.toFixed(1)}</td>
                <td>{s.cost_score_diff_mean <= 0 ? "Adaptive lower cost" : "Immediate lower cost"}</td>
              </tr>
            </tbody>
          </table>

          {/* Statistical significance note */}
          <div className="exp-stats-note">
            <h5>Statistical Notes</h5>
            <ul>
              <li>Paired experiments: each seed runs both policies on identical conditions</li>
              <li>Win determined by end-to-end delivery time (adaptive faster than immediate by more than 0.1 min)</li>
              <li>Standard deviation shows consistency across runs</li>
              <li>Run multi-scenario to see condition-dependent effects</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
}