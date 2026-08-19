import { useEffect, useState } from "react";
import { api } from "../api";
import type { RootCausesResponse, RootCauseAnalysis } from "../types";

export function RootCausePanel({ finished }: { finished: boolean }) {
  const [data, setData] = useState<RootCausesResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedOrder, setSelectedOrder] = useState<RootCauseAnalysis | null>(null);

  useEffect(() => {
    if (!finished) {
      setData(null);
      setLoading(false);
      setError(null);
      return;
    }
    let mounted = true;
    setLoading(true);
    async function fetchData() {
      try {
        const res = await api.rootCauses();
        if (mounted) {
          setData(res);
          setError(null);
        }
      } catch (e) {
        if (mounted) setError(String(e));
      } finally {
        if (mounted) setLoading(false);
      }
    }
    fetchData();
    return () => { mounted = false; };
  }, [finished]);

  if (!finished) {
    return (
      <div className="panel">
        <div className="panel-head">
          <h3>Root Cause Analysis</h3>
          <span className="badge">no run yet</span>
        </div>
        <p>Run a simulation to see root-cause analysis for late orders.</p>
      </div>
    );
  }
  if (loading) return <div className="panel">Loading root cause analysis...</div>;
  if (error) return <div className="panel error">Error: {error}</div>;
  if (!data) return <div className="panel">No data available</div>;

  const { aggregate, late_orders } = data;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Root Cause Analysis</h3>
        <span className="badge">{aggregate.late_orders} late of {aggregate.total_orders} orders</span>
      </div>

      <div className="rc-grid">
        {/* Primary causes breakdown */}
        <div className="rc-section">
          <h4>Primary Root Causes</h4>
          <div className="rc-bars">
            {Object.entries(aggregate.primary_cause_percentages)
              .sort(([, a], [, b]) => b - a)
              .map(([cause, pct]) => {
                const count = aggregate.root_cause_distribution[cause] || 0;
                return (
                  <div key={cause} className="rc-bar-row">
                    <span className="rc-label">{cause}</span>
                    <div className="rc-bar-wrap">
                      <div
                        className="rc-bar"
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      />
                      <span className="rc-value">{pct.toFixed(1)}% ({count})</span>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Contributing factors */}
        <div className="rc-section">
          <h4>Contributing Factors</h4>
          <div className="rc-bars">
            {Object.entries(aggregate.contributing_factor_percentages)
              .sort(([, a], [, b]) => b - a)
              .map(([factor, pct]) => {
                const count = aggregate.contributing_factor_counts[factor] || 0;
                return (
                  <div key={factor} className="rc-bar-row">
                    <span className="rc-label">{factor}</span>
                    <div className="rc-bar-wrap">
                      <div
                        className="rc-bar contributing"
                        style={{ width: `${Math.min(pct, 100)}%` }}
                      />
                      <span className="rc-value">{pct.toFixed(1)}% ({count})</span>
                    </div>
                  </div>
                );
              })}
          </div>
        </div>

        {/* Late orders table */}
        <div className="rc-section full-width">
          <h4>Late Orders (Top {Math.min(late_orders.length, 20)})</h4>
          <table className="rc-table">
            <thead>
              <tr>
                <th>Order</th>
                <th>Delivery</th>
                <th>Promise</th>
                <th>Late By</th>
                <th>Primary Cause</th>
                <th>Contributing</th>
              </tr>
            </thead>
            <tbody>
              {late_orders.slice(0, 20).map((order) => (
                <tr key={order.order_id} onClick={() => setSelectedOrder(order)} style={{ cursor: "pointer" }}>
                  <td>#{order.order_id}</td>
                  <td>{order.delivery_time_min.toFixed(1)} min</td>
                  <td>{order.promise_time_min.toFixed(1)} min</td>
                  <td className="late">{order.lateness_min.toFixed(1)} min</td>
                  <td><span className="cause-badge">{order.primary_root_cause}</span></td>
                  <td>{order.contributing_factors.join(", ") || "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Order detail modal */}
      {selectedOrder && (
        <div className="modal-overlay" onClick={() => setSelectedOrder(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <div className="modal-header">
              <h3>Order #{selectedOrder.order_id} - Root Cause Detail</h3>
              <button onClick={() => setSelectedOrder(null)}>×</button>
            </div>
            <div className="modal-body">
              <div className="detail-grid">
                <div className="detail-item">
                  <span className="detail-label">Delivery Time</span>
                  <span className="detail-value">{selectedOrder.delivery_time_min.toFixed(1)} min</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Promise</span>
                  <span className="detail-value">{selectedOrder.promise_time_min.toFixed(1)} min</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Lateness</span>
                  <span className="detail-value late">{selectedOrder.lateness_min.toFixed(1)} min</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Primary Cause</span>
                  <span className="detail-value cause-badge">{selectedOrder.primary_root_cause}</span>
                </div>
                <div className="detail-item">
                  <span className="detail-label">Contributing</span>
                  <span className="detail-value">{selectedOrder.contributing_factors.join(", ") || "—"}</span>
                </div>
              </div>

              <h4>Stage Durations</h4>
              <div className="stage-bars">
                {Object.entries(selectedOrder.stage_durations).map(([stage, duration]) => (
                  <div key={stage} className="stage-bar-row">
                    <span className="stage-label">{stage.replace(/_/g, " ")}</span>
                    <div className="stage-bar-wrap">
                      <div
                        className="stage-bar"
                        style={{ width: `${Math.min(duration / selectedOrder.delivery_time_min * 100, 100)}%` }}
                      />
                      <span className="stage-value">{duration.toFixed(1)} min</span>
                    </div>
                  </div>
                ))}
              </div>

              <div className="modal-footer">
                <button onClick={() => setSelectedOrder(null)}>Close</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}