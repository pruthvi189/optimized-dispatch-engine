import type { Decision } from "../types";
import { formatPrepEst, formatDeliveryEst, formatNumberWithUnit } from "../utils/format";

interface KitchenSummary {
  kitchen_id: number;
  min_est: number;
  max_est: number;
  avg_est: number;
  rider_count: number;
  min_rider_to_kitchen: number;
  max_rider_to_kitchen: number;
  queue_len: number | null;
  staff_level: number | null;
  kitchen_distance_km: number | null;
}

/**
 * "Why This Dispatch?" visualization.
 * Shows the most recent dispatch decision:
 *   ORDER → Candidate kitchens (4 rows) → Selected kitchen+rider → Why
 */
export function DispatchDecisionVisual({ decisions }: { decisions: Decision[] }) {
  const latest = decisions[0];

  if (!latest) {
    return (
      <div className="panel dispatch-visual">
        <div className="panel-head">
          <h3>Why This Dispatch?</h3>
        </div>
        <div className="panel-body">
          <div className="empty">Start a simulation to see dispatch decisions visualized.</div>
        </div>
      </div>
    );
  }

  const prepEst = formatPrepEst(latest.prep_mean, latest.prep_low, latest.prep_high);
  const items = latest.items ?? null;
  const complexity = latest.complexity ?? null;
  const candidateEvaluations = latest.inputs?.evaluations ?? [];
  const selectedKitchenId = latest.selected_kitchen_id;
  const selectedRiderId = latest.selected_rider_id;

  // Aggregate evaluations by kitchen (kitchen-level summary)
  const kitchenSummaries: KitchenSummary[] = [];
  if (candidateEvaluations.length > 0) {
    const byKitchen = new Map<number, { ests: number[]; r2k: number[]; queue_len: number | null; staff_level: number | null; kdist: number | null }>();
    for (const ev of candidateEvaluations) {
      const k = byKitchen.get(ev.kitchen_id) ?? { ests: [], r2k: [], queue_len: null, staff_level: null, kdist: null };
      if (ev.total_est_min != null) k.ests.push(ev.total_est_min);
      if (ev.rider_to_kitchen_km != null) k.r2k.push(ev.rider_to_kitchen_km);
      if (ev.queue_len != null) k.queue_len = ev.queue_len;
      if (ev.staff_level != null) k.staff_level = ev.staff_level;
      if (ev.kitchen_distance_km != null) k.kdist = ev.kitchen_distance_km;
      byKitchen.set(ev.kitchen_id, k);
    }
    for (const [kid, data] of byKitchen) {
      kitchenSummaries.push({
        kitchen_id: kid,
        min_est: Math.min(...data.ests),
        max_est: Math.max(...data.ests),
        avg_est: data.ests.reduce((a, b) => a + b, 0) / data.ests.length,
        rider_count: data.ests.length,
        min_rider_to_kitchen: data.r2k.length ? Math.min(...data.r2k) : 0,
        max_rider_to_kitchen: data.r2k.length ? Math.max(...data.r2k) : 0,
        queue_len: data.queue_len,
        staff_level: data.staff_level,
        kitchen_distance_km: data.kdist,
      });
    }
    // Sort by kitchen_id for consistent display
    kitchenSummaries.sort((a, b) => a.kitchen_id - b.kitchen_id);
  }

  // Find the exact selected evaluation (kitchen + rider pair)
  const selectedEval = candidateEvaluations.find(
    e => e.kitchen_id === selectedKitchenId && e.rider_id === selectedRiderId
  );

  return (
    <div className="panel dispatch-visual">
      <div className="panel-head">
        <h3>Why This Dispatch?</h3>
        <span className="panel-meta">order #{latest.order_id}</span>
      </div>
      <div className="panel-body">
        <div className="dv-flow">
          {/* Step 1: Order */}
          <div className="dv-step">
            <div className="dv-step-label">Order</div>
            <div className="dv-card dv-order">
              <span className="dv-order-id">#{latest.order_id}</span>
              <span className="dv-order-policy">{latest.policy ?? "—"}</span>
            </div>
          </div>

          <div className="dv-arrow">&#8595;</div>

          {/* Step 2: Order details */}
          <div className="dv-step">
            <div className="dv-step-label">Order details</div>
            <div className="dv-factors">
              {items != null && (
                <div className="dv-factor">
                  <span className="dv-factor-label">Items</span>
                  <span className="dv-factor-value">{items}</span>
                </div>
              )}
              {complexity != null && (
                <div className="dv-factor">
                  <span className="dv-factor-label">Complexity</span>
                  <span className="dv-factor-value" style={{ textTransform: "capitalize" }}>{complexity}</span>
                </div>
              )}
            </div>
          </div>

          <div className="dv-arrow">&#8595;</div>

          {/* Step 3: Candidate Kitchens (4 rows) */}
          {kitchenSummaries.length > 0 && (
            <div className="dv-step">
              <div className="dv-step-label">
                Candidate kitchens
                <span className="eval-count">
                  ({candidateEvaluations.length} kitchen&times;rider combinations evaluated)
                </span>
              </div>
              <div className="dv-kitchen-table">
                <table>
                  <thead>
                    <tr>
                      <th>Kitchen</th>
                      <th>Est. delivery (min)</th>
                      <th>Rider→Kitchen (km)</th>
                      <th>Queue / Staff</th>
                    </tr>
                  </thead>
                  <tbody>
                    {kitchenSummaries.map((ks) => {
                      const isSelected = ks.kitchen_id === selectedKitchenId;
                      return (
                        <tr key={ks.kitchen_id} className={isSelected ? "selected" : ""}>
                          <td className={isSelected ? "selected-kitchen" : ""}>
                            Kitchen {ks.kitchen_id}
                            {isSelected && <span className="selected-badge">SELECTED</span>}
                          </td>
                          <td>{ks.min_est.toFixed(1)} – {ks.max_est.toFixed(1)}</td>
                          <td>{ks.min_rider_to_kitchen.toFixed(1)} – {ks.max_rider_to_kitchen.toFixed(1)} km</td>
                          <td>
                            {ks.queue_len != null
                              ? `${ks.queue_len} orders` + (ks.staff_level != null ? ` / ${ks.staff_level} staff` : "")
                              : "—"}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <div className="dv-arrow">&#8595;</div>

          {/* Step 4: Why This Decision? */}
          {selectedEval ? (
            <div className="dv-step">
              <div className="dv-step-label">Why this decision?</div>
              <div className="dv-why-box">
                <div className="dv-why-header">
                  <div className="dv-why-identity">
                    <span className="dv-why-kitchen">Kitchen {selectedEval.kitchen_id}</span>
                    <span className="dv-why-plus">+</span>
                    <span className="dv-why-rider">Rider {selectedEval.rider_id}</span>
                  </div>
                  <span className="dv-why-selected">SELECTED</span>
                </div>
                <div className="dv-why-factors">
                  <div className="dv-why-row">
                    <span className="dv-why-label">Kitchen → Customer</span>
                    <span className="dv-why-val">{formatNumberWithUnit(selectedEval.kitchen_distance_km ?? null, "km")}</span>
                  </div>
                  <div className="dv-why-row">
                    <span className="dv-why-label">Rider → Kitchen</span>
                    <span className="dv-why-val">{formatNumberWithUnit(selectedEval.rider_to_kitchen_km ?? null, "km")}</span>
                  </div>
                  <div className="dv-why-row">
                    <span className="dv-why-label">Prep time</span>
                    <span className="dv-why-val">{prepEst} min</span>
                  </div>
                  <div className="dv-why-row">
                    <span className="dv-why-label">Kitchen queue</span>
                    <span className="dv-why-val">
                      {selectedEval.queue_len != null
                        ? `${selectedEval.queue_len} order${selectedEval.queue_len === 1 ? "" : "s"}`
                        : "—"}
                      {selectedEval.staff_level != null ? ` (${selectedEval.staff_level} staff)` : ""}
                    </span>
                  </div>
                  <div className="dv-why-row dv-why-highlight">
                    <span className="dv-why-label">Estimated delivery</span>
                    <span className="dv-why-val">{formatDeliveryEst(selectedEval.delivery_est_min, selectedEval.total_est_min)}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : latest.rationale ? (
            <div className="dv-step">
              <div className="dv-step-label">Why this decision?</div>
              <div className="dv-why-box">
                <div className="dv-why-factors">
                  <div className="dv-why-row dv-why-highlight">
                    <span className="dv-why-label">Decision rationale</span>
                    <span className="dv-why-val">{latest.rationale}</span>
                  </div>
                  {latest.selected_kitchen_id != null && (
                    <div className="dv-why-row">
                      <span className="dv-why-label">Selected kitchen</span>
                      <span className="dv-why-val">Kitchen {latest.selected_kitchen_id}</span>
                    </div>
                  )}
                  {latest.selected_rider_id != null && (
                    <div className="dv-why-row">
                      <span className="dv-why-label">Selected rider</span>
                      <span className="dv-why-val">Rider {latest.selected_rider_id}</span>
                    </div>
                  )}
                  {latest.selected_kitchen_distance != null && (
                    <div className="dv-why-row">
                      <span className="dv-why-label">Kitchen → Customer</span>
                      <span className="dv-why-val">{formatNumberWithUnit(latest.selected_kitchen_distance, "km")}</span>
                    </div>
                  )}
                  {latest.travel_to_kitchen_min != null && (
                    <div className="dv-why-row">
                      <span className="dv-why-label">Rider → Kitchen</span>
                      <span className="dv-why-val">{formatNumberWithUnit(latest.travel_to_kitchen_min)}</span>
                    </div>
                  )}
                  <div className="dv-why-row">
                    <span className="dv-why-label">Prep time</span>
                    <span className="dv-why-val">{prepEst} min</span>
                  </div>
                  {latest.risk_buffer_min != null && (
                    <div className="dv-why-row">
                      <span className="dv-why-label">Risk buffer</span>
                      <span className="dv-why-val">{formatNumberWithUnit(latest.risk_buffer_min)}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          ) : null}
        </div>

        {/* Recent decisions mini-table */}
        {decisions.length > 1 && (
          <div className="dv-recent">
            <div className="dv-recent-label">Recent decisions</div>
            <table className="dv-recent-table">
              <thead>
                <tr>
                  <th>Order</th>
                  <th>Items</th>
                  <th>Prep est</th>
                  <th>Travel</th>
                  <th>Buffer</th>
                  <th>Dispatch @</th>
                </tr>
              </thead>
              <tbody>
                {decisions.slice(1, 6).map((d, i) => (
                  <tr key={`${d.order_id}-${i}`}>
                    <td>#{d.order_id}</td>
                    <td>{d.items != null ? `${d.items} (${d.complexity ?? "?"})` : "—"}</td>
                    <td>{formatPrepEst(d.prep_mean, d.prep_low, d.prep_high)}</td>
                    <td>{formatNumberWithUnit(d.travel_to_kitchen_min)}</td>
                    <td>{formatNumberWithUnit(d.risk_buffer_min)}</td>
                    <td>{formatNumberWithUnit(d.dispatch_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}