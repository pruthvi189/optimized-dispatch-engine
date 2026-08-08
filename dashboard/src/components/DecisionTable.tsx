import type { Decision } from "../types";

const UNC_TONE: Record<string, string> = {
  low: "good",
  medium: "",
  high: "warn",
};

export function DecisionTable({ decisions }: { decisions: Decision[] }) {
  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Dispatch decisions</h3>
        <span className="panel-meta">{decisions.length} recent</span>
      </div>
      <div className="panel-body">
        {decisions.length === 0 ? (
          <div className="empty">no decisions yet</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>order</th>
                <th>policy</th>
                <th className="num">dispatch @</th>
                <th>prep est (low/mean)</th>
                <th>unc</th>
                <th className="num">buffer</th>
                <th>rationale</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d, i) => (
                <tr key={`${d.order_id}-${i}`}>
                  <td>{d.order_id ?? "—"}</td>
                  <td>{d.policy ?? "—"}</td>
                  <td className="num">
                    {d.dispatch_at != null ? d.dispatch_at.toFixed(1) : "—"}
                  </td>
                  <td>
                    {d.prep_low != null
                      ? `${d.prep_low.toFixed(1)} / ${d.prep_mean?.toFixed(1)}`
                      : "—"}
                  </td>
                  <td>
                    {d.uncertainty
                      ? <span className={`tone ${UNC_TONE[d.uncertainty]}`}>{d.uncertainty}</span>
                      : "—"}
                  </td>
                  <td className="num">
                    {d.risk_buffer_min != null
                      ? d.risk_buffer_min.toFixed(1)
                      : "—"}
                  </td>
                  <td className="rationale" title={d.rationale ?? ""}>
                    {d.rationale ?? ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}