import type { RiderState } from "../types";

const RIDER_TONE: Record<string, string> = {
  idle: "idle",
  delivering: "good",
  assigned: "warn",
  unavailable: "warn",
};

export function RiderPanel({ riders }: { riders: RiderState[] }) {
  const delivering = riders.filter((r) => r.status === "delivering").length;

  return (
    <div className="panel">
      <div className="panel-head">
        <h3>Riders</h3>
        <span className="panel-meta">
          {riders.length ? `${riders.length} fleet · ${delivering} delivering` : "—"}
        </span>
      </div>
      <div className="panel-body">
        {!riders.length ? (
          <div className="empty">not started</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>rider</th>
                <th>status</th>
                <th className="num">busy (m)</th>
                <th>order</th>
              </tr>
            </thead>
            <tbody>
              {riders.map((r) => (
                <tr key={r.id}>
                  <td>R{r.id}</td>
                  <td className={`tone ${RIDER_TONE[r.status] ?? ""}`}>{r.status}</td>
                  <td className="num">{r.busy_min.toFixed(0)}m</td>
                  <td>{r.assigned_to != null ? `#${r.assigned_to}` : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}