import type { Snapshot } from "../types";
import { fmtClock } from "../types";
import type { ConnState } from "../useSimStream";

const STREAM_LABEL: Record<ConnState, string> = {
  connecting: "connecting",
  connected: "connected",
  disconnected: "offline",
};

export function Header({ snap, conn }: { snap: Snapshot | null; conn: ConnState }) {
  const st = snap
    ? snap.finished
      ? "finished"
      : snap.paused
        ? "paused"
        : snap.running
          ? "running"
          : "idle"
    : "idle";

  return (
    <header className="header">
      <div className="brand">
        <div className="title">Optimised Dispatch Engine</div>
        <div className="sub">
          {snap
            ? `${snap.scenario} · seed ${snap.seed} · ${snap.policy}`
            : "connecting to API…"}
        </div>
      </div>
      <div className="spacer" />
      <div className="hcell">
        <span className="k">weather</span>
        <span className="v">{snap?.weather ?? "—"}</span>
      </div>
      <div className="hcell">
        <span className="k">traffic</span>
        <span className="v">{snap?.traffic ?? "—"}</span>
      </div>
      <div className="hcell">
        <span className="k">state</span>
        <span className={`status ${st}`}>{st}</span>
      </div>
      <span className={`stream ${conn}`} title={`websocket: ${conn}`}>
        <i />
        stream {STREAM_LABEL[conn]}
      </span>
      <div className="hcell clock-cell">
        <span className="k">sim time</span>
        <span className="clock">{snap ? fmtClock(snap.sim_time_min) : "00:00"}</span>
      </div>
    </header>
  );
}