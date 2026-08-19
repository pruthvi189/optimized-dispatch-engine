import type { RunnerConfig, Snapshot } from "../types";
import { ALL_POLICIES, MAIN_POLICIES, SCENARIOS } from "../types";

type Action = "start" | "pause" | "resume" | "step" | "reset";

interface Props {
  cfg: RunnerConfig;
  setCfg: (c: RunnerConfig) => void;
  snap: Snapshot | null;
  busy: boolean;
  onAction: (kind: Action) => void;
  /** When true, show only the two main policies (Baseline / Optimized). */
  mainMode?: boolean;
}

export function Controls({ cfg, setCfg, snap, busy, onAction, mainMode }: Props) {
  const running = snap?.running ?? false;
  const paused = snap?.paused ?? false;
  const finished = snap?.finished ?? false;

  const policies = mainMode ? MAIN_POLICIES.map((p) => p.key) : ALL_POLICIES;
  const policyLabel = mainMode
    ? MAIN_POLICIES.find((p) => p.key === cfg.policy)?.label ?? cfg.policy
    : undefined;

  const set = <K extends keyof RunnerConfig>(k: K, v: RunnerConfig[K]) =>
    setCfg({ ...cfg, [k]: v });

  return (
    <div className="panel controls">
      <div className="group">
        <div className="field">
          <label htmlFor="cfg-scenario">scenario</label>
          <select id="cfg-scenario" value={cfg.scenario} disabled={busy} onChange={(e) => set("scenario", e.target.value)}>
            {SCENARIOS.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="field">
          <label htmlFor="cfg-policy">policy</label>
          <div className="policy-wrap">
            <select id="cfg-policy" value={cfg.policy} disabled={busy} onChange={(e) => set("policy", e.target.value)}>
              {policies.map((p) => (
                <option key={p} value={p}>
                  {mainMode ? (MAIN_POLICIES.find((mp) => mp.key === p)?.label ?? p) : p}
                </option>
              ))}
            </select>
            {policyLabel && <span className="policy-pill" title="active policy">{policyLabel}</span>}
          </div>
        </div>
        <div className="field">
          <label htmlFor="cfg-seed">seed</label>
          <input
            id="cfg-seed"
            type="number"
            value={cfg.seed}
            disabled={busy}
            onChange={(e) => set("seed", Number(e.target.value) || 0)}
          />
        </div>
        <div className="field">
          <label htmlFor="cfg-days">days</label>
          <input
            id="cfg-days"
            type="number"
            min={1}
            max={30}
            value={cfg.days}
            disabled={busy}
            onChange={(e) =>
              set("days", Math.min(30, Math.max(1, Number(e.target.value) || 1)))
            }
          />
        </div>
        <div className="field">
          <label htmlFor="cfg-speed">speed (sim-min/s)</label>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              id="cfg-speed"
              type="range"
              min={10}
              max={1000}
              step={10}
              value={cfg.speed ?? 60}
              disabled={busy}
              onChange={(e) => set("speed", Number(e.target.value))}
            />
            <span className="speed-val">{cfg.speed ?? "max"}</span>
          </div>
        </div>
      </div>
      <div className="divider" />
      <div className="actions">
        <button className="btn primary" disabled={busy || running} onClick={() => onAction("start")}>
          Start
        </button>
        <button className="btn" disabled={busy || !running || paused || finished} onClick={() => onAction("pause")}>
          Pause
        </button>
        <button className="btn" disabled={busy || !paused} onClick={() => onAction("resume")}>
          Resume
        </button>
        <button className="btn" disabled={busy || (running && !paused) || finished} onClick={() => onAction("step")}>
          Step +5
        </button>
        <button className="btn danger" disabled={busy} onClick={() => onAction("reset")}>
          Reset
        </button>
      </div>
    </div>
  );
}
