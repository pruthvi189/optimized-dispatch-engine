import { useCallback, useState } from "react";
import { Header } from "./components/Header";
import { KpiCards } from "./components/KpiCards";
import { KitchenPanel } from "./components/KitchenPanel";
import { RiderPanel } from "./components/RiderPanel";
import { DecisionTable } from "./components/DecisionTable";
import { EventLog } from "./components/EventLog";
import { Controls } from "./components/Controls";
import { ComparePanel } from "./components/ComparePanel";
import { RootCausePanel } from "./components/RootCausePanel";
import { ExperimentPanel } from "./components/ExperimentPanel";
import { DeliveryDistributionPanel } from "./components/DeliveryDistributionPanel";
import { DispatchDecisionVisual } from "./components/DispatchDecisionVisual";
import { useSimStream } from "./useSimStream";
import { useCompare } from "./useCompare";
import { api } from "./api";
import type { RunnerConfig } from "./types";

const DEFAULT_CFG: RunnerConfig = {
  scenario: "normal",
  seed: 42,
  policy: "nearest_heuristic",
  days: 1,
  speed: 60,
  step_minutes: 1,
};

type Action = "start" | "pause" | "resume" | "step" | "reset";
type Tab = "dashboard" | "experiments";

export default function App() {
  const [cfg, setCfg] = useState<RunnerConfig>(DEFAULT_CFG);
  const { snapshot, conn } = useSimStream(cfg);
  const compare = useCompare(cfg);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("dashboard");

  const onAction = useCallback(
    async (kind: Action) => {
      setError(null);
      try {
        if (kind === "start") {
          await api.reset(cfg);
          await api.start();
        } else if (kind === "pause") await api.pause();
        else if (kind === "resume") await api.resume();
        else if (kind === "step") await api.step(5);
        else await api.reset(cfg);
      } catch (err) {
        setError(String((err as Error).message ?? err));
      }
    },
    [cfg],
  );

  const busy = compare.state.running;

  return (
    <div className="app">
      <Header snap={snapshot} conn={conn} />
      {error && <div className="error-banner">{error}</div>}

      {/* Tab navigation */}
      <nav className="tabs">
        <button className={`tab ${tab === "dashboard" ? "active" : ""}`} onClick={() => setTab("dashboard")}>
          Dashboard
        </button>
        <button className={`tab ${tab === "experiments" ? "active" : ""}`} onClick={() => setTab("experiments")}>
          Experiments &amp; Diagnostics
        </button>
      </nav>

      {tab === "dashboard" && (
        <>
          {/* Controls — only Baseline / Optimized */}
          <section className="section">
            <Controls cfg={cfg} setCfg={setCfg} snap={snapshot} busy={busy} onAction={onAction} mainMode />
          </section>

          {/* Section 1: Baseline vs Optimized */}
          <section className="section">
            <div className="section-header">
              <h3>Baseline vs Optimized</h3>
              <span className="context-label">Seed {cfg.seed} · {cfg.days}-day simulation</span>
            </div>
            <KpiCards m={snapshot?.metrics ?? null} />
            <div className="panel">
              <div className="panel-head">
                <h3>Comparison</h3>
                <div className="head-actions">
                  {busy && <button className="btn" onClick={compare.cancel}>Cancel</button>}
                  <button className="btn primary" disabled={busy} onClick={compare.run}>
                    {busy ? "Running\u2026" : "Run compare"}
                  </button>
                </div>
              </div>
              <ComparePanel state={compare.state} seed={cfg.seed} days={cfg.days} />
            </div>
          </section>

          {/* Section 2: Why This Dispatch? */}
          <section className="section">
            <DispatchDecisionVisual decisions={snapshot?.recent_decisions ?? []} />
          </section>
        </>
      )}

      {tab === "experiments" && (
        <>
          {/* Controls — all policies */}
          <section className="section">
            <Controls cfg={cfg} setCfg={setCfg} snap={snapshot} busy={busy} onAction={onAction} />
          </section>

          <section className="section">
            <div className="cols state">
              <KitchenPanel kitchens={snapshot?.kitchens ?? []} />
              <RiderPanel riders={snapshot?.riders ?? []} />
            </div>
          </section>

          <section className="section">
            <div className="cols decisions">
              <DecisionTable decisions={snapshot?.recent_decisions ?? []} />
              <EventLog events={snapshot?.events ?? []} />
            </div>
          </section>

          <section className="section">
            <RootCausePanel finished={snapshot?.finished ?? false} />
          </section>

          <section className="section">
            <ExperimentPanel />
          </section>

          <section className="section">
            <DeliveryDistributionPanel />
          </section>
        </>
      )}
    </div>
  );
}
