import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Metrics, RunnerConfig } from "./types";

export interface CompareState {
  immediate: Metrics | null;
  adaptive: Metrics | null;
  running: boolean;
  error: string | null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Runs immediate then adaptive to completion on the same (scenario, seed, days)
 * at max speed, and returns each policy's final metrics. Uses only the Phase 4
 * REST API. Cancel abandons the current step.
 */
export function useCompare(cfg: RunnerConfig) {
  const [state, setState] = useState<CompareState>({
    immediate: null,
    adaptive: null,
    running: false,
    error: null,
  });
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;
  const runningRef = useRef(false);

  const cancel = useCallback(() => {
    runningRef.current = false;
    setState((s) => ({ ...s, running: false }));
  }, []);

  const run = useCallback(async () => {
    runningRef.current = true;
    setState({ immediate: null, adaptive: null, running: true, error: null });
    try {
      const results: Metrics[] = [];
      // Longer runs need a bigger poll budget: 100ms per tick, scaled by days.
      const days = cfgRef.current.days;
      const maxIters = Math.max(600, days * 300);
      for (const policy of ["immediate", "adaptive"] as const) {
        if (!runningRef.current) return;
        const base = { ...cfgRef.current, policy, speed: null };
        await api.reset(base);
        if (!runningRef.current) return;
        await api.start();
        for (let i = 0; i < maxIters; i++) {
          await sleep(100);
          if (!runningRef.current) return;
          const s = await api.status();
          if (s.finished) {
            results.push(s.metrics);
            break;
          }
        }
      }
      if (results.length !== 2) {
        throw new Error(
          `compare timed out after ${(maxIters * 0.1).toFixed(0)}s (${days} day run) — ` +
            "neither policy finished. Increase the poll budget or check the backend runner.",
        );
      }
      // Restore the runner to the user's config so the live UI stream re-syncs.
      await api.reset(cfgRef.current).catch(() => {});
      setState({ immediate: results[0], adaptive: results[1], running: false, error: null });
    } catch (err) {
      setState((s) => ({ ...s, running: false, error: String((err as Error).message ?? err) }));
    } finally {
      runningRef.current = false;
    }
  }, []);

  // abort on unmount
  useEffect(() => () => { runningRef.current = false; }, []);

  return { state, run, cancel };
}
