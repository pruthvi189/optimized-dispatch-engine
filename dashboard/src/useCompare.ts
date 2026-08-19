import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { Metrics, RunnerConfig } from "./types";

export interface CompareState {
  baseline: Metrics | null;
  optimized: Metrics | null;
  running: boolean;
  error: string | null;
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/**
 * Runs baseline (nearest_heuristic) then optimized (joint_optimizer) to
 * completion on the same (scenario, seed, days) at max speed, and returns
 * each policy's final metrics.
 */
export function useCompare(cfg: RunnerConfig) {
  const [state, setState] = useState<CompareState>({
    baseline: null,
    optimized: null,
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
    setState({ baseline: null, optimized: null, running: true, error: null });
    try {
      const results: Metrics[] = [];
      const days = cfgRef.current.days;
      const maxIters = Math.max(600, days * 300);
      for (const policy of ["nearest_heuristic", "joint_optimizer"] as const) {
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
      await api.reset(cfgRef.current).catch(() => {});
      setState({ baseline: results[0], optimized: results[1], running: false, error: null });
    } catch (err) {
      setState((s) => ({ ...s, running: false, error: String((err as Error).message ?? err) }));
    } finally {
      runningRef.current = false;
    }
  }, []);

  useEffect(() => () => { runningRef.current = false; }, []);

  return { state, run, cancel };
}
