import type { RunnerConfig, Snapshot } from "./types";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return (await res.json()) as T;
}

const post = (path: string, body?: unknown) =>
  req<Snapshot>(path, {
    method: "POST",
    body: body === undefined ? undefined : JSON.stringify(body),
  });

export const api = {
  status: () => req<Snapshot>("/sim/status"),
  config: () => req<RunnerConfig>("/sim/config"),
  start: () => post("/sim/start"),
  pause: () => post("/sim/pause"),
  resume: () => post("/sim/resume"),
  step: (minutes = 5) => post("/sim/step", { minutes }),
  reset: (cfg: RunnerConfig) => post("/sim/reset", cfg),
};

export function wsUrl(cfg: Pick<RunnerConfig, "scenario" | "seed" | "policy" | "speed">): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const speed = cfg.speed ?? 60;
  return `${proto}://${location.host}/ws?scenario=${encodeURIComponent(cfg.scenario)}&seed=${cfg.seed}&policy=${encodeURIComponent(cfg.policy)}&speed=${speed}`;
}
