import { useCallback, useEffect, useRef, useState } from "react";
import type { Snapshot } from "./types";
import { wsUrl } from "./api";

export type ConnState = "connecting" | "connected" | "disconnected";

/**
 * Connects to /ws and keeps a live Snapshot. Reconnects with exponential
 * backoff (0.5s -> 2s cap). The URL is rebuilt from the latest control
 * selection so a reconnect re-syncs the server to the same run params.
 */
export function useSimStream(cfg: { scenario: string; seed: number; policy: string; speed: number | null }) {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [conn, setConn] = useState<ConnState>("connecting");
  const wsRef = useRef<WebSocket | null>(null);
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;
  const retryDelayRef = useRef(500);

  const connect = useCallback(() => {
    const ws = new WebSocket(wsUrl(cfgRef.current));
    wsRef.current = ws;
    setConn("connecting");

    ws.onopen = () => {
      retryDelayRef.current = 500;
      setConn("connected");
    };
    ws.onmessage = (ev) => {
      try {
        setSnapshot(JSON.parse(ev.data as string) as Snapshot);
      } catch {
        /* malformed frame - ignore */
      }
    };
    ws.onclose = () => {
      setConn("disconnected");
      if (wsRef.current !== ws) return;
      const delay = retryDelayRef.current;
      retryDelayRef.current = Math.min(delay * 2, 2000);
      window.setTimeout(() => {
        if (wsRef.current === ws) connect();
      }, delay);
    };
    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    const ws = wsRef.current;
    return () => {
      wsRef.current = null;
      ws?.close();
    };
  }, [connect]);

  return { snapshot, conn };
}
