"use client";

import { useEffect, useRef, useState, useCallback } from "react";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export type SseEvent = { type: string; payload: Record<string, unknown> };
type SseHandler = (event: SseEvent) => void;

export function useSSE(onEvent: SseHandler, onReconnect?: () => void) {
  const sourceRef = useRef<EventSource | null>(null);
  const [connected, setConnected] = useState(false);
  const wasConnected = useRef(false);

  const stableHandler = useRef(onEvent);
  stableHandler.current = onEvent;
  const stableReconnect = useRef(onReconnect);
  stableReconnect.current = onReconnect;

  const connect = useCallback(() => {
    const es = new EventSource(`${API_BASE}/api/events`);
    sourceRef.current = es;

    es.onopen = () => {
      setConnected(true);
      // Refresh data on reconnect (not on first connect)
      if (wasConnected.current) {
        stableReconnect.current?.();
      }
      wasConnected.current = true;
    };
    es.onerror = () => {
      setConnected(false);
      es.close();
      setTimeout(connect, 3000);
    };
    es.onmessage = (event) => {
      try {
        stableHandler.current(JSON.parse(event.data));
      } catch { /* ignore */ }
    };
  }, []);

  useEffect(() => {
    connect();
    return () => sourceRef.current?.close();
  }, [connect]);

  return { connected };
}
