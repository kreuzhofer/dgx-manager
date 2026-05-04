"use client";

import { LogViewer } from "./log-viewer";

interface FinetuneLogProps {
  jobId: string;
  logs: string;
}

/**
 * Thin shim around the shared LogViewer for backwards compatibility with
 * existing call sites. Tail-follow + "⬇ Follow" button behavior lives in
 * LogViewer; this component just preserves the named API and the
 * "Waiting for logs..." placeholder.
 */
export function FinetuneLog({ jobId: _jobId, logs }: FinetuneLogProps) {
  return <LogViewer content={logs || "Waiting for logs..."} />;
}
