"use client";

import { useEffect, useRef } from "react";

interface FinetuneLogProps {
  jobId: string;
  logs: string;
}

export function FinetuneLog({ jobId, logs }: FinetuneLogProps) {
  const ref = useRef<HTMLPreElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.scrollTop = ref.current.scrollHeight;
    }
  }, [logs]);

  return (
    <pre
      ref={ref}
      data-log-viewer
      className="mt-3 bg-black/50 border border-gray-800 rounded p-3 text-xs text-gray-400 font-mono max-h-64 overflow-y-auto whitespace-pre-wrap"
    >
      {logs || "Waiting for logs..."}
    </pre>
  );
}
