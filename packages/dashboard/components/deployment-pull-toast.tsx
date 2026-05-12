"use client";

import { useEffect, useRef } from "react";
import { toast } from "sonner";
import { useSSE, type SseEvent } from "@/lib/sse";

interface ToastState {
  toastId: string | number;
  modelName?: string;
}

export function DeploymentPullToast() {
  const active = useRef<Map<string, ToastState>>(new Map());

  const handle = (event: SseEvent) => {
    if (event.type === "deployment:progress") {
      const p = event.payload as {
        deploymentId: string;
        phase: string;
        phaseProgress: number;
        current?: number | null;
        total?: number | null;
      };
      if (p.phase !== "downloading") return;
      const prev = active.current.get(p.deploymentId);
      const message = `Pulling model… ${p.phaseProgress}%`;
      const description =
        p.current != null && p.total != null
          ? `${formatBytes(p.current)} / ${formatBytes(p.total)}`
          : "Streaming layers from Ollama registry";
      if (prev) {
        toast.message(message, { id: prev.toastId, description, duration: Infinity });
      } else {
        const id = toast.loading(message, { description, duration: Infinity });
        active.current.set(p.deploymentId, { toastId: id });
      }
    }
    if (event.type === "deployment:status") {
      const p = event.payload as { deploymentId: string; status: string; error?: string };
      const prev = active.current.get(p.deploymentId);
      if (!prev) return;
      if (p.status === "running") {
        toast.success("Model loaded", { id: prev.toastId, duration: 4000 });
        active.current.delete(p.deploymentId);
      } else if (["failed", "stopped", "evicted"].includes(p.status)) {
        toast.error(p.status === "failed" ? "Pull failed" : "Deployment stopped", {
          id: prev.toastId,
          description: p.error,
          duration: 6000,
        });
        active.current.delete(p.deploymentId);
      }
    }
    if (event.type === "deployment:deleted") {
      const p = event.payload as { deploymentId: string };
      const prev = active.current.get(p.deploymentId);
      if (prev) {
        toast.dismiss(prev.toastId);
        active.current.delete(p.deploymentId);
      }
    }
  };

  // Use a ref-stable handler since useSSE captures it once.
  const handleRef = useRef(handle);
  handleRef.current = handle;
  useSSE((e) => handleRef.current(e));

  useEffect(() => {
    // Dismiss any orphaned toasts on unmount (e.g. dev HMR reload).
    return () => {
      for (const { toastId } of active.current.values()) toast.dismiss(toastId);
      active.current.clear();
    };
  }, []);

  return null;
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
