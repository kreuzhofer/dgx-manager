"use client";

import { useEffect, useState } from "react";
import { fetchClaudeLaunch, type ClaudeLaunch } from "@/lib/claude-launch";

type Shell = "bash" | "powershell";

type Props = {
  deploymentId: string;
  deploymentLabel: string;
  onClose: () => void;
};

export function ClaudeLaunchModal({ deploymentId, deploymentLabel, onClose }: Props) {
  const [data, setData] = useState<ClaudeLaunch | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shell, setShell] = useState<Shell>("bash");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchClaudeLaunch(deploymentId)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, [deploymentId]);

  const snippet = data ? data.shells[shell] : "";

  async function handleCopy() {
    try {
      if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(snippet);
      } else {
        // Fallback for non-secure contexts (dashboard is often http://<ip>:3000)
        const ta = document.createElement("textarea");
        ta.value = snippet;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.top = "0";
        ta.style.left = "0";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        if (!ok) throw new Error("execCommand('copy') returned false");
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("copy failed", err);
      alert("Copy failed — please select and copy the snippet manually.");
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-lg p-6 w-[640px] max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-xl font-semibold mb-1">Launch Claude Code</h2>
        <p className="text-sm text-gray-400 mb-4">Target: {deploymentLabel}</p>

        {error && (
          <div className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded p-3 mb-4">
            {error}
          </div>
        )}

        {!data && !error && <p className="text-sm text-gray-400">Resolving served model…</p>}

        {data && (
          <>
            <div className="flex gap-2 mb-3">
              {(["bash", "powershell"] as Shell[]).map((s) => (
                <button
                  key={s}
                  onClick={() => setShell(s)}
                  className={`text-xs px-3 py-1.5 rounded transition-colors ${
                    shell === s ? "bg-green-600 text-white" : "bg-gray-800 hover:bg-gray-700 text-gray-300"
                  }`}
                >
                  {s === "bash" ? "bash / zsh" : "PowerShell"}
                </button>
              ))}
            </div>

            <pre className="bg-gray-950 rounded p-3 font-mono text-xs whitespace-pre-wrap break-all border border-gray-800 mb-3">
              {snippet}
            </pre>

            <div className="flex gap-2 items-center mb-4">
              <button
                onClick={handleCopy}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors inline-flex items-center gap-1.5 ${
                  copied ? "bg-emerald-500 text-white" : "bg-green-600 hover:bg-green-500 text-white"
                }`}
              >
                {copied ? (<><span aria-hidden>&#10003;</span> Copied</>) : "Copy snippet"}
              </button>
              <button
                onClick={onClose}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
              >
                Close
              </button>
            </div>

            <p className="text-xs text-gray-500">
              Paste into a new shell, then run <code className="text-gray-400">claude</code>. Requires this
              model served with tool-calling enabled on a vLLM build that exposes{" "}
              <code className="text-gray-400">/v1/messages</code>.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
