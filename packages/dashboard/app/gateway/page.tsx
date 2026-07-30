"use client";

import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

interface PoolMember {
  deploymentId: string;
  node: string;
  runtime: string | null;
  port: number | null;
  inflight: number;
  serving: boolean;
  reason?: string;
  detail?: string;
}

interface Pool {
  publishedName: string;
  members: PoolMember[];
  servingCount: number;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

const RUNTIME_COLORS: Record<string, string> = {
  vllm: "bg-blue-900/50 text-blue-300",
  ollama: "bg-purple-900/50 text-purple-300",
};

/** Why a member is not serving, in the words an operator would use. */
const REASON_LABELS: Record<string, string> = {
  "not-running": "not running",
  "no-port": "no port bound",
  "no-node-address": "no node address",
  "agent-offline": "agent offline",
};

function CopyableUrl({ url }: { url: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(url);
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      }}
      className="group flex items-center gap-2 rounded border border-gray-700 bg-gray-900 px-3 py-2 font-mono text-sm text-gray-200 hover:border-gray-500"
      title="Copy to clipboard"
    >
      <span>{url}</span>
      <span className="text-xs text-gray-500 group-hover:text-gray-300">
        {copied ? "copied" : "copy"}
      </span>
    </button>
  );
}

export default function GatewayPage() {
  const [pools, setPools] = useState<Pool[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState(false);

  const load = useCallback(async () => {
    try {
      const data = await apiFetch<{ pools: Pool[] }>("/api/gateway");
      setPools(data.pools);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load");
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    void load();
    // In-flight counts change with traffic, not with a database write, so there
    // is no event to subscribe to — poll while the page is open.
    const timer = setInterval(() => void load(), 3000);
    return () => clearInterval(timer);
  }, [load]);

  const totalServing = pools.reduce((n, p) => n + p.servingCount, 0);

  return (
    <div>
      <h1 className="text-2xl font-bold mb-2">Gateway</h1>
      <p className="text-gray-400 mb-6">
        One OpenAI-compatible URL for every running deployment. Send a model name; the gateway
        routes to whichever deployment serves it.
      </p>

      <div className="mb-8 flex flex-wrap items-center gap-3">
        <CopyableUrl url={`${API_BASE}/v1`} />
        <span className="text-sm text-gray-500">
          {totalServing} {totalServing === 1 ? "member" : "members"} serving{" "}
          {pools.length} {pools.length === 1 ? "model" : "models"}
        </span>
      </div>

      {error && (
        <div className="mb-6 rounded border border-red-900 bg-red-950/50 px-4 py-3 text-red-300">
          {error}
        </div>
      )}

      {loaded && pools.length === 0 && !error && (
        <div className="rounded border border-gray-800 bg-gray-900/50 px-4 py-8 text-center text-gray-400">
          <p className="mb-1">Nothing is published yet.</p>
          <p className="text-sm text-gray-500">
            A deployment appears here once it starts serving and the gateway knows the name it
            answers to.
          </p>
        </div>
      )}

      <div className="space-y-4">
        {pools.map((pool) => (
          <div key={pool.publishedName} className="rounded border border-gray-800 bg-gray-900/50">
            <div className="flex flex-wrap items-baseline justify-between gap-2 border-b border-gray-800 px-4 py-3">
              <code className="font-mono text-lg text-gray-100">{pool.publishedName}</code>
              <span className="text-sm text-gray-400">
                {pool.servingCount} of {pool.members.length}{" "}
                {pool.members.length === 1 ? "member" : "members"} serving
              </span>
            </div>

            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase tracking-wide text-gray-500">
                  <th className="px-4 py-2 font-medium">Node</th>
                  <th className="px-4 py-2 font-medium">Runtime</th>
                  <th className="px-4 py-2 font-medium">Port</th>
                  <th className="px-4 py-2 font-medium">In flight</th>
                  <th className="px-4 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {pool.members.map((m) => (
                  <tr key={m.deploymentId} className="border-t border-gray-800/70">
                    <td className="px-4 py-2 text-gray-200">{m.node}</td>
                    <td className="px-4 py-2">
                      {m.runtime && (
                        <span
                          className={`rounded px-2 py-0.5 text-xs ${
                            RUNTIME_COLORS[m.runtime] ?? "bg-gray-800 text-gray-300"
                          }`}
                        >
                          {m.runtime}
                        </span>
                      )}
                    </td>
                    <td className="px-4 py-2 font-mono text-gray-400">{m.port ?? "—"}</td>
                    <td className="px-4 py-2 font-mono text-gray-300">{m.inflight}</td>
                    <td className="px-4 py-2">
                      {m.serving ? (
                        <span className="rounded bg-green-900/50 px-2 py-0.5 text-xs text-green-300">
                          serving
                        </span>
                      ) : (
                        <span
                          className="rounded bg-amber-900/50 px-2 py-0.5 text-xs text-amber-300"
                          title={m.detail}
                        >
                          excluded — {REASON_LABELS[m.reason ?? ""] ?? m.reason}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
