"use client";

import { useEffect, useState } from "react";
import { apiFetch } from "@/lib/api";

interface CreatedToken {
  id: string;
  token: string;
}

interface TokenRecord {
  id: string;
  usedAt: string | null;
  usedByNodeId: string | null;
}

interface NodeSummary {
  id: string;
  name: string;
  status: string;
  gpuModel: string | null;
  vramTotal: number | null;
  agentVersion: string | null;
  arch: string | null;
  dockerAvailable: boolean;
  lastSeen: string | null;
}

interface Props {
  serverHost: string;
  onCreated?: (token: CreatedToken) => void;
}

export function getServerHost(): string {
  const api = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";
  try {
    const url = new URL(api);
    return `${url.hostname}:${url.port || "4000"}`;
  } catch {
    return "localhost:4000";
  }
}

export default function OnboardingCommand({ serverHost, onCreated }: Props) {
  const [label, setLabel] = useState("");
  const [created, setCreated] = useState<CreatedToken | null>(null);
  const [creating, setCreating] = useState(false);
  const [copied, setCopied] = useState(false);

  const installCommand = created
    ? `curl -sL http://${serverHost}/api/agent/install.sh | sudo bash -s -- --token ${created.token}`
    : "";

  async function handleCreate() {
    setCreating(true);
    try {
      const data = await apiFetch<CreatedToken>("/api/tokens", {
        method: "POST",
        body: JSON.stringify({ label: label || null }),
      });
      setCreated(data);
      setCopied(false);
      onCreated?.(data);
    } catch (err) {
      alert(String(err));
    } finally {
      setCreating(false);
    }
  }

  async function handleCopy() {
    try {
      if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(installCommand);
      } else {
        // Fallback for non-secure contexts (e.g. http://<ip>:3000)
        const ta = document.createElement("textarea");
        ta.value = installCommand;
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
      alert("Copy failed — please select and copy the command manually.");
    }
  }

  function reset() {
    setCreated(null);
    setLabel("");
  }

  if (created) {
    return (
      <div className="space-y-3">
        <p className="text-sm text-green-400 font-medium">
          Token created. Copy and run this on the target node:
        </p>
        <div className="bg-gray-950 rounded p-3 font-mono text-xs break-all border border-gray-800">
          {installCommand}
        </div>
        <div className="flex gap-2 items-center">
          <button
            onClick={handleCopy}
            className={`px-3 py-1.5 rounded text-xs font-medium transition-colors inline-flex items-center gap-1.5 ${
              copied
                ? "bg-emerald-500 text-white"
                : "bg-green-600 hover:bg-green-500 text-white"
            }`}
          >
            {copied ? (
              <>
                <span aria-hidden>&#10003;</span> Copied
              </>
            ) : (
              "Copy command"
            )}
          </button>
          <button
            onClick={reset}
            className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
          >
            Create another
          </button>
          {copied && (
            <span className="text-xs text-emerald-400">Command copied to clipboard.</span>
          )}
        </div>
        <RegistrationCheck tokenId={created.id} />
      </div>
    );
  }

  return (
    <div className="flex gap-2">
      <input
        type="text"
        placeholder="Label (optional, e.g. dgx-spark-01)"
        value={label}
        onChange={(e) => setLabel(e.target.value)}
        className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
      />
      <button
        onClick={handleCreate}
        disabled={creating}
        className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:bg-gray-600 rounded text-sm font-medium transition-colors"
      >
        {creating ? "Creating..." : "Create token"}
      </button>
    </div>
  );
}

function RegistrationCheck({ tokenId }: { tokenId: string }) {
  const [nodeId, setNodeId] = useState<string | null>(null);
  const [node, setNode] = useState<NodeSummary | null>(null);
  const [expectedVersion, setExpectedVersion] = useState<string | null>(null);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (nodeId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const tokens = await apiFetch<TokenRecord[]>("/api/tokens");
        if (cancelled) return;
        const me = tokens.find((t) => t.id === tokenId);
        if (me?.usedByNodeId) setNodeId(me.usedByNodeId);
      } catch {
        /* ignore transient errors */
      }
    };
    tick();
    const interval = setInterval(tick, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [tokenId, nodeId]);

  useEffect(() => {
    if (!nodeId) return;
    let cancelled = false;
    const tick = async () => {
      try {
        const [n, v] = await Promise.all([
          apiFetch<NodeSummary>(`/api/nodes/${nodeId}`),
          apiFetch<{ version: string }>("/api/agent/version"),
        ]);
        if (cancelled) return;
        setNode(n);
        setExpectedVersion(v.version);
      } catch {
        /* ignore transient errors */
      }
    };
    tick();
    const interval = setInterval(tick, 3000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [nodeId]);

  if (!nodeId) {
    return (
      <div className="mt-4 flex items-center gap-2 text-sm text-gray-400 rounded border border-gray-800 bg-gray-950/60 p-3">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
        Waiting for the node to connect…
      </div>
    );
  }

  if (!node) {
    return (
      <div className="mt-4 text-sm text-gray-400 rounded border border-gray-800 bg-gray-950/60 p-3">
        Loading node details…
      </div>
    );
  }

  const lastSeenMs = node.lastSeen ? now - new Date(node.lastSeen).getTime() : Infinity;
  const checks: { label: string; state: "ok" | "warn" | "fail"; detail?: string }[] = [
    {
      label: "Agent connected",
      state: node.status === "online" ? "ok" : "fail",
      detail: node.status,
    },
    {
      label: "Hostname",
      state: node.name && node.name !== "unknown" ? "ok" : "warn",
      detail: node.name,
    },
    {
      label: "Architecture",
      state: node.arch ? "ok" : "warn",
      detail: node.arch || "not reported",
    },
    {
      label: "GPU",
      state: node.gpuModel ? "ok" : "fail",
      detail: node.gpuModel
        ? `${node.gpuModel}${node.vramTotal ? ` (${node.vramTotal} MB)` : ""}`
        : "not detected",
    },
    {
      label: "Docker",
      state: node.dockerAvailable ? "ok" : "fail",
    },
    {
      label: "Metrics flowing",
      state: lastSeenMs < 15_000 ? "ok" : lastSeenMs < 60_000 ? "warn" : "fail",
      detail:
        lastSeenMs < 60_000
          ? `last seen ${Math.max(0, Math.round(lastSeenMs / 1000))}s ago`
          : "stale / no metrics",
    },
    {
      label: "Agent version",
      state:
        expectedVersion && node.agentVersion === expectedVersion
          ? "ok"
          : expectedVersion && node.agentVersion
            ? "warn"
            : "warn",
      detail:
        node.agentVersion && expectedVersion
          ? node.agentVersion === expectedVersion
            ? node.agentVersion
            : `${node.agentVersion} (expected ${expectedVersion})`
          : node.agentVersion || "unknown",
    },
  ];

  const allOk = checks.every((c) => c.state === "ok");

  return (
    <div className="mt-4 space-y-2 rounded border border-gray-800 bg-gray-950/60 p-3">
      <p className={`text-sm font-medium ${allOk ? "text-green-400" : "text-yellow-400"}`}>
        {allOk
          ? "Node registered and operating normally."
          : "Node registered — review the checks below."}
      </p>
      <ul className="space-y-1">
        {checks.map((c) => (
          <CheckRow key={c.label} {...c} />
        ))}
      </ul>
    </div>
  );
}

function CheckRow({
  label,
  state,
  detail,
}: {
  label: string;
  state: "ok" | "warn" | "fail";
  detail?: string;
}) {
  const mark = state === "ok" ? "\u2713" : state === "warn" ? "!" : "\u2717";
  const color =
    state === "ok"
      ? "text-green-400"
      : state === "warn"
        ? "text-yellow-400"
        : "text-red-400";
  return (
    <li className="flex items-center gap-2 text-xs">
      <span className={`${color} font-bold w-4 text-center`}>{mark}</span>
      <span className="text-gray-300 min-w-[140px]">{label}</span>
      {detail && <span className="text-gray-500">{detail}</span>}
    </li>
  );
}
