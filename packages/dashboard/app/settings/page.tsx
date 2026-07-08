"use client";

import { useEffect, useState, useCallback } from "react";
import OnboardingCommand, { getServerHost } from "@/components/onboarding-command";
import { OllamaModelsSection } from "@/components/ollama-models-section";
import { RegistriesSection } from "@/components/registries-section";

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

interface JoinToken {
  id: string;
  label: string | null;
  tokenSuffix: string;
  expiresAt: string | null;
  usedAt: string | null;
  usedByNodeId: string | null;
  revokedAt: string | null;
  createdAt: string;
  status: "active" | "used" | "expired" | "revoked";
}

export default function SettingsPage() {
  const [tokens, setTokens] = useState<JoinToken[]>([]);
  const [agentVersion, setAgentVersion] = useState("...");
  const [rolling, setRolling] = useState(false);
  const [rollResult, setRollResult] = useState<string | null>(null);
  const serverHost = getServerHost();

  const loadTokens = useCallback(async () => {
    const res = await fetch(`${API}/api/tokens`);
    setTokens(await res.json());
  }, []);

  useEffect(() => {
    loadTokens();
    fetch(`${API}/api/agent/version`).then(r => r.json()).then(d => setAgentVersion(d.version));
  }, [loadTokens]);

  async function revokeToken(id: string) {
    await fetch(`${API}/api/tokens/${id}`, { method: "DELETE" });
    loadTokens();
  }

  async function updateAllAgents() {
    if (!window.confirm(`Roll all online agents to the bundled version (v${agentVersion})? Running deployments keep serving.`)) return;
    setRolling(true);
    setRollResult(null);
    try {
      const res = await fetch(`${API}/api/nodes/update-agent-all`, { method: "POST" });
      const d = await res.json();
      const disp = d.dispatched?.length ?? 0, skip = d.skipped?.length ?? 0, off = d.offline?.length ?? 0;
      setRollResult(
        disp > 0
          ? `Updating ${disp} node${disp > 1 ? "s" : ""} → v${d.version}${skip ? `; ${skip} already current` : ""}${off ? `; ${off} offline` : ""}.`
          : `All ${skip} online node${skip !== 1 ? "s" : ""} already on v${d.version}${off ? `; ${off} offline` : ""}.`,
      );
    } catch (e) {
      setRollResult(`Failed: ${String(e)}`);
    } finally {
      setRolling(false);
    }
  }

  const statusColor: Record<string, string> = {
    active: "text-green-400",
    used: "text-blue-400",
    expired: "text-yellow-400",
    revoked: "text-red-400",
  };

  return (
    <div className="space-y-8">
      <h1 className="text-2xl font-bold">Settings</h1>

      {/* Join Tokens Section */}
      <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">Join Tokens</h2>
        <p className="text-sm text-gray-400 mb-4">
          Generate tokens to bootstrap new nodes without SSH. Each token can be used once.
        </p>

        {/* Create Token + Command */}
        <div className="mb-6">
          <OnboardingCommand serverHost={serverHost} onCreated={loadTokens} />
        </div>

        {/* Token List */}
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-500 text-left border-b border-gray-800">
              <th className="pb-2 font-medium">Label</th>
              <th className="pb-2 font-medium">Token</th>
              <th className="pb-2 font-medium">Status</th>
              <th className="pb-2 font-medium">Created</th>
              <th className="pb-2 font-medium"></th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((t) => (
              <tr key={t.id} className="border-b border-gray-800/50">
                <td className="py-2">{t.label || <span className="text-gray-600">-</span>}</td>
                <td className="py-2 font-mono text-gray-400">...{t.tokenSuffix}</td>
                <td className={`py-2 ${statusColor[t.status] || "text-gray-400"}`}>{t.status}</td>
                <td className="py-2 text-gray-400">{new Date(t.createdAt).toLocaleDateString()}</td>
                <td className="py-2">
                  {t.status === "active" && (
                    <button
                      onClick={() => revokeToken(t.id)}
                      className="text-red-500 hover:text-red-400 text-xs transition-colors"
                    >
                      Revoke
                    </button>
                  )}
                </td>
              </tr>
            ))}
            {tokens.length === 0 && (
              <tr>
                <td colSpan={5} className="py-4 text-center text-gray-600">No tokens yet</td>
              </tr>
            )}
          </tbody>
        </table>
      </section>

      {/* Agent Info Section */}
      <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">Agent</h2>
        <div className="space-y-3 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Bundled agent version</span>
            <span className="font-mono">{agentVersion}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Bundle download</span>
            <a href={`${API}/api/agent/bundle`} className="text-green-400 hover:text-green-300 transition-colors">
              Download agent-bundle.tar.gz
            </a>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Roll all agents</span>
            <button
              onClick={updateAllAgents}
              disabled={rolling}
              className="text-xs px-3 py-1 rounded bg-green-600/20 text-green-300 hover:bg-green-600/30 disabled:opacity-50 transition-colors"
            >
              {rolling ? "Rolling…" : "Update all agents"}
            </button>
          </div>
          {rollResult && <p className="text-xs text-gray-500">{rollResult}</p>}
        </div>
      </section>

      {/* Install Command Section */}
      <section className="bg-gray-900 rounded-lg p-6 border border-gray-800">
        <h2 className="text-lg font-semibold mb-4">Quick Install</h2>
        <p className="text-sm text-gray-400 mb-3">
          Run this on a new machine to install the DGX Manager agent. Replace <code className="text-green-400">TOKEN</code> with a join token.
        </p>
        <div className="bg-gray-950 rounded p-3 font-mono text-xs break-all">
          curl -sL http://{serverHost}/api/agent/install.sh | sudo bash -s -- --token TOKEN
        </div>
      </section>

      <RegistriesSection />
      <OllamaModelsSection />
    </div>
  );
}
