"use client";

import { useEffect, useState, useCallback } from "react";

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
  fullToken?: string; // Only present immediately after creation
}

export default function SettingsPage() {
  const [tokens, setTokens] = useState<JoinToken[]>([]);
  const [settings, setSettings] = useState<Record<string, string>>({});
  const [newTokenLabel, setNewTokenLabel] = useState("");
  const [createdToken, setCreatedToken] = useState<{ token: string; id: string } | null>(null);
  const [copied, setCopied] = useState(false);
  const [agentVersion, setAgentVersion] = useState("...");
  const [serverHost, setServerHost] = useState("");

  const loadTokens = useCallback(async () => {
    const res = await fetch(`${API}/api/tokens`);
    setTokens(await res.json());
  }, []);

  const loadSettings = useCallback(async () => {
    const res = await fetch(`${API}/api/settings`);
    setSettings(await res.json());
  }, []);

  useEffect(() => {
    loadTokens();
    loadSettings();
    fetch(`${API}/api/agent/version`).then(r => r.json()).then(d => setAgentVersion(d.version));
    // Derive server host from the API URL
    try {
      const url = new URL(API);
      setServerHost(`${url.hostname}:${url.port || "4000"}`);
    } catch {
      setServerHost("localhost:4000");
    }
  }, [loadTokens, loadSettings]);

  async function createToken() {
    const res = await fetch(`${API}/api/tokens`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label: newTokenLabel || null }),
    });
    const data = await res.json();
    setCreatedToken({ token: data.token, id: data.id });
    setNewTokenLabel("");
    setCopied(false);
    loadTokens();
  }

  async function revokeToken(id: string) {
    await fetch(`${API}/api/tokens/${id}`, { method: "DELETE" });
    loadTokens();
  }

  function getInstallCommand(token: string) {
    return `curl -sL http://${serverHost}/api/agent/install.sh | sudo bash -s -- --token ${token}`;
  }

  async function copyToClipboard(text: string) {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
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

        {/* Create Token */}
        <div className="flex gap-3 mb-6">
          <input
            type="text"
            placeholder="Label (optional)"
            value={newTokenLabel}
            onChange={(e) => setNewTokenLabel(e.target.value)}
            className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-green-500"
          />
          <button
            onClick={createToken}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 rounded text-sm font-medium transition-colors"
          >
            Generate Token
          </button>
        </div>

        {/* Created Token Display */}
        {createdToken && (
          <div className="mb-6 p-4 bg-gray-800 rounded border border-green-800">
            <p className="text-sm text-green-400 font-medium mb-2">Token created! Copy the install command:</p>
            <div className="bg-gray-950 rounded p-3 font-mono text-xs break-all mb-2">
              {getInstallCommand(createdToken.token)}
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => copyToClipboard(getInstallCommand(createdToken.token))}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
              >
                {copied ? "Copied!" : "Copy Command"}
              </button>
              <button
                onClick={() => copyToClipboard(createdToken.token)}
                className="px-3 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
              >
                Copy Token Only
              </button>
              <button
                onClick={() => setCreatedToken(null)}
                className="px-3 py-1 text-gray-500 hover:text-gray-300 text-xs transition-colors"
              >
                Dismiss
              </button>
            </div>
          </div>
        )}

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
    </div>
  );
}
