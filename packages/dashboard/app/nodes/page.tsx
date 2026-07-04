"use client";

import { useEffect, useState, useCallback } from "react";
import { apiFetch, reseedKnownHosts } from "@/lib/api";
import { useSSE, type SseEvent } from "@/lib/sse";
import OnboardingCommand, { getServerHost } from "@/components/onboarding-command";

interface PrereqCheck {
  name: string;
  status: "green" | "yellow" | "red";
  detail: string;
}

interface ProvisionReport {
  reachable: boolean;
  sudoAvailable: boolean;
  systemInfo: string;
  checks: PrereqCheck[];
}

interface Node {
  id: string;
  name: string;
  ipAddress: string;
  status: string;
  provisionStatus: string;
  provisionLog: string | null;
  gpuModel: string | null;
  agentVersion: string | null;
  arch: string | null;
  dockerAvailable: boolean;
  ollamaInstalled: boolean;
  createdAt: string;
}

interface ProvisionStep {
  step: string;
  status: string;
  detail?: string;
}

const checkBadge: Record<string, string> = {
  green: "bg-green-900/60 text-green-300 border-green-800",
  yellow: "bg-yellow-900/60 text-yellow-300 border-yellow-800",
  red: "bg-red-900/60 text-red-300 border-red-800",
  checking: "bg-blue-900/60 text-blue-300 border-blue-800 animate-pulse",
  installing: "bg-blue-900/60 text-blue-300 border-blue-800 animate-pulse",
  installed: "bg-green-900/60 text-green-300 border-green-800",
};

const provisionBadge: Record<string, string> = {
  pending: "bg-gray-700 text-gray-300",
  auditing: "bg-blue-900 text-blue-300",
  audited: "bg-green-900 text-green-300",
  provisioning: "bg-blue-900 text-blue-300",
  provisioned: "bg-green-900 text-green-300",
  "agent-deployed": "bg-green-900 text-green-300",
  unreachable: "bg-red-900 text-red-300",
  failed: "bg-red-900 text-red-300",
};

export default function NodesPage() {
  const [nodes, setNodes] = useState<Node[]>([]);
  const [expectedVersion, setExpectedVersion] = useState<string | null>(null);
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [upgrading, setUpgrading] = useState<Record<string, boolean>>({});
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [savingRename, setSavingRename] = useState(false);
  const [reseeding, setReseeding] = useState(false);
  // Offboarding modal state. `offboarding` holds the node being removed;
  // `offboardPhase` drives the modal UI (spinner vs. the timed-out "remove
  // anyhow" prompt); `offboardError` surfaces an unexpected failure.
  const [offboarding, setOffboarding] = useState<Node | null>(null);
  const [offboardPhase, setOffboardPhase] = useState<"working" | "timed-out">("working");
  const [offboardError, setOffboardError] = useState<string | null>(null);
  const serverHost = getServerHost();

  const startRename = (node: Node) => {
    setRenamingId(node.id);
    setRenameDraft(node.name);
    setRenameError(null);
  };

  const cancelRename = () => {
    setRenamingId(null);
    setRenameDraft("");
    setRenameError(null);
  };

  const saveRename = async (nodeId: string) => {
    const newName = renameDraft.trim();
    if (!newName) return;
    setSavingRename(true);
    setRenameError(null);
    try {
      const updated = await apiFetch<Node>(`/api/nodes/${nodeId}`, {
        method: "PATCH",
        body: JSON.stringify({ name: newName }),
      });
      setNodes((prev) => prev.map((n) => (n.id === nodeId ? { ...n, name: updated.name } : n)));
      cancelRename();
    } catch (err) {
      setRenameError(err instanceof Error ? err.message : String(err));
    } finally {
      setSavingRename(false);
    }
  };

  // Real-time provision steps per node
  const [liveSteps, setLiveSteps] = useState<Record<string, ProvisionStep[]>>({});

  const [activeDeployments, setActiveDeployments] = useState<Record<string, number>>({});

  const loadNodes = () => {
    apiFetch<Node[]>("/api/nodes").then(setNodes).catch(console.error);
    apiFetch<{ version: string }>("/api/nodes/agent-version")
      .then((d) => setExpectedVersion(d.version))
      .catch(console.error);
    apiFetch<{ nodeId: string; status: string }[]>("/api/deployments")
      .then((deps) => {
        const counts: Record<string, number> = {};
        for (const d of deps) {
          if (["pending", "running", "starting", "restarting"].includes(d.status)) {
            counts[d.nodeId] = (counts[d.nodeId] || 0) + 1;
          }
        }
        setActiveDeployments(counts);
      })
      .catch(console.error);
  };

  const nodeIsBusy = (node: Node) => (activeDeployments[node.id] || 0) > 0;

  const isOutdated = (node: Node) => {
    if (!node.agentVersion || !expectedVersion) return false;
    const agent = node.agentVersion.split(".").map(Number);
    const expected = expectedVersion.split(".").map(Number);
    for (let i = 0; i < 3; i++) {
      if ((agent[i] || 0) < (expected[i] || 0)) return true;
      if ((agent[i] || 0) > (expected[i] || 0)) return false;
    }
    return false; // equal = not outdated
  };

  const upgradeAgent = async (nodeId: string) => {
    setUpgrading((prev) => ({ ...prev, [nodeId]: true }));
    try {
      await apiFetch(`/api/nodes/${nodeId}/update-agent`, { method: "POST" });
    } catch (err) {
      alert(String(err));
    } finally {
      setUpgrading((prev) => ({ ...prev, [nodeId]: false }));
    }
  };

  const handleReseed = async () => {
    setReseeding(true);
    try {
      const report = await reseedKnownHosts();
      const okCount = report.perNode.filter((p) => p.ok).length;
      alert(`SSH trust reseeded: ${okCount}/${report.perNode.length} nodes, ${report.trustedIps.length} IPs.`);
    } catch (err) {
      alert(`Reseed failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setReseeding(false);
    }
  };

  useEffect(() => {
    loadNodes();
  }, []);

  const handleSSE = useCallback((event: SseEvent) => {
    if (event.type === "node:provision") {
      const { nodeId, step, status, detail, provisionStatus, report } =
        event.payload as {
          nodeId: string;
          step: string;
          status: string;
          detail?: string;
          provisionStatus?: string;
          report?: ProvisionReport;
        };

      // Append live step
      setLiveSteps((prev) => {
        const existing = prev[nodeId] || [];
        // Update existing step or append new
        const idx = existing.findIndex((s) => s.step === step);
        const updated =
          idx >= 0
            ? existing.map((s, i) => (i === idx ? { step, status, detail } : s))
            : [...existing, { step, status, detail }];
        return { ...prev, [nodeId]: updated };
      });

      // Update node provision status if final
      if (provisionStatus) {
        setNodes((prev) =>
          prev.map((n) =>
            n.id === nodeId
              ? {
                  ...n,
                  provisionStatus,
                  provisionLog: report ? JSON.stringify(report) : n.provisionLog,
                }
              : n
          )
        );
        // Clear live steps after a delay when done
        if (["audited", "provisioned", "unreachable", "failed"].includes(provisionStatus)) {
          setTimeout(() => {
            setLiveSteps((prev) => {
              const copy = { ...prev };
              delete copy[nodeId];
              return copy;
            });
            loadNodes();
          }, 3000);
        }
      }
    }
    if (event.type === "node:status") {
      const { nodeId, status, agentVersion } = event.payload as {
        nodeId: string; status: string; agentVersion?: string;
      };
      setNodes((prev) =>
        prev.map((n) =>
          n.id === nodeId
            ? { ...n, status, ...(agentVersion ? { agentVersion } : {}) }
            : n
        )
      );
    }
    if (event.type === "node:updated") {
      const { nodeId, name } = event.payload as { nodeId: string; name: string };
      setNodes((prev) => prev.map((n) => (n.id === nodeId ? { ...n, name } : n)));
    }
    if (event.type === "node:created") {
      const created = event.payload as unknown as Node;
      setNodes((prev) =>
        prev.some((n) => n.id === created.id) ? prev : [created, ...prev]
      );
    }
  }, []);

  const { connected } = useSSE(handleSSE, loadNodes);

  const provision = async (nodeId: string) => {
    setNodes((prev) =>
      prev.map((n) => (n.id === nodeId ? { ...n, provisionStatus: "provisioning" } : n))
    );
    await apiFetch(`/api/nodes/${nodeId}/provision`, { method: "POST" });
  };

  type OffboardResult = {
    deleted: boolean;
    offboarded?: boolean;
    forced?: boolean;
    timedOut?: boolean;
    reason?: string;
  };

  const deleteNode = async (node: Node) => {
    if (
      !confirm(
        "Delete this node?\n\n" +
          "• All active deployments and fine-tune jobs will be stopped\n" +
          "• The agent will uninstall itself (service, /opt/dgx-agent, systemd unit, sudoers entry)\n" +
          "• Installed software (Docker, Ollama, CUDA) stays on the machine\n" +
          "• The node record is removed from the database"
      )
    )
      return;
    setOffboarding(node);
    setOffboardPhase("working");
    setOffboardError(null);
    try {
      const result = await apiFetch<OffboardResult>(`/api/nodes/${node.id}`, {
        method: "DELETE",
      });
      if (result.deleted) {
        setNodes((prev) => prev.filter((n) => n.id !== node.id));
        setOffboarding(null);
        loadNodes();
      } else if (result.timedOut) {
        setOffboardPhase("timed-out");
      } else {
        setOffboardError("Offboarding did not complete. Try 'Remove anyhow'.");
        setOffboardPhase("timed-out");
      }
    } catch (err) {
      setOffboardError(err instanceof Error ? err.message : String(err));
      setOffboardPhase("timed-out");
    }
  };

  const removeAnyhow = async () => {
    if (!offboarding) return;
    const node = offboarding;
    setOffboardPhase("working");
    setOffboardError(null);
    try {
      await apiFetch<OffboardResult>(`/api/nodes/${node.id}?force=true`, {
        method: "DELETE",
      });
      setNodes((prev) => prev.filter((n) => n.id !== node.id));
      setOffboarding(null);
      loadNodes();
    } catch (err) {
      setOffboardError(err instanceof Error ? err.message : String(err));
      setOffboardPhase("timed-out");
    }
  };

  const getReport = (node: Node): ProvisionReport | null => {
    if (!node.provisionLog) return null;
    try {
      return JSON.parse(node.provisionLog);
    } catch {
      return null;
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold">Node Management</h1>
          <span
            className={`text-xs px-2 py-1 rounded ${
              connected ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"
            }`}
          >
            {connected ? "Live" : "Disconnected"}
          </span>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleReseed}
            disabled={reseeding}
            className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed text-gray-300 px-4 py-2 rounded text-sm font-medium transition-colors"
          >
            {reseeding ? "Reseeding…" : "Reseed SSH trust"}
          </button>
          <button
            onClick={() => setShowOnboarding(true)}
            className="bg-green-600 hover:bg-green-500 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
          >
            Add Node
          </button>
        </div>
      </div>

      {showOnboarding && (
        <OnboardingDialog
          serverHost={serverHost}
          onClose={() => setShowOnboarding(false)}
        />
      )}

      {offboarding && (
        <OffboardingDialog
          node={offboarding}
          phase={offboardPhase}
          error={offboardError}
          onRemoveAnyhow={removeAnyhow}
          onClose={() => setOffboarding(null)}
        />
      )}

      {/* Node List */}
      <div className="space-y-4">
        {nodes.map((node) => {
          const report = getReport(node);
          const steps = liveSteps[node.id];
          const isProvisioning = ["provisioning"].includes(node.provisionStatus);
          const isAuditing = !!steps && !report && node.provisionStatus === "pending";

          return (
            <div
              key={node.id}
              className="bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div>
                  <div className="flex items-center gap-2">
                    {renamingId === node.id ? (
                      <form
                        onSubmit={(e) => { e.preventDefault(); saveRename(node.id); }}
                        className="flex items-center gap-2"
                      >
                        <input
                          autoFocus
                          value={renameDraft}
                          onChange={(e) => setRenameDraft(e.target.value)}
                          onKeyDown={(e) => { if (e.key === "Escape") cancelRename(); }}
                          disabled={savingRename}
                          className="bg-gray-800 border border-gray-700 rounded px-2 py-0.5 text-lg font-semibold focus:outline-none focus:border-green-500 w-48"
                        />
                        <button
                          type="submit"
                          disabled={savingRename || !renameDraft.trim()}
                          className="text-[10px] px-2 py-0.5 rounded bg-green-600 hover:bg-green-500 disabled:bg-gray-600 text-white font-medium"
                        >
                          {savingRename ? "Saving..." : "Save"}
                        </button>
                        <button
                          type="button"
                          onClick={cancelRename}
                          disabled={savingRename}
                          className="text-[10px] px-2 py-0.5 rounded bg-gray-700 hover:bg-gray-600 text-gray-300"
                        >
                          Cancel
                        </button>
                        {renameError && (
                          <span className="text-[10px] text-red-400">{renameError}</span>
                        )}
                      </form>
                    ) : (
                      <>
                        <h3 className="font-semibold text-lg">{node.name}</h3>
                        <button
                          type="button"
                          onClick={() => startRename(node)}
                          title="Rename node"
                          className="text-gray-500 hover:text-gray-200 text-xs px-1 leading-none"
                          aria-label="Rename node"
                        >
                          &#9998;
                        </button>
                      </>
                    )}
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded font-medium ${
                        provisionBadge[node.provisionStatus] || "bg-gray-700 text-gray-300"
                      }`}
                    >
                      {node.provisionStatus}
                    </span>
                    {node.status === "online" && (
                      <span className="flex items-center gap-1 text-[10px] text-green-400">
                        <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                        online
                      </span>
                    )}
                    {node.agentVersion && (
                      <span
                        className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
                          isOutdated(node)
                            ? "bg-orange-900/60 text-orange-300 border border-orange-800"
                            : "text-gray-500"
                        }`}
                      >
                        v{node.agentVersion}
                        {isOutdated(node) && (
                          <span className="text-orange-400"> → v{expectedVersion}</span>
                        )}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 mt-0.5 flex items-center gap-2">
                    <span>{node.ipAddress}</span>
                    {node.gpuModel && (
                      <span className="text-gray-400">{node.gpuModel}</span>
                    )}
                    {node.arch && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400 font-mono">
                        {node.arch}
                      </span>
                    )}
                  </p>
                </div>
                <div className="flex gap-2">
                  {isOutdated(node) && node.status === "online" && (
                    <button
                      onClick={() => upgradeAgent(node.id)}
                      disabled={upgrading[node.id]}
                      title="Upgrade agent to latest version (active deployments are preserved)"
                      className="bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white px-3 py-1 rounded text-xs font-medium transition-colors"
                    >
                      {upgrading[node.id] ? "Upgrading..." : "Upgrade Agent"}
                    </button>
                  )}
                  {!isOutdated(node) && node.status === "online" && (
                    <button
                      onClick={() => upgradeAgent(node.id)}
                      disabled={upgrading[node.id]}
                      title="Redeploy agent (update config, env vars)"
                      className="bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 text-gray-300 px-3 py-1 rounded text-xs font-medium transition-colors"
                    >
                      {upgrading[node.id] ? "Redeploying..." : "Redeploy"}
                    </button>
                  )}
                  {report &&
                    report.checks.some((c) => c.status === "yellow") &&
                    !isProvisioning && (
                      <button
                        onClick={() => provision(node.id)}
                        className="bg-yellow-600 hover:bg-yellow-700 text-white px-3 py-1 rounded text-xs font-medium transition-colors"
                      >
                        Provision
                      </button>
                    )}
                  <button
                    onClick={() => deleteNode(node)}
                    className="bg-red-800/60 hover:bg-red-700 text-red-300 px-3 py-1 rounded text-xs font-medium transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>

              {/* Live provisioning steps */}
              {steps && steps.length > 0 && (
                <div className="mb-3 bg-black/30 border border-gray-800 rounded p-3">
                  <p className="text-[10px] text-gray-500 uppercase tracking-wide mb-2">
                    {isProvisioning ? "Provisioning" : "Auditing"} in progress...
                  </p>
                  <div className="space-y-1">
                    {steps.map((s, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs">
                        <StepIndicator status={s.status} />
                        <span className="text-gray-300 font-medium">{s.step}</span>
                        {s.detail && (
                          <span className="text-gray-500 truncate">{s.detail}</span>
                        )}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Prerequisite Checklist */}
              {report && report.checks.length > 0 && !steps && (
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {report.checks.map((check) => (
                    <div
                      key={check.name}
                      className={`text-xs px-2.5 py-1.5 rounded border ${
                        checkBadge[check.status] || "bg-gray-800 text-gray-400 border-gray-700"
                      }`}
                    >
                      <span className="font-medium">{check.name}</span>
                      <span className="block text-[10px] opacity-80 truncate mt-0.5">
                        {check.detail}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {/* Not yet audited */}
              {!report && !steps && node.provisionStatus === "pending" && (
                <p className="text-xs text-gray-500 italic">Audit starting...</p>
              )}
            </div>
          );
        })}
      </div>

      {nodes.length === 0 && (
        <div className="text-center py-12 text-gray-500">
          <p>No nodes yet. Add a DGX Spark above to get started.</p>
        </div>
      )}
    </div>
  );
}

function OffboardingDialog({
  node,
  phase,
  error,
  onRemoveAnyhow,
  onClose,
}: {
  node: Node;
  phase: "working" | "timed-out";
  error: string | null;
  onRemoveAnyhow: () => void;
  onClose: () => void;
}) {
  const working = phase === "working";
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6 max-w-md w-full shadow-xl">
        <div className="flex items-center gap-3 mb-3">
          {working && (
            <span className="w-5 h-5 rounded-full border-2 border-gray-600 border-t-green-400 animate-spin" />
          )}
          <h2 className="text-lg font-semibold">
            {working ? "Offboarding" : "Offboarding stalled"} {node.name}
            {working ? "…" : ""}
          </h2>
        </div>

        {working ? (
          <p className="text-sm text-gray-400">
            Stopping deployments and waiting for the agent to uninstall itself
            (up to 30s). This can take a moment on a busy or unreachable node.
          </p>
        ) : (
          <div className="space-y-3">
            <p className="text-sm text-gray-300">
              {error
                ? error
                : "Offboarding didn't respond after 30s. The node may be powered off or factory-reset."}
            </p>
            <p className="text-sm text-gray-400">
              You can remove the node record anyhow. Any agent still running on the
              machine won&apos;t be uninstalled remotely, but the record here will be
              deleted.
            </p>
            <div className="flex justify-end gap-2 pt-1">
              <button
                onClick={onClose}
                className="bg-gray-700 hover:bg-gray-600 text-gray-300 px-4 py-2 rounded text-sm font-medium transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={onRemoveAnyhow}
                className="bg-red-700 hover:bg-red-600 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
              >
                Remove anyhow
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function OnboardingDialog({
  serverHost,
  onClose,
}: {
  serverHost: string;
  onClose: () => void;
}) {
  return (
    <div
      className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <div
        className="bg-gray-900 border border-gray-800 rounded-lg p-6 max-w-2xl w-full shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Onboard a new DGX node</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-300 text-lg leading-none"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        <p className="text-sm text-gray-400 mb-2">
          Before generating a token, make sure the target node has:
        </p>
        <ul className="text-sm text-gray-300 space-y-2 mb-5 pl-5 list-disc">
          <li>
            <span className="font-medium">NVIDIA GPU drivers installed</span> —{" "}
            <code className="text-green-400">nvidia-smi</code> must work on the node.
          </li>
          <li>
            <span className="font-medium">Network access to this manager</span> at{" "}
            <code className="text-green-400">{serverHost}</code>.
          </li>
          <li>
            <span className="font-medium">Passwordless SSH from this manager to the node</span> —
            copy the manager&apos;s public key to the agent user on the target, e.g.{" "}
            <code className="text-green-400">ssh-copy-id user@node-ip</code>. Verify with{" "}
            <code className="text-green-400">ssh user@node-ip true</code> (no prompt).
          </li>
          <li>
            <span className="font-medium">Passwordless sudo on the node</span> — add a
            visudo entry on the target, e.g.{" "}
            <code className="text-green-400">user ALL=(ALL) NOPASSWD: ALL</code> in{" "}
            <code className="text-green-400">/etc/sudoers.d/user</code>.
          </li>
        </ul>

        <OnboardingCommand serverHost={serverHost} />

        <p className="text-xs text-gray-500 mt-4">
          Run the command on the target node. It will install the agent and register
          with this manager. The node will appear in the list once connected.
        </p>
      </div>
    </div>
  );
}

function StepIndicator({ status }: { status: string }) {
  if (status === "running" || status === "checking" || status === "installing") {
    return (
      <span className="w-4 h-4 flex items-center justify-center">
        <span className="w-2 h-2 rounded-full bg-blue-400 animate-pulse" />
      </span>
    );
  }
  if (status === "done" || status === "green" || status === "installed") {
    return (
      <span className="w-4 h-4 flex items-center justify-center text-green-400 text-[10px]">
        &#10003;
      </span>
    );
  }
  if (status === "failed" || status === "red") {
    return (
      <span className="w-4 h-4 flex items-center justify-center text-red-400 text-[10px]">
        &#10007;
      </span>
    );
  }
  if (status === "yellow") {
    return (
      <span className="w-4 h-4 flex items-center justify-center text-yellow-400 text-[10px]">
        &#9888;
      </span>
    );
  }
  return <span className="w-4 h-4 flex items-center justify-center text-gray-500 text-[10px]">&#8226;</span>;
}
