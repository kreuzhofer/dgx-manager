import { Router } from "express";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { prisma } from "../prisma.js";
import { auditNode, provisionNode } from "../ssh/provisioner.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { metricsBuffer } from "../metrics-buffer.js";
import type { AgentHub } from "../ws/agent-hub.js";

const __dirname = dirname(fileURLToPath(import.meta.url));

function getExpectedAgentVersion(): string {
  // Baked into the server container at build time — this is the minimum
  // acceptable agent version. Agents reporting >= this are fine.
  try {
    return JSON.parse(readFileSync(join(__dirname, "../../../agent/package.json"), "utf-8")).version;
  } catch {
    return "unknown";
  }
}

export const nodesRouter = Router();

// GET /api/nodes
// Returns each node with `metrics: [latestSample]` (or `metrics: []` if no
// sample has been seen since boot). We read from the in-memory metricsBuffer
// instead of MetricSnapshot — Prisma's `include: { metrics: take: 1 }`
// emits a single SQL with a global ORDER BY across all nodeIds, which
// forces a temp B-tree sort of every matching row (~10s with 2.4M rows).
// Live updates flow through SSE (`node:metrics`) after the initial paint,
// so the buffer is the source of truth for "now" anyway.
nodesRouter.get("/", async (_req, res) => {
  const nodes = await prisma.node.findMany({ orderBy: { name: "asc" } });
  const enriched = nodes.map((n) => {
    const history = metricsBuffer.getHistory(n.id);
    const latest = history[history.length - 1];
    return {
      ...n,
      metrics: latest
        ? [
            {
              gpuUtil: latest.gpuUtil,
              vramUsed: latest.vramUsed,
              temperature: latest.temperature,
              tps: latest.tps,
              activeRequests: latest.activeRequests,
              timestamp: new Date(latest.timestamp).toISOString(),
            },
          ]
        : [],
    };
  });
  res.json(enriched);
});

// GET /api/nodes/idle — online nodes available for deployments
// Now returns all online nodes since multi-deployment is supported.
// VRAM admission check happens at deploy time, not here.
nodesRouter.get("/idle", async (_req, res) => {
  const nodes = await prisma.node.findMany({
    where: {
      status: "online",
    },
    orderBy: { name: "asc" },
  });
  res.json(nodes);
});

// GET /api/nodes/agent-version
nodesRouter.get("/agent-version", (_req, res) => {
  res.json({ version: getExpectedAgentVersion() });
});

// GET /api/nodes/:id/metrics/history
nodesRouter.get("/:id/metrics/history", (req, res) => {
  res.json(metricsBuffer.getHistory(req.params.id));
});

// GET /api/nodes/:id
nodesRouter.get("/:id", async (req, res) => {
  const node = await prisma.node.findUnique({
    where: { id: req.params.id },
    include: {
      metrics: { orderBy: { timestamp: "desc" }, take: 60 },
      deployments: { include: { model: true } },
    },
  });
  if (!node) return res.status(404).json({ error: "Node not found" });
  res.json(node);
});

// POST /api/nodes
nodesRouter.post("/", async (req, res) => {
  const { name, ipAddress } = req.body;
  if (!name) {
    return res.status(400).json({ error: "name is required" });
  }

  const node = await prisma.node.create({
    data: { name, ipAddress: ipAddress || null },
  });

  // Run audit in background (only for SSH-managed nodes with an IP)
  if (!ipAddress) {
    return res.status(201).json(node);
  }
  auditNode(ipAddress, node.id).then(async (report) => {
    await prisma.node.update({
      where: { id: node.id },
      data: {
        provisionStatus: report.reachable ? "audited" : "unreachable",
        provisionLog: JSON.stringify(report),
        gpuModel: report.checks.find((c) => c.name === "NVIDIA Drivers")?.detail || null,
        dockerAvailable: report.checks.find((c) => c.name === "Docker")?.status === "green",
        ollamaInstalled: report.checks.find((c) => c.name === "Ollama")?.status === "green",
      },
    });
    sseBroadcast({
      type: "node:provision",
      payload: {
        nodeId: node.id,
        step: "Audit complete",
        status: report.reachable ? "done" : "failed",
        detail: report.reachable ? "All checks complete" : "Node unreachable",
        provisionStatus: report.reachable ? "audited" : "unreachable",
        report,
      },
    });
  }).catch((err) => {
    console.error(`Audit failed for ${ipAddress}:`, err);
  });

  res.status(201).json(node);
});

// PATCH /api/nodes/:id — rename a node
nodesRouter.patch("/:id", async (req, res) => {
  const { name } = req.body || {};
  if (typeof name !== "string" || name.trim().length === 0) {
    return res.status(400).json({ error: "name is required" });
  }
  const trimmed = name.trim();
  try {
    const node = await prisma.node.update({
      where: { id: req.params.id },
      data: { name: trimmed },
    });
    sseBroadcast({ type: "node:updated", payload: { nodeId: node.id, name: node.name } });
    res.json(node);
  } catch (e: unknown) {
    const code = (e as { code?: string }).code;
    if (code === "P2002") {
      return res.status(409).json({ error: `A node named "${trimmed}" already exists` });
    }
    if (code === "P2025") {
      return res.status(404).json({ error: "Node not found" });
    }
    throw e;
  }
});

// POST /api/nodes/:id/provision
nodesRouter.post("/:id/provision", async (req, res) => {
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });

  const report = node.provisionLog ? JSON.parse(node.provisionLog) : null;
  if (!report?.checks) {
    return res.status(400).json({ error: "Node has not been audited yet" });
  }

  await prisma.node.update({
    where: { id: node.id },
    data: { provisionStatus: "provisioning" },
  });

  if (!node.ipAddress) {
    return res.status(400).json({ error: "Cannot provision a node without an IP address (token-bootstrapped nodes are self-provisioned)" });
  }

  // Provision in background
  provisionNode(node.ipAddress, report.checks, node.id).then(async (log) => {
    // Re-audit after provisioning
    const newReport = await auditNode(node.ipAddress!, node.id);
    await prisma.node.update({
      where: { id: node.id },
      data: {
        provisionStatus: "provisioned",
        provisionLog: JSON.stringify(newReport),
        dockerAvailable: newReport.checks.find((c) => c.name === "Docker")?.status === "green",
        ollamaInstalled: newReport.checks.find((c) => c.name === "Ollama")?.status === "green",
      },
    });
    sseBroadcast({
      type: "node:provision",
      payload: {
        nodeId: node.id,
        step: "Provisioning complete",
        status: "done",
        provisionStatus: "provisioned",
        report: newReport,
      },
    });
  }).catch(async (err) => {
    await prisma.node.update({
      where: { id: node.id },
      data: { provisionStatus: "failed", provisionLog: String(err) },
    });
    sseBroadcast({
      type: "node:provision",
      payload: { nodeId: node.id, step: "Provisioning", status: "failed", detail: String(err), provisionStatus: "failed" },
    });
  });

  res.json({ status: "provisioning" });
});

// POST /api/nodes/:id/update-agent — push the latest agent bundle via WebSocket.
// The agent self-updates: downloads the arch-appropriate bundle over HTTP, swaps
// /opt/dgx-agent, and restarts via systemd. No SSH, no NFS.
nodesRouter.post("/:id/update-agent", async (req, res) => {
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });

  const agentHub: AgentHub = req.app.get("agentHub");
  if (!agentHub.isAgentOnline(node.id)) {
    return res.status(400).json({
      error: "Agent is offline — cannot push an update. Re-run the install script on the node.",
    });
  }

  const managerHost = process.env.MANAGER_ADVERTISE_HOST || process.env.MANAGER_HOST || "192.168.44.36";
  const port = process.env.PORT || "4000";
  const archQuery =
    node.arch === "amd64" || node.arch === "arm64" ? `?arch=${node.arch}` : "";
  const bundleUrl = `http://${managerHost}:${port}/api/agent/bundle${archQuery}`;
  const version = getExpectedAgentVersion();

  agentHub.sendToAgent(node.id, {
    type: "cmd:update",
    payload: { bundleUrl, version },
  });

  res.json({ status: "updating", version, bundleUrl });
});

// DELETE /api/nodes/:id
nodesRouter.delete("/:id", async (req, res) => {
  const node = await prisma.node.findUnique({
    where: { id: req.params.id },
    include: {
      deployments: true,
      finetuneJobs: { where: { status: { in: ["pending", "running"] } } },
    },
  });
  if (!node) return res.status(404).json({ error: "Node not found" });

  const agentHub: AgentHub = req.app.get("agentHub");

  // Stop all active deployments
  for (const deployment of node.deployments) {
    if (["pending", "running", "starting", "restarting"].includes(deployment.status)) {
      agentHub.sendToAgent(node.id, {
        type: "cmd:undeploy",
        payload: { deploymentId: deployment.id },
      });
    }
  }

  // Cancel running fine-tune jobs
  for (const job of node.finetuneJobs) {
    agentHub.sendToAgent(node.id, {
      type: "cmd:finetune:cancel",
      payload: { jobId: job.id },
    });
  }

  // Tell the agent to uninstall itself if it's online. The agent spawns a
  // detached cleanup script (systemctl stop/disable, rm /opt/dgx-agent, drop
  // the systemd unit and sudoers entry) and closes the WebSocket. Installed
  // software (Docker, Ollama, etc.) stays on the machine.
  if (agentHub.isAgentOnline(node.id)) {
    agentHub.sendToAgent(node.id, { type: "cmd:deprovision", payload: {} });
    // Small grace window so the agent can ack + launch its cleanup before we
    // drop its DB record (prevents a spurious reconnect attempt with a now-
    // deleted nodeId).
    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  // Clean up DB: delete related records then the node
  await prisma.metricSnapshot.deleteMany({ where: { nodeId: node.id } });
  await prisma.clusterNode.deleteMany({ where: { nodeId: node.id } });
  await prisma.loadBalancerEndpoint.deleteMany({
    where: { deployment: { nodeId: node.id } },
  });
  await prisma.clusterNode.deleteMany({ where: { deployment: { nodeId: node.id } } });
  await prisma.deployment.deleteMany({ where: { nodeId: node.id } });
  await prisma.fineTuneJob.deleteMany({ where: { nodeId: node.id } });
  await prisma.node.delete({ where: { id: node.id } });
  metricsBuffer.remove(node.id);

  res.json({ deleted: true, stoppedDeployments: node.deployments.length });
});
