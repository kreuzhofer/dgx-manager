import { Router } from "express";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { prisma } from "../prisma.js";
import { auditNode, provisionNode } from "../ssh/provisioner.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { metricsBuffer } from "../metrics-buffer.js";
import type { AgentHub } from "../ws/agent-hub.js";
import { powerCommand, macCaptureCmd, normalizeMac, type PowerAction } from "../nodes/power.js";
import { sshExec as defaultSshExec } from "../ssh/executor.js";
import { broadcastFor, sendMagicPacket as defaultWolSend } from "../nodes/wol.js";
import { triggerClusterReseed } from "../ssh/known-hosts-trigger.js";

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

/**
 * @openapi
 * /api/nodes:
 *   get:
 *     tags: [Nodes]
 *     summary: List all registered DGX nodes
 *     description: >
 *       Returns every registered DGX Spark node ordered by name, each enriched with
 *       a `metrics` array containing the latest GPU sample from the in-memory metrics
 *       buffer (empty array if no sample has arrived since server start). Live metric
 *       updates flow via SSE (`node:metrics`) after the initial paint — this endpoint
 *       is the source of truth for the current snapshot, not the persistent
 *       MetricSnapshot table (which is retained for history).
 *     responses:
 *       '200':
 *         description: Array of node objects with latest metrics snapshot
 */
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

/**
 * @openapi
 * /api/nodes/idle:
 *   get:
 *     tags: [Nodes]
 *     summary: List online nodes available for new deployments
 *     description: >
 *       Returns all nodes whose status is `online`, ordered by name. These are
 *       candidates for deployment targets. VRAM admission runs at deploy time
 *       (POST /api/deployments), not here — this endpoint does not filter by
 *       current VRAM usage. Use nodeId="auto" on the deploy endpoint to let the
 *       server pick the first available idle node.
 *     responses:
 *       '200':
 *         description: Array of online Node objects
 */
nodesRouter.get("/idle", async (_req, res) => {
  const nodes = await prisma.node.findMany({
    where: {
      status: "online",
    },
    orderBy: { name: "asc" },
  });
  res.json(nodes);
});

/**
 * @openapi
 * /api/nodes/agent-version:
 *   get:
 *     tags: [Nodes]
 *     summary: Return the server-expected agent version
 *     description: >
 *       Returns the minimum acceptable agent version baked into the server image at
 *       build time. The dashboard compares each connected agent's reported version
 *       against this and offers an upgrade button when the agent is outdated. Agents
 *       reporting a version >= this value are considered up-to-date.
 *     responses:
 *       '200':
 *         description: Object with `version` string (semver)
 */
nodesRouter.get("/agent-version", (_req, res) => {
  res.json({ version: getExpectedAgentVersion() });
});

/**
 * @openapi
 * /api/nodes/{id}/metrics/history:
 *   get:
 *     tags: [Nodes]
 *     summary: Return the in-memory GPU metrics ring buffer for a node
 *     description: >
 *       Returns the full ring buffer of recent GPU metric samples for the given node
 *       from the in-memory metrics buffer (not the persisted MetricSnapshot table).
 *       Samples are kept for roughly the last N minutes depending on buffer size.
 *       Used by the node detail chart for sparkline / time-series rendering.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *         description: Node ID
 *     responses:
 *       '200':
 *         description: Array of metric samples (gpuUtil, vramUsed, temperature, tps, etc.)
 */
nodesRouter.get("/:id/metrics/history", (req, res) => {
  res.json(metricsBuffer.getHistory(req.params.id));
});

/**
 * @openapi
 * /api/nodes/{id}:
 *   get:
 *     tags: [Nodes]
 *     summary: Get a single node with its recent metrics and deployments
 *     description: >
 *       Returns a node record including the last 60 MetricSnapshot rows (for historical
 *       charts), all associated deployments (with model info), plus GPU/Docker audit
 *       fields. Use this for the node detail page where more history is needed than
 *       the live metrics buffer provides.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *         description: Node ID
 *     responses:
 *       '200':
 *         description: Node object with metrics and deployments included
 *       '404':
 *         description: Node not found
 */
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

/**
 * @openapi
 * /api/nodes:
 *   post:
 *     tags: [Nodes]
 *     summary: Register a new DGX node
 *     description: >
 *       Creates a new Node record and, if an IP address is provided, immediately runs
 *       a background SSH audit (checking NVIDIA drivers, Docker, Ollama, sparkrun).
 *       The audit result is written back to the node row and broadcast over SSE as
 *       `node:provision`. Nodes without an IP are token-bootstrapped (the agent
 *       self-registers using a join token) and do not need SSH access.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [name]
 *             properties:
 *               name: { type: string, description: "Human-readable node label (unique)." }
 *               ipAddress: { type: string, description: "SSH-reachable management IP. Omit for token-bootstrapped nodes." }
 *     responses:
 *       '201':
 *         description: Created node record; audit runs asynchronously in background
 *       '400':
 *         description: name is required
 */
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

/**
 * @openapi
 * /api/nodes/{id}:
 *   patch:
 *     tags: [Nodes]
 *     summary: Rename a node
 *     description: >
 *       Updates the human-readable display name of a node. The name is trimmed and
 *       must be non-empty. Returns 409 if another node already uses the requested name,
 *       404 if the node doesn't exist. Broadcasts `node:updated` over SSE on success.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [name]
 *             properties:
 *               name: { type: string, description: "New display name for the node." }
 *     responses:
 *       '200':
 *         description: Updated node object
 *       '400':
 *         description: name is required
 *       '404':
 *         description: Node not found
 *       '409':
 *         description: Name already in use by another node
 */
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

/**
 * @openapi
 * /api/nodes/{id}/provision:
 *   post:
 *     tags: [Nodes]
 *     summary: Provision a node (install Docker, sparkrun, nvidia-container-toolkit)
 *     description: >
 *       Runs the full provisioning sequence over SSH on the node: installs Docker,
 *       nvidia-container-toolkit, sparkrun, and any other prerequisites reported as
 *       missing by the prior audit. Provisioning is asynchronous — this endpoint
 *       returns immediately with `{ status: "provisioning" }` and broadcasts progress
 *       events over SSE (`node:provision`). The node must have been audited first
 *       (provisionLog must contain checks). Token-bootstrapped nodes (no ipAddress)
 *       cannot be provisioned via this endpoint.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Provisioning started; progress flows via SSE
 *       '400':
 *         description: Node not audited yet or no IP address
 *       '404':
 *         description: Node not found
 */
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
    // A newly provisioned node must trust (and be trusted by) the rest of the
    // cluster, else node→node SSH for multi-node deploys fails with rc=255.
    // force:true — onboarding is deliberate and must not be suppressed by the throttle.
    triggerClusterReseed({ force: true }).catch((e) => console.error(`known-hosts reseed after provision failed: ${(e as Error).message}`));
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

/**
 * @openapi
 * /api/nodes/{id}/update-agent:
 *   post:
 *     tags: [Nodes]
 *     summary: Push latest agent bundle to an online node
 *     description: >
 *       Sends a `cmd:update` WebSocket message to the connected agent on the given
 *       node. The agent downloads the arch-appropriate tarball from
 *       GET /api/agent/bundle, swaps /opt/dgx-agent, and restarts itself via
 *       systemd — no SSH required. The agent must be online for this to work; if
 *       it is offline, use the install script (GET /api/agent/install.sh) instead.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Update command dispatched; returns { status, version, bundleUrl }
 *       '400':
 *         description: Agent is offline
 *       '404':
 *         description: Node not found
 */
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

// POST /api/nodes/:id/power — reboot / shutdown / sleep a node over SSH.
// powerState transitions: reboot -> "rebooting", shutdown -> "off", sleep -> "asleep".
// MAC is captured (best-effort) before a shutdown/sleep so the node can be woken later.
nodesRouter.post("/:id/power", async (req, res) => {
  const action = req.body?.action as PowerAction;
  if (action !== "reboot" && action !== "shutdown" && action !== "sleep") {
    return res.status(400).json({ error: `Invalid action: ${action}` });
  }
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });
  if (!node.ipAddress) return res.status(400).json({ error: "Node has no ipAddress" });

  // Injected so tests can stub it; defaults to the real ssh2-backed exec.
  const sshExec = (req.app.get("sshExec") || defaultSshExec) as typeof defaultSshExec;

  // Best-effort MAC capture while the node is still reachable (skip for reboot —
  // it comes right back; capture for shutdown/sleep where we may need WOL).
  let macAddress = node.macAddress;
  if (action !== "reboot") {
    try {
      const r = await sshExec(node.ipAddress, macCaptureCmd(node.ipAddress), { timeout: 10_000 });
      const mac = normalizeMac(r.stdout);
      if (mac) macAddress = mac;
    } catch {
      // non-fatal: WOL just won't be available if we never captured a MAC
    }
  }

  try {
    await sshExec(node.ipAddress, powerCommand(action), { timeout: 15_000 });
  } catch (err) {
    return res.status(502).json({ error: `Power command failed: ${String(err)}` });
  }

  const powerState = action === "reboot" ? "rebooting" : action === "sleep" ? "asleep" : "off";
  await prisma.node.update({
    where: { id: node.id },
    data: { powerState, ...(macAddress ? { macAddress } : {}) },
  });
  sseBroadcast({ type: "node:status", payload: { nodeId: node.id, powerState } });

  res.json({ status: "ok", powerState });
});

// POST /api/nodes/:id/wake — send a Wake-on-LAN magic packet to a powered-off node.
nodesRouter.post("/:id/wake", async (req, res) => {
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });
  if (!node.macAddress) {
    return res.status(409).json({
      error: "No MAC captured for this node yet — it must have been audited or shut down via the manager at least once.",
    });
  }
  if (!node.ipAddress) return res.status(400).json({ error: "Node has no ipAddress" });

  const wolSend = (req.app.get("wolSend") || defaultWolSend) as typeof defaultWolSend;
  try {
    await wolSend(node.macAddress, broadcastFor(node.ipAddress));
  } catch (err) {
    return res.status(502).json({ error: `WOL send failed: ${String(err)}` });
  }

  await prisma.node.update({ where: { id: node.id }, data: { powerState: "waking" } });
  sseBroadcast({ type: "node:status", payload: { nodeId: node.id, powerState: "waking" } });
  res.json({ status: "ok", powerState: "waking" });
});

/**
 * @openapi
 * /api/nodes/{id}:
 *   delete:
 *     tags: [Nodes]
 *     summary: Remove a node and clean up all associated resources
 *     description: >
 *       Stops all active deployments on the node (via `cmd:undeploy`), cancels any
 *       running fine-tune jobs, tells the agent to uninstall itself (`cmd:deprovision`
 *       if online), then deletes the node row and all related DB records (metrics,
 *       deployments, fine-tune jobs, cluster memberships, load-balancer endpoints).
 *       Installed software on the machine (Docker, Ollama, etc.) is NOT removed.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: Node deleted; returns { deleted, stoppedDeployments }
 *       '404':
 *         description: Node not found
 */
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
