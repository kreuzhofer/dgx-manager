import { Router } from "express";
import { prisma } from "../prisma.js";
import { auditNode, provisionNode } from "../ssh/provisioner.js";
import { deployAgent } from "../ssh/agent-deployer.js";

export const nodesRouter = Router();

// GET /api/nodes
nodesRouter.get("/", async (_req, res) => {
  const nodes = await prisma.node.findMany({
    orderBy: { createdAt: "desc" },
    include: {
      metrics: { orderBy: { timestamp: "desc" }, take: 1 },
    },
  });
  res.json(nodes);
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
  if (!name || !ipAddress) {
    return res.status(400).json({ error: "name and ipAddress required" });
  }

  const node = await prisma.node.create({
    data: { name, ipAddress },
  });

  // Run audit in background
  auditNode(ipAddress).then(async (report) => {
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
  }).catch((err) => {
    console.error(`Audit failed for ${ipAddress}:`, err);
  });

  res.status(201).json(node);
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

  // Provision in background
  provisionNode(node.ipAddress, report.checks).then(async (log) => {
    // Re-audit after provisioning
    const newReport = await auditNode(node.ipAddress);
    await prisma.node.update({
      where: { id: node.id },
      data: {
        provisionStatus: "provisioned",
        provisionLog: JSON.stringify(newReport),
        dockerAvailable: newReport.checks.find((c) => c.name === "Docker")?.status === "green",
        ollamaInstalled: newReport.checks.find((c) => c.name === "Ollama")?.status === "green",
      },
    });
  }).catch(async (err) => {
    await prisma.node.update({
      where: { id: node.id },
      data: { provisionStatus: "failed", provisionLog: String(err) },
    });
  });

  res.json({ status: "provisioning" });
});

// POST /api/nodes/:id/deploy-agent
nodesRouter.post("/:id/deploy-agent", async (req, res) => {
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });

  const managerHost = process.env.MANAGER_HOST || "0.0.0.0";
  const managerPort = Number(process.env.PORT || 4000);

  const log = await deployAgent(node.ipAddress, node.id, managerHost, managerPort);
  await prisma.node.update({
    where: { id: node.id },
    data: { provisionStatus: "agent-deployed" },
  });

  res.json({ status: "agent-deployed", log });
});

// DELETE /api/nodes/:id
nodesRouter.delete("/:id", async (req, res) => {
  await prisma.node.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});
