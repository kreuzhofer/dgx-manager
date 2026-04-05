import { Router } from "express";
import { prisma } from "../prisma.js";

export const loadbalancerRouter = Router();

// GET /api/lb/rules
loadbalancerRouter.get("/rules", async (_req, res) => {
  const rules = await prisma.loadBalancerRule.findMany({
    orderBy: { createdAt: "desc" },
    include: {
      endpoints: { include: { deployment: { include: { node: true, model: true } } } },
    },
  });
  res.json(rules);
});

// POST /api/lb/rules
loadbalancerRouter.post("/rules", async (req, res) => {
  const { name, modelName, strategy, listenPath } = req.body;
  if (!name || !modelName) {
    return res.status(400).json({ error: "name and modelName required" });
  }
  const rule = await prisma.loadBalancerRule.create({
    data: { name, modelName, strategy, listenPath },
  });
  res.status(201).json(rule);
});

// PUT /api/lb/rules/:id
loadbalancerRouter.put("/rules/:id", async (req, res) => {
  const { name, modelName, strategy, listenPath } = req.body;
  const rule = await prisma.loadBalancerRule.update({
    where: { id: req.params.id },
    data: { name, modelName, strategy, listenPath },
  });
  res.json(rule);
});

// DELETE /api/lb/rules/:id
loadbalancerRouter.delete("/rules/:id", async (req, res) => {
  await prisma.loadBalancerEndpoint.deleteMany({ where: { ruleId: req.params.id } });
  await prisma.loadBalancerRule.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});

// POST /api/lb/rules/:id/endpoints
loadbalancerRouter.post("/rules/:id/endpoints", async (req, res) => {
  const { deploymentId, weight } = req.body;
  if (!deploymentId) {
    return res.status(400).json({ error: "deploymentId required" });
  }
  const endpoint = await prisma.loadBalancerEndpoint.create({
    data: { ruleId: req.params.id, deploymentId, weight: weight ?? 1 },
  });
  res.status(201).json(endpoint);
});

// DELETE /api/lb/endpoints/:id
loadbalancerRouter.delete("/endpoints/:id", async (req, res) => {
  await prisma.loadBalancerEndpoint.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});
