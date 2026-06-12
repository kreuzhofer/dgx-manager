import { Router } from "express";
import { prisma } from "../prisma.js";

export const loadbalancerRouter = Router();

/**
 * @openapi
 * /api/lb/rules:
 *   get:
 *     tags: [Load Balancer]
 *     summary: List all load-balancer rules
 *     description: >
 *       Returns every LoadBalancerRule ordered by creation date descending, each
 *       including its endpoints (with the linked Deployment, Node, and Model).
 *       A rule defines a named round-robin proxy: `listenPath` is the URL prefix
 *       the proxy accepts, `modelName` is the served model name, `strategy` is the
 *       balancing strategy (default: round-robin). Requests to `/lb/{listenPath}`
 *       are distributed across the rule's endpoints.
 *     responses:
 *       '200':
 *         description: Array of rule objects with endpoints included
 */
loadbalancerRouter.get("/rules", async (_req, res) => {
  const rules = await prisma.loadBalancerRule.findMany({
    orderBy: { createdAt: "desc" },
    include: {
      endpoints: { include: { deployment: { include: { node: true, model: true } } } },
    },
  });
  res.json(rules);
});

/**
 * @openapi
 * /api/lb/rules:
 *   post:
 *     tags: [Load Balancer]
 *     summary: Create a new load-balancer rule
 *     description: >
 *       Creates a LoadBalancerRule that acts as a round-robin proxy to one or more
 *       deployment endpoints. After creating the rule, add endpoints via
 *       POST /api/lb/rules/{id}/endpoints. The `listenPath` determines the URL prefix
 *       under `/lb/` where traffic is accepted; it defaults to the rule name if not set.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [name, modelName]
 *             properties:
 *               name: { type: string, description: "Human-readable rule label." }
 *               modelName: { type: string, description: "The OpenAI model name to advertise at /lb/..." }
 *               strategy: { type: string, description: "Balancing strategy (e.g. 'round-robin'). Optional." }
 *               listenPath: { type: string, description: "URL path prefix under /lb/ to match. Optional." }
 *     responses:
 *       '201':
 *         description: Created rule record
 *       '400':
 *         description: name and modelName required
 */
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

/**
 * @openapi
 * /api/lb/rules/{id}:
 *   put:
 *     tags: [Load Balancer]
 *     summary: Update a load-balancer rule
 *     description: >
 *       Replaces the mutable fields of a LoadBalancerRule: name, modelName, strategy,
 *       and listenPath. Endpoints are managed separately via the endpoints sub-resource.
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
 *             properties:
 *               name: { type: string }
 *               modelName: { type: string }
 *               strategy: { type: string }
 *               listenPath: { type: string }
 *     responses:
 *       '200':
 *         description: Updated rule record
 *       '404':
 *         description: Rule not found
 */
loadbalancerRouter.put("/rules/:id", async (req, res) => {
  const { name, modelName, strategy, listenPath } = req.body;
  const rule = await prisma.loadBalancerRule.update({
    where: { id: req.params.id },
    data: { name, modelName, strategy, listenPath },
  });
  res.json(rule);
});

/**
 * @openapi
 * /api/lb/rules/{id}:
 *   delete:
 *     tags: [Load Balancer]
 *     summary: Delete a load-balancer rule and its endpoints
 *     description: >
 *       Removes all LoadBalancerEndpoint rows for the rule, then deletes the rule itself.
 *       Running deployments referenced by the endpoints are not affected — only the
 *       routing configuration is removed.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ deleted: true }'
 */
loadbalancerRouter.delete("/rules/:id", async (req, res) => {
  await prisma.loadBalancerEndpoint.deleteMany({ where: { ruleId: req.params.id } });
  await prisma.loadBalancerRule.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});

/**
 * @openapi
 * /api/lb/rules/{id}/endpoints:
 *   post:
 *     tags: [Load Balancer]
 *     summary: Add a deployment endpoint to a load-balancer rule
 *     description: >
 *       Associates a running Deployment with a LoadBalancerRule as a backend endpoint.
 *       The inference proxy at `/lb/` will distribute requests across all endpoints for
 *       the rule using the configured strategy (default: round-robin). The optional
 *       `weight` field allows weighted distribution — higher weight means more traffic.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *         description: Rule ID
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [deploymentId]
 *             properties:
 *               deploymentId: { type: string, description: "ID of the running Deployment to add as a backend." }
 *               weight: { type: number, description: "Relative traffic weight (default: 1)." }
 *     responses:
 *       '201':
 *         description: Created endpoint record
 *       '400':
 *         description: deploymentId required
 */
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

/**
 * @openapi
 * /api/lb/endpoints/{id}:
 *   delete:
 *     tags: [Load Balancer]
 *     summary: Remove a backend endpoint from a load-balancer rule
 *     description: >
 *       Deletes a single LoadBalancerEndpoint row, removing the associated deployment
 *       from the proxy rotation. The deployment itself is not affected. Use this to
 *       drain traffic from a deployment before stopping it.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *         description: Endpoint ID
 *     responses:
 *       '200':
 *         description: '{ deleted: true }'
 */
loadbalancerRouter.delete("/endpoints/:id", async (req, res) => {
  await prisma.loadBalancerEndpoint.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});
