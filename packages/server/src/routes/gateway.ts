import { Router } from "express";
import { prisma } from "../prisma.js";
import { buildPoolView } from "../gateway/pool-view.js";
import { outstandingFor } from "../gateway/inflight.js";

/**
 * Management view of the inference gateway.
 *
 * Deliberately on the management API rather than under `/v1`: that surface is
 * an allowlist of OpenAI operations and must not grow endpoints describing the
 * cluster's internals. See docs/adr/0001-inference-gateway.md.
 */
export const gatewayViewRouter = Router();

/**
 * @openapi
 * /api/gateway:
 *   get:
 *     tags: [Gateway]
 *     summary: What the cluster publishes through the inference gateway
 *     description: >
 *       Every deployment carrying a published name, grouped into pools by that
 *       name. A pool is the set of deployments serving one name; a request for
 *       it may be answered by any serving member. Each member reports its node,
 *       runtime, bound port and the number of requests the gateway currently has
 *       in flight to it. A member that cannot take a request is returned with
 *       `serving: false` and the reason — `not-running`, `no-port`,
 *       `no-node-address` or `agent-offline` — so eligibility is visible before
 *       a request fails rather than only after.
 *     responses:
 *       '200':
 *         description: Pools, their members, and why any member is not serving
 */
/**
 * The URL a client on another machine should point at.
 *
 * Computed server-side, the same way the agent install script derives its own
 * URL: a build-time value baked into the dashboard says `localhost` outside
 * compose, and this is the one string an operator copies and pastes elsewhere.
 */
function gatewayBaseUrl(reqHostname: string): string {
  const host = process.env.MANAGER_ADVERTISE_HOST || reqHostname;
  const port = process.env.PORT || "4000";
  return `http://${host}:${port}/v1`;
}

gatewayViewRouter.get("/", async (req, res) => {
  const deployments = await prisma.deployment.findMany({
    where: { publishedName: { not: null } },
    orderBy: { id: "asc" },
    select: {
      id: true,
      publishedName: true,
      status: true,
      port: true,
      model: { select: { runtime: true } },
      node: { select: { id: true, name: true, ipAddress: true } },
    },
  });

  const agentHub = req.app.get("agentHub") as { isAgentOnline(nodeId: string): boolean } | undefined;

  const pools = buildPoolView(
    deployments.flatMap((d) => (d.publishedName === null ? [] : [{
      id: d.id,
      publishedName: d.publishedName,
      status: d.status,
      port: d.port,
      nodeId: d.node.id,
      nodeName: d.node.name,
      nodeIp: d.node.ipAddress,
      runtime: d.model.runtime,
    }])),
    (nodeId) => agentHub?.isAgentOnline(nodeId) ?? false,
    outstandingFor,
  );

  res.json({ baseUrl: gatewayBaseUrl(req.hostname), pools });
});
