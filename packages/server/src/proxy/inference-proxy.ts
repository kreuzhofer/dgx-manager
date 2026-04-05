import { Router, Request, Response } from "express";
import { prisma } from "../prisma.js";

export const inferenceProxy = Router();

// Round-robin state per rule
const rrCounters = new Map<string, number>();

inferenceProxy.all("/:ruleName/v1/*", async (req: Request, res: Response) => {
  const rule = await prisma.loadBalancerRule.findUnique({
    where: { name: req.params.ruleName },
    include: {
      endpoints: {
        include: {
          deployment: { include: { node: true } },
        },
      },
    },
  });

  if (!rule) return res.status(404).json({ error: "Rule not found" });

  const activeEndpoints = rule.endpoints.filter(
    (ep) => ep.deployment.status === "running" && ep.deployment.port
  );

  if (activeEndpoints.length === 0) {
    return res.status(503).json({ error: "No active endpoints" });
  }

  // Select target based on strategy
  let target;
  if (rule.strategy === "round-robin") {
    const counter = (rrCounters.get(rule.id) ?? 0) % activeEndpoints.length;
    target = activeEndpoints[counter];
    rrCounters.set(rule.id, counter + 1);
  } else {
    // Default: first available
    target = activeEndpoints[0];
  }

  const targetUrl = `http://${target.deployment.node.ipAddress}:${target.deployment.port}/v1/${(req.params as Record<string, string>)[0]}`;

  try {
    const proxyRes = await fetch(targetUrl, {
      method: req.method,
      headers: { "Content-Type": "application/json" },
      body: req.method !== "GET" ? JSON.stringify(req.body) : undefined,
    });

    res.status(proxyRes.status);
    for (const [key, value] of proxyRes.headers.entries()) {
      res.setHeader(key, value);
    }

    if (proxyRes.body) {
      const reader = proxyRes.body.getReader();
      const pump = async (): Promise<void> => {
        const { done, value } = await reader.read();
        if (done) { res.end(); return; }
        res.write(value);
        return pump();
      };
      await pump();
    } else {
      const text = await proxyRes.text();
      res.send(text);
    }
  } catch (err) {
    res.status(502).json({ error: "Proxy error", detail: String(err) });
  }
});
