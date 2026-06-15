import { Router } from "express";
import { sshExec as defaultSshExec } from "../ssh/executor.js";
import { broadcast } from "../sse.js";
import { reseedClusterKnownHosts } from "../ssh/known-hosts.js";
import { loadOnlineSeedNodes } from "../ssh/known-hosts-trigger.js";

export const clusterRouter = Router();

// Manually reseed the cross-node known_hosts mesh. Bypasses the throttle
// (operator intent is explicit). 200 if any node seeded, 502 if none.
clusterRouter.post("/reseed-known-hosts", async (req, res) => {
  const sshExec = (req.app.get("sshExec") || defaultSshExec) as typeof defaultSshExec;
  try {
    const nodes = await loadOnlineSeedNodes();
    const report = await reseedClusterKnownHosts(nodes, { sshExec, broadcast, logger: console }, { force: true });
    const anyOk = report.perNode.some((p) => p.ok);
    res.status(anyOk ? 200 : 502).json(report);
  } catch (e) {
    res.status(500).json({ error: (e as Error).message });
  }
});
