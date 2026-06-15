import { Router } from "express";
import { sshExec as defaultSshExec } from "../ssh/executor.js";
import { broadcast } from "../sse.js";
import { reseedClusterKnownHosts } from "../ssh/known-hosts.js";
import { loadOnlineSeedNodes } from "../ssh/known-hosts-trigger.js";

export const clusterRouter = Router();

/**
 * @openapi
 * /api/cluster/reseed-known-hosts:
 *   post:
 *     tags: [Cluster]
 *     summary: Reseed the cross-node SSH known_hosts trust mesh
 *     description: >
 *       SSHes every online node to discover its live IPs (`ip -o -4 addr`), unions
 *       and filters them to the cluster subnets, then re-scans them into each node's
 *       system-wide `/etc/ssh/ssh_known_hosts`. This is what lets the head node SSH
 *       into workers during multi-node (tensor-parallel) deploys without hitting
 *       `rc=255 Host key verification failed`. Bypasses the automatic-trigger throttle
 *       (operator intent is explicit). Also fired automatically on node provision and
 *       on agent (re)connect.
 *     responses:
 *       '200':
 *         description: >
 *           At least one node was seeded. Body is the per-node report:
 *           `{ trustedIps: string[], perNode: [{ nodeId, host, ipsSeeded, ok, error? }] }`.
 *       '502':
 *         description: No node could be seeded (all unreachable). Same report shape.
 *       '500':
 *         description: Unexpected server error.
 */
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
