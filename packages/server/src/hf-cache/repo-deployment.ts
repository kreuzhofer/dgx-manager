/** Durable per-repo "last deployed" store. The models (HF cache) page derives
 *  last-deployed by matching cached repo ids against deployments, but Deployment
 *  rows are hard-deleted by the cleanup workflow — so the timestamp vanished as
 *  soon as a deployment was torn down. This module persists it independently:
 *  recordRepoDeployment stamps at deploy time, loadRepoLastDeployed feeds the
 *  hf-cache route. Keys are lowercased (see deploymentRepoKeys). */
import type { PrismaClient } from "../generated/prisma/client.js";

/** Upsert lastDeployedAt for each repo key. Called at deploy creation with the
 *  deployment's createdAt, which is always the newest timestamp for that repo
 *  at that moment, so a plain overwrite keeps the newest value. No-op on an
 *  empty key set (e.g. a deployment we couldn't resolve any model name for). */
export async function recordRepoDeployment(
  prisma: PrismaClient,
  keys: string[],
  at: Date,
): Promise<void> {
  for (const repoId of keys) {
    await prisma.repoDeployment.upsert({
      where: { repoId },
      create: { repoId, lastDeployedAt: at },
      update: { lastDeployedAt: at },
    });
  }
}

/** Map of lowercased repoId → ISO lastDeployedAt for every recorded repo. The
 *  route looks up each cache repo by repoId.toLowerCase(). */
export async function loadRepoLastDeployed(prisma: PrismaClient): Promise<Map<string, string>> {
  const rows = await prisma.repoDeployment.findMany();
  return new Map(rows.map((r) => [r.repoId, r.lastDeployedAt.toISOString()]));
}
