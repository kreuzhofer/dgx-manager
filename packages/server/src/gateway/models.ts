/**
 * The cluster's model list, assembled from what the manager knows it deployed.
 *
 * It is never proxied. An allocation-inducing runtime's own list enumerates
 * every model present on that node, including ones nobody deployed — exactly
 * what the gateway exists to conceal. See docs/adr/0001-inference-gateway.md,
 * Decision 2.
 */

/** One entry in an OpenAI `/v1/models` response. */
export interface OpenAiModel {
  id: string;
  object: "model";
  /** Unix seconds — when the cluster started serving this name. */
  created: number;
  owned_by: string;
}

export interface OpenAiModelList {
  object: "list";
  data: OpenAiModel[];
}

/** A running deployment, as far as the model list is concerned. */
export interface PublishedDeployment {
  publishedName: string | null;
  createdAt: Date;
}

const OWNER = "dgx-manager";

/**
 * Distinct published names, sorted, each dated from the earliest deployment
 * serving it.
 *
 * Deployments sharing a name are a **pool** — one model to a client, which
 * picks a name and never a member. A deployment with no published name yet is
 * not serving under any name a client could ask for, so it is not listed.
 */
export function toModelList(deployments: PublishedDeployment[]): OpenAiModelList {
  const earliest = new Map<string, number>();

  for (const d of deployments) {
    if (!d.publishedName) continue;
    const created = Math.floor(d.createdAt.getTime() / 1000);
    const seen = earliest.get(d.publishedName);
    if (seen === undefined || created < seen) earliest.set(d.publishedName, created);
  }

  return {
    object: "list",
    data: [...earliest.entries()]
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([id, created]) => ({ id, object: "model" as const, created, owned_by: OWNER })),
  };
}
