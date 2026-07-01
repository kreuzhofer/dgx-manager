import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import {
  groupInventories, matchRepoToModels, deploymentModelCandidates, deploymentRepoKeys,
  newerIso, repoUsage,
  type HfCacheNodeInventory, type DeploymentUsage,
} from "./grouping.js";

function inv(nodeId: string, cacheId: string, scannedAt = "2026-06-13T10:00:00.000Z"): HfCacheNodeInventory {
  return {
    nodeId, cacheId, scannedAt,
    hfHome: "/mnt/tank/models", totalBytes: 0, diskFreeBytes: 0, repos: [],
  };
}

describe("groupInventories", () => {
  it("merges nodes sharing a cacheId into one group (shared NFS)", () => {
    const groups = groupInventories([inv("n1", "cache-A"), inv("n2", "cache-A")]);
    expect(groups).toHaveLength(1);
    expect(groups[0].cacheId).toBe("cache-A");
    expect(groups[0].nodeIds.sort()).toEqual(["n1", "n2"]);
  });

  it("keeps distinct cacheIds separate (per-node local disks)", () => {
    const groups = groupInventories([inv("n1", "cache-A"), inv("n2", "cache-B")]);
    expect(groups).toHaveLength(2);
  });

  it("handles mixed topologies", () => {
    const groups = groupInventories([inv("n1", "shared"), inv("n2", "shared"), inv("n3", "local-3")]);
    expect(groups).toHaveLength(2);
    expect(groups.find((g) => g.cacheId === "shared")!.nodeIds).toHaveLength(2);
  });

  it("falls back to a per-node group when cacheId is empty (scan-error inventories)", () => {
    const groups = groupInventories([inv("n1", ""), inv("n2", "")]);
    expect(groups).toHaveLength(2);
    expect(groups.map((g) => g.cacheId).sort()).toEqual(["node:n1", "node:n2"]);
  });

  it("the newest scannedAt inventory wins within a group", () => {
    const stale = inv("n1", "shared", "2026-06-13T09:00:00.000Z");
    const fresh = inv("n2", "shared", "2026-06-13T11:00:00.000Z");
    fresh.totalBytes = 42;
    const groups = groupInventories([stale, fresh]);
    expect(groups[0].newest.nodeId).toBe("n2");
    expect(groups[0].newest.totalBytes).toBe(42);
  });
});

describe("matchRepoToModels", () => {
  /** Invariant: matching is case-insensitive in both directions and ignores
   *  null/undefined candidates. */
  fcIt.prop([fc.stringMatching(/^[A-Za-z0-9._-]{1,20}\/[A-Za-z0-9._-]{1,20}$/)])(
    "matches its own upper/lower-cased variants",
    (repoId) => {
      expect(matchRepoToModels(repoId, [repoId.toUpperCase()])).toBe(true);
      expect(matchRepoToModels(repoId, [repoId.toLowerCase()])).toBe(true);
      expect(matchRepoToModels(repoId, [null, undefined, "unrelated/model"])).toBe(false);
    },
  );
});

describe("deploymentModelCandidates", () => {
  it("collects model name and config modelName", () => {
    expect(deploymentModelCandidates("org/m1", JSON.stringify({ modelName: "org/m2" })))
      .toEqual(["org/m1", "org/m2"]);
  });

  it("survives malformed config JSON", () => {
    expect(deploymentModelCandidates("org/m1", "{not json")).toEqual(["org/m1"]);
  });

  it("handles nulls", () => {
    expect(deploymentModelCandidates(null, null)).toEqual([]);
  });
});

describe("deploymentRepoKeys", () => {
  it("collects all four sources, lowercased and deduped", () => {
    expect(
      deploymentRepoKeys(
        "Recipe-Slug",
        JSON.stringify({ modelName: "Org/Alpha", recipeFile: "recipes/x.yaml" }),
        "Org/Alpha", // recipe HF id — dupes config.modelName after lowercasing
        "Org/Base",
      ).sort(),
    ).toEqual(["org/alpha", "org/base", "recipe-slug"]);
  });

  it("drops empties and survives all-null input", () => {
    expect(deploymentRepoKeys(null, null, null, null)).toEqual([]);
    expect(deploymentRepoKeys("", "{}", "", "")).toEqual([]);
  });

  it("includes the recipe HF id for a registry-ref deploy whose Model.name is a slug", () => {
    // The exact bug scenario: Model.name is the recipe slug, cached repo is the HF id.
    const keys = deploymentRepoKeys("gemma4-26b-a4b", JSON.stringify({ recipeFile: "recipes/g.yaml" }),
      "google/gemma-4-26B-A4B-it", null);
    expect(keys).toContain("google/gemma-4-26b-a4b-it");
  });
});

describe("newerIso", () => {
  it("returns the later timestamp, tolerating nulls on either side", () => {
    expect(newerIso("2026-05-01T00:00:00.000Z", "2026-06-01T00:00:00.000Z")).toBe("2026-06-01T00:00:00.000Z");
    expect(newerIso("2026-06-01T00:00:00.000Z", "2026-05-01T00:00:00.000Z")).toBe("2026-06-01T00:00:00.000Z");
    expect(newerIso(null, "2026-06-01T00:00:00.000Z")).toBe("2026-06-01T00:00:00.000Z");
    expect(newerIso("2026-06-01T00:00:00.000Z", null)).toBe("2026-06-01T00:00:00.000Z");
    expect(newerIso(null, null)).toBeNull();
  });
});

function dep(over: Partial<DeploymentUsage>): DeploymentUsage {
  return {
    status: "running", nodeId: "n1", createdAt: "2026-06-01T00:00:00.000Z",
    label: "my-deploy", candidates: ["org/alpha"], clusterNodeIds: [],
    ...over,
  };
}

describe("repoUsage", () => {
  const group = new Set(["n1", "n2"]);

  it("flags in-use for an active deployment on a group node", () => {
    const usage = repoUsage("org/alpha", group, [dep({})]);
    expect(usage.inUse).toBe(true);
    expect(usage.inUseBy).toEqual(["my-deploy"]);
  });

  it("stopped/failed deployments do not block but still set lastDeployedAt", () => {
    const usage = repoUsage("org/alpha", group, [
      dep({ status: "stopped", createdAt: "2026-05-01T00:00:00.000Z" }),
      dep({ status: "failed", createdAt: "2026-05-20T00:00:00.000Z" }),
    ]);
    expect(usage.inUse).toBe(false);
    expect(usage.lastDeployedAt).toBe("2026-05-20T00:00:00.000Z");
  });

  it("evicted deployments count as in use (they can be restored onto the GPU)", () => {
    expect(repoUsage("org/alpha", group, [dep({ status: "evicted" })]).inUse).toBe(true);
  });

  it("ignores deployments on nodes outside the cache group", () => {
    const usage = repoUsage("org/alpha", group, [dep({ nodeId: "other" })]);
    expect(usage.inUse).toBe(false);
    expect(usage.lastDeployedAt).toBeNull();
  });

  it("counts multi-node deployments whose cluster nodes intersect the group", () => {
    const usage = repoUsage("org/alpha", group, [dep({ nodeId: "other", clusterNodeIds: ["n2"] })]);
    expect(usage.inUse).toBe(true);
  });

  it("never matches a repo no deployment references", () => {
    expect(repoUsage("org/unrelated", group, [dep({})]).lastDeployedAt).toBeNull();
  });
});
