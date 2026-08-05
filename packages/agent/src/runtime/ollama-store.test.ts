import { afterAll, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

/**
 * Ollama deployments must leave the persistent store the way sparkrun and
 * dgxrun do. They are what lets the reconnect reconcile restore them after a
 * reboot — and an entry that outlives its deployment is worse than no entry at
 * all, because the reconcile would keep bringing a dead deployment back.
 */

// The store reads its path at module load, so this must precede the imports.
const TMP = mkdtempSync(join(tmpdir(), "ollama-store-"));
process.env.DEPLOYMENT_STORE_PATH = join(TMP, "deployments.json");

const { clearDeployments, loadDeployments, saveDeployment } = await import("./deployment-store.js");
const { stopModel } = await import("./ollama.js");

beforeEach(clearDeployments);
afterAll(() => rmSync(TMP, { recursive: true, force: true }));

const record = (deploymentId: string) => ({
  deploymentId,
  kind: "ollama" as const,
  recipeFile: "qwen3-embedding:8b",
  recipeName: "qwen3-embedding:8b",
  port: 11434,
  startedAt: "2026-08-05T00:00:00.000Z",
});

describe("stopModel — persistent store", () => {
  // The orphan case: after an agent restart the in-memory tracking is empty, so
  // an undeploy could return early and leave the record behind forever.
  it("removes the record even when the deployment is not tracked in memory", async () => {
    saveDeployment(record("d1"));
    expect(loadDeployments()).toHaveLength(1);

    // No model name known and none supplied — the early-return path.
    await stopModel("d1");

    expect(loadDeployments()).toEqual([]);
  });

  it("removes the record when the model name is supplied by the manager", async () => {
    saveDeployment(record("d2"));

    await stopModel("d2", "qwen3-embedding:8b");

    expect(loadDeployments()).toEqual([]);
  });

  it("leaves other deployments' records alone", async () => {
    saveDeployment(record("keep"));
    saveDeployment(record("drop"));

    await stopModel("drop", "qwen3-embedding:8b");

    expect(loadDeployments().map((d) => d.deploymentId)).toEqual(["keep"]);
  });
});
