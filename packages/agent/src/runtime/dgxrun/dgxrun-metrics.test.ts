import { describe, it, expect, vi, beforeEach } from "vitest";

/**
 * The health tick's absent-handling. Specifically the launch race: between
 * `persist()` and the container existing, `docker inspect` correctly reports
 * `absent` — and reporting that as "container missing" tears down every rank.
 */

const { loadDeploymentsMock, inspectMock, captureLogsMock } = vi.hoisted(() => ({
  loadDeploymentsMock: vi.fn(),
  inspectMock: vi.fn(),
  captureLogsMock: vi.fn(),
}));

vi.mock("../deployment-store.js", () => ({ loadDeployments: loadDeploymentsMock }));
vi.mock("./dgxrun.js", () => ({
  inspectDgxrunContainerResult: inspectMock,
  captureCrashedDgxrunLogs: captureLogsMock,
  dgxrunContainerName: (id: string) => `dgxrun_${id}`,
}));
vi.mock("./dgxrun-dropcache.js", () => ({ stopDropCacheLoop: vi.fn() }));

import { checkDgxrunDeployments } from "./dgxrun-metrics.js";

const dep = (over: Record<string, unknown> = {}) => ({
  deploymentId: "d1", recipeFile: "img", recipeName: "m", port: 8000,
  startedAt: new Date().toISOString(), kind: "dgxrun", rank: 0, ...over,
});

beforeEach(() => {
  loadDeploymentsMock.mockReset();
  inspectMock.mockReset();
  captureLogsMock.mockReset();
  captureLogsMock.mockResolvedValue("");
});

describe("checkDgxrunDeployments — absent handling", () => {
  /**
   * The #91 gate-1 failure, 2026-09-14. `launchDgxrun` persists the deployment
   * BEFORE `docker run`, so there is a window where the store has the entry and
   * the container does not exist. That window used to be covered accidentally:
   * `dropCachesOnce()` was a blocking spawnSync, which froze the event loop and
   * stopped this tick from running at all. Making it async (#36) removed that
   * accidental mutual exclusion and the tick started killing launches — two
   * absent ticks, "container missing", manager tears down every rank, and the
   * container was never even created.
   */
  it("does NOT report a starting deployment as missing, however many ticks", async () => {
    loadDeploymentsMock.mockReturnValue([dep({ starting: true })]);
    inspectMock.mockResolvedValue({ kind: "absent" });

    for (let i = 0; i < 5; i++) {
      const out = await checkDgxrunDeployments();
      expect(out).toEqual([]);
    }
  });

  // Once docker run has returned, `starting` is cleared and absent means absent.
  it("still reports a non-starting deployment missing after two absent ticks", async () => {
    loadDeploymentsMock.mockReturnValue([dep()]);
    inspectMock.mockResolvedValue({ kind: "absent" });

    expect(await checkDgxrunDeployments()).toEqual([]); // first absent: tolerated
    const out = await checkDgxrunDeployments();
    expect(out).toHaveLength(1);
    expect(out[0]).toMatchObject({ alive: false, error: "container missing" });
  });

  // Pre-existing behaviour that must not regress.
  it("does not report an intentionally stopping deployment as missing", async () => {
    loadDeploymentsMock.mockReturnValue([dep({ stopping: true })]);
    inspectMock.mockResolvedValue({ kind: "absent" });
    expect(await checkDgxrunDeployments()).toEqual([]);
    expect(await checkDgxrunDeployments()).toEqual([]);
  });

  // `unknown` means we failed to ASK docker; it must never be read as absent.
  it("skips the tick entirely when the inspect is inconclusive", async () => {
    loadDeploymentsMock.mockReturnValue([dep()]);
    inspectMock.mockResolvedValue({ kind: "unknown", reason: "timed out" });
    expect(await checkDgxrunDeployments()).toEqual([]);
    expect(await checkDgxrunDeployments()).toEqual([]);
  });
});
