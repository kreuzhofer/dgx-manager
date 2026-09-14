import { describe, it, expect, vi, beforeEach } from "vitest";

/**
 * Launch/teardown ORCHESTRATION for a dgxrun rank — specifically the ordering
 * guarantees that used to be bought with blocking spawnSync (#36). Argv
 * construction is `dgxrun-args.ts`'s job and is tested there; this file mocks it
 * out so the sequencing is what's under test.
 */

function makeChild() {
  const h: Record<string, ((a: unknown) => void)[]> = {};
  return {
    stdout: { on: (e: string, cb: (a: unknown) => void) => { (h["stdout." + e] ||= []).push(cb); } },
    stderr: { on: (e: string, cb: (a: unknown) => void) => { (h["stderr." + e] ||= []).push(cb); } },
    on: (e: string, cb: (a: unknown) => void) => { (h[e] ||= []).push(cb); },
    kill: vi.fn(),
    __exit: (code: number | null) => (h["exit"] || []).forEach((cb) => cb(code)),
  };
}

const { spawnMock, execCaptureMock, dropCachesOnceMock } = vi.hoisted(() => ({
  spawnMock: vi.fn(),
  execCaptureMock: vi.fn(),
  dropCachesOnceMock: vi.fn(),
}));

vi.mock("node:child_process", () => ({ spawn: spawnMock }));
vi.mock("../exec-capture.js", () => ({ execCapture: execCaptureMock }));
vi.mock("../deployment-store.js", () => ({ saveDeployment: vi.fn(), removeDeployment: vi.fn() }));
vi.mock("../sparkrun.js", () => ({ resolveHfHome: () => "/mnt/tank/hf" }));
vi.mock("./dgxrun-dropcache.js", () => ({
  dropCachesOnce: dropCachesOnceMock,
  startDropCacheLoop: vi.fn(),
  stopDropCacheLoop: vi.fn(),
}));
vi.mock("./dgxrun-args.js", () => ({
  buildDgxrunDockerArgs: () => ["run", "-d", "--name", "dgxrun_d1", "img:latest"],
  missingModDirs: () => [],
  DEFAULT_MODS_DIR: "/opt/mods",
}));

import { launchDgxrun, stopDgxrun } from "./dgxrun.js";

const ok = { status: 0, stdout: "", stderr: "" };
const RECIPE = { container: "img:latest", model: "m", defaults: { port: 8000 } } as never;
const ARGS = { recipe: RECIPE, rank: 0, nnodes: 1, masterAddr: "10.0.0.1", masterPort: 29500 };

beforeEach(() => {
  spawnMock.mockReset();
  spawnMock.mockImplementation(() => makeChild());
  execCaptureMock.mockReset();
  execCaptureMock.mockResolvedValue(ok);
  dropCachesOnceMock.mockReset();
  dropCachesOnceMock.mockResolvedValue(true);
});

describe("launchDgxrun ordering", () => {
  /**
   * The 2026-07-09 incident: `docker rm -f` overlapping `docker run -d` removes
   * nothing and orphans the container that is being created. The removal used to
   * be sequenced by spawnSync blocking the whole event loop; now it is sequenced
   * by an await, and this test is what keeps it sequenced.
   */
  it("does not start docker run until docker rm -f has completed", async () => {
    let releaseRm!: () => void;
    const rmDone = new Promise<void>((res) => { releaseRm = res; });
    execCaptureMock.mockImplementation(async (_f: string, argv: string[]) => {
      if (argv.includes("rm")) { await rmDone; }
      return ok;
    });

    const launching = launchDgxrun("d1", ARGS, () => {}, () => {});
    await new Promise((r) => setTimeout(r, 20));

    // rm is still in flight — nothing may have been spawned yet.
    expect(spawnMock).not.toHaveBeenCalled();

    releaseRm();
    await launching;
    expect(spawnMock).toHaveBeenCalledTimes(1);
    expect(spawnMock.mock.calls[0]?.[1]).toContain("run");
  });

  // The page cache must be freed BEFORE the container streams its weights, or
  // the drop reclaims nothing that matters and the CUDA-graph capture headroom
  // is already gone (GB10 unified memory).
  it("drops caches before docker run, not after", async () => {
    const order: string[] = [];
    // Records only AFTER a real tick, so the assertion depends on the caller
    // actually awaiting the drop rather than merely starting it.
    dropCachesOnceMock.mockImplementation(async () => {
      await new Promise((r) => setTimeout(r, 10));
      order.push("drop");
      return true;
    });
    spawnMock.mockImplementation(() => { order.push("run"); return makeChild(); });

    await launchDgxrun("d1", ARGS, () => {}, () => {});
    expect(order).toEqual(["drop", "run"]);
  });

  it("fails fast without spawning when the image is absent locally", async () => {
    execCaptureMock.mockImplementation(async (_f: string, argv: string[]) =>
      argv.includes("image") ? { status: 1, stdout: "", stderr: "No such image" } : ok,
    );
    const onExit = vi.fn();
    const logs: string[] = [];

    await launchDgxrun("d1", ARGS, (l) => logs.push(l), onExit);

    expect(spawnMock).not.toHaveBeenCalled();
    expect(onExit).toHaveBeenCalledWith(1);
    expect(logs.join("")).toContain("not found locally");
  });
});

describe("stopDgxrun", () => {
  it("removes the container by name", async () => {
    await stopDgxrun("d1");
    const rm = execCaptureMock.mock.calls.find((c) => (c[1] as string[]).includes("rm"));
    expect(rm?.[1]).toEqual(["rm", "-f", "dgxrun_d1"]);
  });

  // `finally`, not `catch`: a wedged docker must not strand the local record,
  // or the health tick keeps reporting a deployment we have already disowned.
  it("still drops the local record when docker never answers", async () => {
    const { removeDeployment } = await import("../deployment-store.js");
    execCaptureMock.mockResolvedValue({
      status: null, stdout: "", stderr: "", error: new Error("docker timed out after 60000ms"),
    });
    await stopDgxrun("d1");
    expect(removeDeployment).toHaveBeenCalledWith("d1");
  });
});
