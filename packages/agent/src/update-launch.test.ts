import { describe, it, expect } from "vitest";
import { launchUpdater } from "./update-launch.js";

function harness(inFlight = false) {
  const calls: string[] = [];
  const deps = {
    bundleUrl: "u", version: "1.0.0", updaterPath: "/opt/dgx-agent/updater.js",
    nodeIdFile: "/opt/dgx-agent/node-id", tmpPath: "/tmp/dgx-updater-1.js",
    copyFile: (_a: string, _b: string) => calls.push("copy"),
    spawnDetached: (_c: string, _a: string[]) => calls.push("spawn"),
    lockExists: () => inFlight,
    makeLock: () => calls.push("lock"),
  };
  return { deps, calls };
}

describe("launchUpdater", () => {
  it("copies updater, makes lock, spawns detached, returns launched", () => {
    const h = harness(false);
    expect(launchUpdater(h.deps)).toBe("launched");
    expect(h.calls).toEqual(["lock", "copy", "spawn"]);
  });
  it("skips when an update is already in flight", () => {
    const h = harness(true);
    expect(launchUpdater(h.deps)).toBe("in-flight");
    expect(h.calls).toEqual([]);
  });
});
