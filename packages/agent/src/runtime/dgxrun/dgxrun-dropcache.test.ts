import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  startDropCacheLoop,
  stopDropCacheLoop,
  isDropCacheLoopRunning,
} from "./dgxrun-dropcache.js";

describe("dgxrun drop-cache loop", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    // The loops registry is module-level — clear anything a test left running.
    stopDropCacheLoop("d1");
    stopDropCacheLoop("d2");
    vi.useRealTimers();
  });

  // Invariant: a started loop drops once immediately, then once per interval.
  it("drops immediately and on each interval", () => {
    const drop = vi.fn();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    expect(drop).toHaveBeenCalledTimes(1); // immediate
    vi.advanceTimersByTime(500);
    expect(drop).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(1000);
    expect(drop).toHaveBeenCalledTimes(4);
  });

  // Invariant: stopping halts further drops and clears the running flag.
  it("stops dropping after stop", () => {
    const drop = vi.fn();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    vi.advanceTimersByTime(500);
    expect(drop).toHaveBeenCalledTimes(2);
    stopDropCacheLoop("d1");
    expect(isDropCacheLoopRunning("d1")).toBe(false);
    vi.advanceTimersByTime(5000);
    expect(drop).toHaveBeenCalledTimes(2); // no more drops
  });

  // Invariant: a second start for the same id is a no-op (exactly one loop).
  it("is idempotent per deploymentId", () => {
    const drop = vi.fn();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    expect(drop).toHaveBeenCalledTimes(1); // the 2nd start did NOT drop again
    vi.advanceTimersByTime(500);
    expect(drop).toHaveBeenCalledTimes(2); // one loop firing, not two
  });

  // Invariant: the maxMs backstop stops the loop even without an explicit stop.
  it("auto-stops after maxMs", () => {
    const drop = vi.fn();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 2000, dropFn: drop });
    expect(isDropCacheLoopRunning("d1")).toBe(true);
    vi.advanceTimersByTime(2000); // hit the cap
    expect(isDropCacheLoopRunning("d1")).toBe(false);
    const after = drop.mock.calls.length;
    vi.advanceTimersByTime(5000);
    expect(drop).toHaveBeenCalledTimes(after); // no drops after auto-stop
  });

  // Invariant: deployments have independent loops.
  it("tracks loops independently per deployment", () => {
    const d1 = vi.fn();
    const d2 = vi.fn();
    startDropCacheLoop("d1", { dropFn: d1 });
    startDropCacheLoop("d2", { dropFn: d2 });
    expect(isDropCacheLoopRunning("d1")).toBe(true);
    expect(isDropCacheLoopRunning("d2")).toBe(true);
    stopDropCacheLoop("d1");
    expect(isDropCacheLoopRunning("d1")).toBe(false);
    expect(isDropCacheLoopRunning("d2")).toBe(true);
  });
});
