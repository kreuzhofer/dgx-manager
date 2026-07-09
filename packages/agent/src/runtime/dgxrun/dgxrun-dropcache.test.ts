import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  startDropCacheLoop,
  stopDropCacheLoop,
  isDropCacheLoopRunning,
} from "./dgxrun-dropcache.js";

/** A drop that completes synchronously (the common case: spawn returns at once). */
const fastDrop = () => vi.fn((done: () => void) => done());

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
    const drop = fastDrop();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    expect(drop).toHaveBeenCalledTimes(1); // immediate
    vi.advanceTimersByTime(500);
    expect(drop).toHaveBeenCalledTimes(2);
    vi.advanceTimersByTime(1000);
    expect(drop).toHaveBeenCalledTimes(4);
  });

  // Invariant: stopping halts further drops and clears the running flag.
  it("stops dropping after stop", () => {
    const drop = fastDrop();
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
    const drop = fastDrop();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    expect(drop).toHaveBeenCalledTimes(1); // the 2nd start did NOT drop again
    vi.advanceTimersByTime(500);
    expect(drop).toHaveBeenCalledTimes(2); // one loop firing, not two
  });

  // Invariant: the maxMs backstop stops the loop even without an explicit stop.
  it("auto-stops after maxMs", () => {
    const drop = fastDrop();
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
    const d1 = fastDrop();
    const d2 = fastDrop();
    startDropCacheLoop("d1", { dropFn: d1 });
    startDropCacheLoop("d2", { dropFn: d2 });
    expect(isDropCacheLoopRunning("d1")).toBe(true);
    expect(isDropCacheLoopRunning("d2")).toBe(true);
    stopDropCacheLoop("d1");
    expect(isDropCacheLoopRunning("d1")).toBe(false);
    expect(isDropCacheLoopRunning("d2")).toBe(true);
  });

  // The bug this fixes: a slow `sync` (seconds, under a ~400 GB NFS weight
  // stream) must never queue drops behind each other. While one drop is still
  // running, every tick is skipped — the loop cannot pile up.
  it("skips ticks while a drop is still in flight", () => {
    let finish: (() => void) | null = null;
    const slowDrop = vi.fn((done: () => void) => { finish = done; });
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: slowDrop });
    expect(slowDrop).toHaveBeenCalledTimes(1);

    vi.advanceTimersByTime(5000); // 10 ticks would have fired
    expect(slowDrop).toHaveBeenCalledTimes(1); // still exactly one — all skipped

    finish!(); // the slow drop finally completes
    vi.advanceTimersByTime(500);
    expect(slowDrop).toHaveBeenCalledTimes(2); // loop resumes
  });

  // A drop that throws must not wedge the loop forever by leaving the
  // in-flight flag set.
  it("clears the in-flight flag when a drop throws", () => {
    const boom = vi.fn(() => { throw new Error("spawn failed"); });
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: boom });
    expect(boom).toHaveBeenCalledTimes(1);
    vi.advanceTimersByTime(500);
    expect(boom).toHaveBeenCalledTimes(2); // not stuck
  });

  // Stopping must clear in-flight state, so a later deploy with the same id
  // is not permanently skipped.
  it("stop clears in-flight so a restart can drop again", () => {
    const stuck = vi.fn((_done: () => void) => { /* never completes */ });
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: stuck });
    expect(stuck).toHaveBeenCalledTimes(1);
    stopDropCacheLoop("d1");

    const drop = fastDrop();
    startDropCacheLoop("d1", { intervalMs: 500, maxMs: 60_000, dropFn: drop });
    expect(drop).toHaveBeenCalledTimes(1); // not blocked by the stale flag
  });
});
