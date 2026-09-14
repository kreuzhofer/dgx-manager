import { describe, it, expect } from "vitest";
import { execCapture } from "./exec-capture.js";
import { classifyDockerInspect } from "./dgxrun/dgxrun.js";

describe("execCapture", () => {
  it("captures stdout and a zero exit", async () => {
    const r = await execCapture("sh", ["-c", "printf hello"], { timeout: 10_000 });
    expect(r.status).toBe(0);
    expect(r.stdout).toBe("hello");
    expect(r.error).toBeUndefined();
  });

  it("captures stderr alongside a non-zero exit code", async () => {
    const r = await execCapture("sh", ["-c", "printf oops >&2; exit 7"], { timeout: 10_000 });
    expect(r.status).toBe(7);
    expect(r.stderr).toBe("oops");
    expect(r.error).toBeUndefined();
  });

  // THE contract this whole helper exists to preserve. spawnSync reports a
  // timeout as `status === null` with `error` set, and classifyDockerInspect
  // keys on exactly that to return `unknown` rather than `absent`. An async
  // replacement that reported, say, status 143 would silently turn "docker
  // didn't answer" into "the container is gone" — which is the bug that tore
  // down a healthy four-rank cluster (#36).
  it("reports a timeout as status null with an error set", async () => {
    const r = await execCapture("sh", ["-c", "sleep 30"], { timeout: 250 });
    expect(r.status).toBeNull();
    expect(r.error).toBeInstanceOf(Error);
    expect(r.error?.message).toMatch(/timed out/i);
  });

  it("settles at the deadline rather than waiting for a wedged child", async () => {
    const started = Date.now();
    // Ignores SIGTERM, so a helper that waited for `close` would hang here.
    await execCapture("sh", ["-c", "trap '' TERM; sleep 30"], { timeout: 250 });
    expect(Date.now() - started).toBeLessThan(2_000);
  });

  it("reports a missing binary as status null with an error set", async () => {
    const r = await execCapture("definitely-not-a-real-binary-xyz", [], { timeout: 10_000 });
    expect(r.status).toBeNull();
    expect(r.error).toBeInstanceOf(Error);
  });

  it("truncates at maxBuffer instead of failing the call", async () => {
    const r = await execCapture("sh", ["-c", "printf 'aaaaaaaaaa%.0s' $(seq 1 100)"], {
      timeout: 10_000, maxBuffer: 64,
    });
    expect(r.status).toBe(0);
    expect(r.stdout.length).toBeLessThanOrEqual(1024);
    expect(r.error).toBeUndefined();
  });
});

describe("execCapture feeding classifyDockerInspect", () => {
  // The seam, asserted end-to-end: a timed-out inspect must classify as
  // `unknown` (skip the tick, leave the last known status standing), never
  // `absent` (report the container gone → the manager tears the cluster down).
  it("classifies a timed-out inspect as unknown, never absent", async () => {
    const r = await execCapture("sh", ["-c", "sleep 30"], { timeout: 250 });
    const res = classifyDockerInspect(r.status, r.stdout, r.stderr, "dgxrun_x", r.error);
    expect(res.kind).toBe("unknown");
  });

  it("still classifies a genuine 'no such object' as absent", async () => {
    const r = await execCapture("sh", ["-c", "printf 'Error: No such object: dgxrun_x' >&2; exit 1"], {
      timeout: 10_000,
    });
    const res = classifyDockerInspect(r.status, r.stdout, r.stderr, "dgxrun_x", r.error);
    expect(res.kind).toBe("absent");
  });
});
