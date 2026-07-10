import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { parseSystemctlShow } from "./systemctl-parse.js";

const show = (o: Record<string, string>) =>
  Object.entries(o).map(([k, v]) => `${k}=${v}`).join("\n") + "\n";

describe("parseSystemctlShow", () => {
  it("reads a live unit as active", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "active", ExecMainStatus: "0" }), "");
    expect(r).toEqual({ kind: "active" });
  });

  it("treats activating as active", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "activating", ExecMainStatus: "0" }), "");
    expect(r.kind).toBe("active");
  });

  it("reads a finished unit's exit code", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "inactive", ExecMainStatus: "0" }), "");
    expect(r).toEqual({ kind: "exited", code: 0 });
  });

  it("reads a failed unit's exit code", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "failed", ExecMainStatus: "1" }), "");
    expect(r).toEqual({ kind: "exited", code: 1 });
  });

  it("reports a not-found unit as missing", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "not-found", ActiveState: "inactive", ExecMainStatus: "0" }), "");
    expect(r).toEqual({ kind: "missing" });
  });

  // The whole point. A timeout, a busy dbus, an empty answer: we do not know.
  it.each([
    [null, "", "spawn timeout"],
    [1, "", "Failed to connect to bus"],
    [0, "", ""],
    [0, "garbage without equals signs", ""],
  ])("returns unknown for status=%j stdout=%j", (status, stdout, stderr) => {
    expect(parseSystemctlShow(status as number | null, stdout, stderr).kind).toBe("unknown");
  });

  /**
   * INVARIANT — the one that matters. An inconclusive answer must NEVER be read
   * as a finished job. Reporting `exited(0)` for "we could not tell" would mark a
   * running benchmark complete and parse a result file that does not exist; the
   * mirror-image mistake (`exited(1)`) killed four healthy GLM-5.2 ranks.
   */
  test.prop([
    fc.oneof(fc.constant(null), fc.integer({ min: -1, max: 3 })),
    fc.string(),
    fc.string(),
  ])("never reports exited without an explicit ExecMainStatus", (status, stdout, stderr) => {
    const r = parseSystemctlShow(status, stdout, stderr);
    if (r.kind === "exited") {
      expect(stdout).toMatch(/ExecMainStatus=\d+/);
      expect(stdout).toMatch(/ActiveState=(inactive|failed|deactivating)/);
    }
  });

  /** Invariant: `missing` requires LoadState to positively say not-found. */
  test.prop([fc.string()])("never reports missing without LoadState=not-found", (stdout) => {
    const r = parseSystemctlShow(0, stdout, "");
    if (r.kind === "missing") expect(stdout).toContain("LoadState=not-found");
  });
});
