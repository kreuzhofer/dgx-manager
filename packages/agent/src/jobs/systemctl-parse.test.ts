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

  it.each([
    ["empty ExecMainStatus", "LoadState=loaded\nActiveState=failed\nExecMainStatus=\n"],
    ["whitespace ExecMainStatus", "LoadState=loaded\nActiveState=failed\nExecMainStatus=   \n"],
    ["hex ExecMainStatus", "LoadState=loaded\nActiveState=failed\nExecMainStatus=0x1\n"],
  ])("treats a finished unit with a non-integer ExecMainStatus (%s) as unknown, not a false exit", (_label, stdout) => {
    expect(parseSystemctlShow(0, stdout, "").kind).toBe("unknown");
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
  // Build systemctl-show-shaped output so the property actually exercises the
  // exited/finished branch, instead of bare fc.string() which almost never does.
  const activeStateArb = fc.constantFrom("active", "activating", "inactive", "failed", "deactivating", "reloading");
  const execStatusArb = fc.oneof(
    fc.integer({ min: -1, max: 255 }).map(String), // valid integer
    fc.constantFrom("", "   ", "0x1", "n/a", "abc"), // non-integer / empty
  );
  test.prop([fc.constantFrom("loaded", "not-found"), activeStateArb, execStatusArb])(
    "never reports exited without an explicit integer ExecMainStatus",
    (load, active, exec) => {
      const stdout = `LoadState=${load}\nActiveState=${active}\nExecMainStatus=${exec}\n`;
      const r = parseSystemctlShow(0, stdout, "");
      if (r.kind === "exited") {
        // exited is only legal when ExecMainStatus is an explicit integer string
        expect(/^-?\d+$/.test(exec)).toBe(true);
      }
    },
  );

  /** Invariant: `missing` requires LoadState to positively say not-found. */
  test.prop([fc.string()])("never reports missing without LoadState=not-found", (stdout) => {
    const r = parseSystemctlShow(0, stdout, "");
    if (r.kind === "missing") expect(stdout).toContain("LoadState=not-found");
  });
});
