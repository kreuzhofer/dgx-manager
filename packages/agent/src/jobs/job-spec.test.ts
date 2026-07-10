import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { jobUnitName, jobDir, buildWrapperScript, buildSystemdRunArgv } from "./job-spec.js";

describe("jobUnitName", () => {
  it("prefixes the runId", () => {
    expect(jobUnitName("cmremi4os0033")).toBe("dgxbench-cmremi4os0033");
  });

  // The runId reaches a shell command line and a systemd unit name. It comes
  // from the DB, but "it's a cuid" is an assumption, not a guarantee.
  it.each([["a b"], ["a;id"], ["../x"], ["a$(id)"], [""], ["a/b"], ["a\nb"]])(
    "rejects unsafe runId %j",
    (bad) => {
      expect(() => jobUnitName(bad)).toThrow(/unsafe/i);
    },
  );

  /** Invariant: every accepted unit name is alphanumeric + dash only. */
  test.prop([fc.stringMatching(/^[a-z0-9]{1,32}$/)])("accepts cuid-shaped ids", (id) => {
    expect(jobUnitName(id)).toMatch(/^dgxbench-[A-Za-z0-9]+$/);
  });
});

describe("jobDir", () => {
  it("is per-run under the agent state dir", () => {
    expect(jobDir("/home/daniel", "abc123")).toBe("/home/daniel/.dgx-agent/jobs/abc123");
  });
  it("rejects an unsafe runId", () => {
    expect(() => jobDir("/home/daniel", "../etc")).toThrow(/unsafe/i);
  });
});

describe("buildWrapperScript", () => {
  const script = buildWrapperScript({
    argv: ["uvx", "--from", "lm-eval[api]", "lm_eval", "--tasks", "ifeval"],
    jobDir: "/home/daniel/.dgx-agent/jobs/r1",
    outputDir: "/home/daniel/.dgx-agent/jobs/r1/out",
    resultGlob: "results_*.json",
  });

  it("redirects both streams to the log", () => {
    expect(script).toContain("> log 2>&1");
  });

  /**
   * ORDERING INVARIANT: `exit` must be written LAST, after result.json is in
   * place. The manager treats "exit exists" as "the job is finished and its
   * result is final". Writing exit first would let it read a result.json that
   * has not been copied yet.
   */
  it("writes the exit file after copying the result", () => {
    expect(script.indexOf("result.json")).toBeLessThan(script.lastIndexOf("> exit.tmp"));
  });

  it("writes the exit file atomically", () => {
    expect(script).toContain("mv exit.tmp exit");
  });

  it("preserves the command's exit code, not the copy's", () => {
    expect(script).toMatch(/code=\$\?/);
  });

  it("quotes every argv element", () => {
    expect(script).toContain("'lm-eval[api]'");
  });

  it("neutralises an injected metacharacter in argv", () => {
    const s = buildWrapperScript({
      argv: ["echo", "; rm -rf /"],
      jobDir: "/j", outputDir: "/j/out", resultGlob: "result.json",
    });
    expect(s).toContain("'; rm -rf /'");
    expect(s).not.toMatch(/^; rm -rf \//m);
  });
});

describe("buildSystemdRunArgv", () => {
  const argv = buildSystemdRunArgv({
    unit: "dgxbench-r1", jobDir: "/j", user: "daniel", scriptPath: "/j/cmd.sh",
  });

  it("runs under sudo -n because the agent is not root", () => {
    expect(argv.slice(0, 2)).toEqual(["sudo", "-n"]);
  });

  it("names the transient unit and drops privileges back to the agent user", () => {
    expect(argv).toContain("--unit=dgxbench-r1");
    expect(argv).toContain("-p");
    expect(argv).toContain("User=daniel");
  });

  it("does not use --collect, so the unit survives for status inspection", () => {
    expect(argv).not.toContain("--collect");
  });

  it("execs the wrapper via sh", () => {
    expect(argv.slice(-2)).toEqual(["/bin/sh", "/j/cmd.sh"]);
  });
});
