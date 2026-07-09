import { describe, expect, it, afterAll } from "vitest";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync, utimesSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { findLmEvalResultFile } from "./lm-eval-result-file.js";

const dirs: string[] = [];
function tmp(): string {
  const d = mkdtempSync(join(tmpdir(), "lm-eval-rf-"));
  dirs.push(d);
  return d;
}
afterAll(() => { for (const d of dirs) rmSync(d, { recursive: true, force: true }); });

describe("findLmEvalResultFile", () => {
  it("returns null when there is no results file", () => {
    expect(findLmEvalResultFile(tmp())).toBeNull();
  });

  it("finds a results file nested in a model subdirectory", () => {
    const root = tmp();
    const sub = join(root, "some__model");
    mkdirSync(sub);
    const f = join(sub, "results_2026-07-09T10-00-00.json");
    writeFileSync(f, "{}");
    expect(findLmEvalResultFile(root)).toBe(f);
  });

  it("returns the newest results file by mtime when several exist", () => {
    const root = tmp();
    const older = join(root, "results_2026-07-09T09-00-00.json");
    const newer = join(root, "results_2026-07-09T11-00-00.json");
    writeFileSync(older, "{}");
    writeFileSync(newer, "{}");
    utimesSync(older, new Date(1_000_000), new Date(1_000_000));
    utimesSync(newer, new Date(2_000_000), new Date(2_000_000));
    expect(findLmEvalResultFile(root)).toBe(newer);
  });

  it("ignores non-results json files", () => {
    const root = tmp();
    writeFileSync(join(root, "samples_ifeval.json"), "{}");
    expect(findLmEvalResultFile(root)).toBeNull();
  });
});
