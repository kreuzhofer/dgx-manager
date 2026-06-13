import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import { mkdtempSync, mkdirSync, writeFileSync, existsSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseRepoDirName, repoDirName, isSafeRepoId, deleteCachedRepo } from "./hf-cache.js";

describe("parseRepoDirName", () => {
  it("decodes a model repo dir", () => {
    expect(parseRepoDirName("models--meta-llama--Llama-3.1-8B-Instruct")).toEqual({
      kind: "model",
      repoId: "meta-llama/Llama-3.1-8B-Instruct",
    });
  });

  it("decodes a dataset repo dir", () => {
    expect(parseRepoDirName("datasets--HuggingFaceH4--ultrachat_200k")).toEqual({
      kind: "dataset",
      repoId: "HuggingFaceH4/ultrachat_200k",
    });
  });

  it("decodes an org-less legacy repo (models--gpt2)", () => {
    expect(parseRepoDirName("models--gpt2")).toEqual({ kind: "model", repoId: "gpt2" });
  });

  it("does not split single dashes inside names", () => {
    expect(parseRepoDirName("models--meta-llama--Meta-Llama-3-8B")).toEqual({
      kind: "model",
      repoId: "meta-llama/Meta-Llama-3-8B",
    });
  });

  it("returns null for non-repo hub entries", () => {
    expect(parseRepoDirName("version.txt")).toBeNull();
    expect(parseRepoDirName(".locks")).toBeNull();
    expect(parseRepoDirName("spaces--foo--bar")).toBeNull(); // unsupported kind
    expect(parseRepoDirName("models--")).toBeNull();          // empty segment
  });
});

describe("repoDirName", () => {
  it("encodes model and dataset repos", () => {
    expect(repoDirName("model", "meta-llama/Llama-3.1-8B-Instruct"))
      .toBe("models--meta-llama--Llama-3.1-8B-Instruct");
    expect(repoDirName("dataset", "squad")).toBe("datasets--squad");
  });
});

/** A single repoId segment as HF allows it: letters, digits, dot, dash,
 *  underscore — excluding `.`/`..` and any `--` (which would be ambiguous in
 *  the directory encoding, a limitation huggingface_hub shares).
 *  Also exclude leading/trailing `-` so that joining two segments with `--`
 *  never produces a run of three or more dashes (which re-introduces `--`). */
const segmentArb = fc
  .stringMatching(/^[A-Za-z0-9._-]{1,32}$/)
  .filter((s) => s !== "." && s !== ".." && !s.includes("--") && !s.startsWith("-") && !s.endsWith("-"));

const repoIdArb = fc
  .oneof(segmentArb, fc.tuple(segmentArb, segmentArb).map(([a, b]) => `${a}/${b}`));

describe("codec round-trip", () => {
  /** Invariant: for any valid repoId whose segments contain no `--`,
   *  encoding to a cache dir name and parsing it back is the identity. */
  fcIt.prop([repoIdArb, fc.constantFrom("model" as const, "dataset" as const)])(
    "parseRepoDirName(repoDirName(kind, id)) === {kind, id}",
    (repoId, kind) => {
      expect(parseRepoDirName(repoDirName(kind, repoId))).toEqual({ kind, repoId });
    },
  );
});

describe("isSafeRepoId", () => {
  it("accepts normal one- and two-segment ids", () => {
    expect(isSafeRepoId("gpt2")).toBe(true);
    expect(isSafeRepoId("meta-llama/Llama-3.1-8B-Instruct")).toBe(true);
  });

  it("rejects traversal and malformed ids", () => {
    expect(isSafeRepoId("")).toBe(false);
    expect(isSafeRepoId("..")).toBe(false);
    expect(isSafeRepoId("../etc")).toBe(false);
    expect(isSafeRepoId("a/..")).toBe(false);
    expect(isSafeRepoId("./a")).toBe(false);
    expect(isSafeRepoId("a/b/c")).toBe(false);
    expect(isSafeRepoId("/etc")).toBe(false);
    expect(isSafeRepoId("a b")).toBe(false);
    expect(isSafeRepoId("a\\b")).toBe(false);
  });

  /** Invariant: every id our generator considers valid is accepted. */
  fcIt.prop([repoIdArb])("accepts all generator-valid ids", (repoId) => {
    expect(isSafeRepoId(repoId)).toBe(true);
  });
});

/** Build a minimal fake cache: hfHome/hub/<repoDir>/blobs/weights */
function makeFakeCache(repoDirs: string[]): string {
  const hfHome = mkdtempSync(join(tmpdir(), "hf-cache-test-"));
  for (const dir of repoDirs) {
    const blobs = join(hfHome, "hub", dir, "blobs");
    mkdirSync(blobs, { recursive: true });
    writeFileSync(join(blobs, "weights"), "x".repeat(1000));
  }
  return hfHome;
}

describe("deleteCachedRepo", () => {
  it("deletes the targeted repo dir and leaves siblings alone", () => {
    const hfHome = makeFakeCache(["models--org--alpha", "models--org--beta"]);
    deleteCachedRepo(hfHome, "model", "org/alpha");
    expect(existsSync(join(hfHome, "hub", "models--org--alpha"))).toBe(false);
    expect(existsSync(join(hfHome, "hub", "models--org--beta"))).toBe(true);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("deletes dataset repos via kind", () => {
    const hfHome = makeFakeCache(["datasets--squad"]);
    deleteCachedRepo(hfHome, "dataset", "squad");
    expect(existsSync(join(hfHome, "hub", "datasets--squad"))).toBe(false);
    rmSync(hfHome, { recursive: true, force: true });
  });

  it("throws for a repo that is not in the cache", () => {
    const hfHome = makeFakeCache([]);
    expect(() => deleteCachedRepo(hfHome, "model", "org/ghost")).toThrow(/not in cache/i);
    rmSync(hfHome, { recursive: true, force: true });
  });

  /** Invariant: any unsafe repoId is rejected BEFORE any filesystem access,
   *  and nothing outside hub/ is ever touched. We plant a sentinel file
   *  outside hub/ and assert it survives every attempt. */
  fcIt.prop([fc.string({ minLength: 1, maxLength: 64 }).filter((s) => !isSafeRepoId(s))])(
    "rejects every unsafe repoId without touching the filesystem",
    (badId) => {
      const hfHome = makeFakeCache(["models--org--alpha"]);
      writeFileSync(join(hfHome, "sentinel.txt"), "intact");
      try {
        expect(() => deleteCachedRepo(hfHome, "model", badId)).toThrow(/invalid repoId/i);
        expect(existsSync(join(hfHome, "sentinel.txt"))).toBe(true);
        expect(existsSync(join(hfHome, "hub", "models--org--alpha"))).toBe(true);
      } finally {
        rmSync(hfHome, { recursive: true, force: true });
      }
    },
  );

  it("rejects classic traversal attempts", () => {
    const hfHome = makeFakeCache(["models--org--alpha"]);
    for (const evil of ["../..", "a/..", "../hub", "/etc", "..\\..", "org/../alpha"]) {
      expect(() => deleteCachedRepo(hfHome, "model", evil)).toThrow(/invalid repoId/i);
    }
    rmSync(hfHome, { recursive: true, force: true });
  });
});
