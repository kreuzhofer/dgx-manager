import { describe, it, expect } from "vitest";
import { formatBytes, sortRepos, type CacheRepo } from "./hf-cache";

describe("formatBytes", () => {
  it("formats across magnitudes", () => {
    expect(formatBytes(0)).toBe("0.0 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(8_000_000_000)).toBe("7.5 GB");
    expect(formatBytes(140 * 1024 ** 3)).toBe("140 GB");
  });

  it("is defensive about garbage", () => {
    expect(formatBytes(-1)).toBe("—");
    expect(formatBytes(NaN)).toBe("—");
  });
});

function r(over: Partial<CacheRepo>): CacheRepo {
  return {
    repoId: "org/x", kind: "model", sizeBytes: 100, nFiles: 1, revisions: 1,
    lastModified: "2026-06-01T00:00:00.000Z", lastDeployedAt: null,
    inUse: false, inUseBy: [],
    ...over,
  };
}

describe("sortRepos", () => {
  it("downloaded desc: newest download first (the default view)", () => {
    const sorted = sortRepos(
      [
        r({ repoId: "old", lastModified: "2026-01-01T00:00:00.000Z" }),
        r({ repoId: "new", lastModified: "2026-06-10T00:00:00.000Z" }),
        r({ repoId: "mid", lastModified: "2026-03-01T00:00:00.000Z" }),
      ],
      "downloaded", "desc",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["new", "mid", "old"]);
  });

  it("downloaded asc: oldest download first", () => {
    const sorted = sortRepos(
      [
        r({ repoId: "new", lastModified: "2026-06-10T00:00:00.000Z" }),
        r({ repoId: "old", lastModified: "2026-01-01T00:00:00.000Z" }),
      ],
      "downloaded", "asc",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["old", "new"]);
  });

  it("size desc / asc", () => {
    const data = [r({ repoId: "a", sizeBytes: 1 }), r({ repoId: "b", sizeBytes: 9 })];
    expect(sortRepos(data, "size", "desc").map((x) => x.repoId)).toEqual(["b", "a"]);
    expect(sortRepos(data, "size", "asc").map((x) => x.repoId)).toEqual(["a", "b"]);
  });

  it("revisions desc", () => {
    const sorted = sortRepos(
      [r({ repoId: "one", revisions: 1 }), r({ repoId: "three", revisions: 3 })],
      "revisions", "desc",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["three", "one"]);
  });

  it("repoId asc / desc (alphabetical)", () => {
    const data = [r({ repoId: "zeta/m" }), r({ repoId: "alpha/m" })];
    expect(sortRepos(data, "repoId", "asc").map((x) => x.repoId)).toEqual(["alpha/m", "zeta/m"]);
    expect(sortRepos(data, "repoId", "desc").map((x) => x.repoId)).toEqual(["zeta/m", "alpha/m"]);
  });

  it("kind asc groups datasets before models", () => {
    const sorted = sortRepos(
      [r({ repoId: "m", kind: "model" }), r({ repoId: "d", kind: "dataset" })],
      "kind", "asc",
    );
    expect(sorted.map((x) => x.kind)).toEqual(["dataset", "model"]);
  });

  it("lastDeployed asc: never-deployed first, then oldest (stalest-first cleanup view)", () => {
    const sorted = sortRepos(
      [
        r({ repoId: "recent", lastDeployedAt: "2026-06-10T00:00:00.000Z" }),
        r({ repoId: "never", lastDeployedAt: null }),
        r({ repoId: "old", lastDeployedAt: "2026-01-01T00:00:00.000Z" }),
      ],
      "lastDeployed", "asc",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["never", "old", "recent"]);
  });

  it("lastDeployed desc: most-recently deployed first, never-deployed last", () => {
    const sorted = sortRepos(
      [
        r({ repoId: "recent", lastDeployedAt: "2026-06-10T00:00:00.000Z" }),
        r({ repoId: "never", lastDeployedAt: null }),
        r({ repoId: "old", lastDeployedAt: "2026-01-01T00:00:00.000Z" }),
      ],
      "lastDeployed", "desc",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["recent", "old", "never"]);
  });

  it("breaks ties deterministically by repoId regardless of input order or direction", () => {
    const a = r({ repoId: "aaa/m", sizeBytes: 5 });
    const b = r({ repoId: "bbb/m", sizeBytes: 5 });
    expect(sortRepos([b, a], "size", "desc").map((x) => x.repoId)).toEqual(["aaa/m", "bbb/m"]);
    expect(sortRepos([a, b], "size", "desc").map((x) => x.repoId)).toEqual(["aaa/m", "bbb/m"]);
    expect(sortRepos([a, b], "size", "asc").map((x) => x.repoId)).toEqual(["aaa/m", "bbb/m"]);
  });

  it("does not mutate its input", () => {
    const input = [r({ sizeBytes: 1 }), r({ sizeBytes: 2 })];
    sortRepos(input, "size", "desc");
    expect(input[0].sizeBytes).toBe(1);
  });
});
