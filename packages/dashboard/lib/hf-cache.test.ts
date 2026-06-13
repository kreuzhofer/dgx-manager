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
  it("size: biggest first", () => {
    const sorted = sortRepos([r({ repoId: "a", sizeBytes: 1 }), r({ repoId: "b", sizeBytes: 9 })], "size");
    expect(sorted.map((x) => x.repoId)).toEqual(["b", "a"]);
  });

  it("lastDeployed: never-deployed first, then oldest deployment first", () => {
    const sorted = sortRepos(
      [
        r({ repoId: "recent", lastDeployedAt: "2026-06-10T00:00:00.000Z" }),
        r({ repoId: "never", lastDeployedAt: null }),
        r({ repoId: "old", lastDeployedAt: "2026-01-01T00:00:00.000Z" }),
      ],
      "lastDeployed",
    );
    expect(sorted.map((x) => x.repoId)).toEqual(["never", "old", "recent"]);
  });

  it("does not mutate its input", () => {
    const input = [r({ sizeBytes: 1 }), r({ sizeBytes: 2 })];
    sortRepos(input, "size");
    expect(input[0].sizeBytes).toBe(1);
  });
});
