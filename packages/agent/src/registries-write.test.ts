import { describe, it, expect, beforeEach } from "vitest";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parse } from "yaml";
import { writeRegistriesFile, registriesConfigPath } from "./registries.js";

describe("writeRegistriesFile", () => {
  beforeEach(() => {
    process.env.HOME = mkdtempSync(join(tmpdir(), "agent-home-"));
  });

  it("writes <HOME>/.config/sparkrun/registries.yaml atomically and parseably", () => {
    writeRegistriesFile([
      { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" },
    ]);
    const path = registriesConfigPath();
    expect(path).toBe(join(process.env.HOME!, ".config", "sparkrun", "registries.yaml"));
    const parsed = parse(readFileSync(path, "utf8"));
    expect(parsed.registries[0].name).toBe("rtx");
  });

  it("creates intermediate directories if they do not exist", () => {
    writeRegistriesFile([
      { name: "myrepo", url: "https://github.com/example/repo.git", subpath: "recipes" },
    ]);
    const path = registriesConfigPath();
    const parsed = parse(readFileSync(path, "utf8"));
    expect(parsed.registries[0].name).toBe("myrepo");
  });

  it("writes an empty registries list", () => {
    writeRegistriesFile([]);
    const path = registriesConfigPath();
    const parsed = parse(readFileSync(path, "utf8"));
    expect(parsed.registries).toEqual([]);
  });
});
