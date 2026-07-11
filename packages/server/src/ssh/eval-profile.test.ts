import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { checksForRole, evalSudoCheck, EVAL_NODE_CHECK_NAMES } from "./eval-profile.js";

const items = [
  { name: "NVIDIA Drivers" }, { name: "Docker" }, { name: "Docker group" },
  { name: "uv (uvx)" }, { name: "nvidia-container-toolkit" }, { name: "Node.js" },
  { name: "Ollama" }, { name: "NFS /mnt/tank" }, { name: "sparkrun" },
];

describe("checksForRole", () => {
  it("keeps only Node.js + uv for an eval node", () => {
    expect(checksForRole(items, "eval").map((c) => c.name).sort()).toEqual(["Node.js", "uv (uvx)"]);
  });

  it("excludes the model-hosting stack from an eval node", () => {
    const names = checksForRole(items, "eval").map((c) => c.name);
    for (const forbidden of ["Docker", "nvidia-container-toolkit", "sparkrun", "Ollama", "NVIDIA Drivers"]) {
      expect(names).not.toContain(forbidden);
    }
  });

  it.each([["gpu"], [null], [undefined], [""], ["EVAL"], ["evaluator"]])(
    "leaves the full list untouched for a non-eval role %j",
    (role) => {
      expect(checksForRole(items, role as string | null | undefined)).toEqual(items);
    },
  );

  /** Invariant: eval is a strict subset; every kept item is in the allow-list. */
  test.prop([fc.string()])("eval output is always a subset of the allow-list", (role) => {
    const out = checksForRole(items, role).map((c) => c.name);
    if (role === "eval") {
      for (const n of out) expect(EVAL_NODE_CHECK_NAMES).toContain(n);
    } else {
      expect(out).toEqual(items.map((c) => c.name));
    }
  });
});

describe("evalSudoCheck", () => {
  it("is red when passwordless sudo is absent (job.* needs sudo -n)", () => {
    const c = evalSudoCheck(false);
    expect(c.status).toBe("red");
    expect(c.detail.toLowerCase()).toContain("sudo");
  });
  it("is green when available", () => {
    expect(evalSudoCheck(true).status).toBe("green");
  });
});
