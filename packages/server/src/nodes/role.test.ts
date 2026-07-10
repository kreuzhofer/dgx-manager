import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  isEvalNode,
  runtimeAllowedOnNode,
  evalNodeRejectionMessage,
  RUNTIMES_ALLOWED_ON_EVAL,
} from "./role.js";

describe("isEvalNode", () => {
  it("recognises the eval role", () => {
    expect(isEvalNode("eval")).toBe(true);
  });

  // A legacy row predating the column, or an unknown value, must never be
  // mistaken for an eval node — that would silently exclude a real GPU node
  // from the deploy picker.
  it.each([["gpu"], [null], [undefined], [""], ["EVAL"], ["evaluator"]])(
    "treats %j as a normal node",
    (role) => {
      expect(isEvalNode(role as string | null | undefined)).toBe(false);
    },
  );
});

describe("runtimeAllowedOnNode", () => {
  it("permits ollama on an eval node", () => {
    expect(runtimeAllowedOnNode("eval", "ollama")).toBe(true);
  });

  it.each([["vllm"], ["dgxrun"], ["sglang"], [""], ["VLLM"]])(
    "refuses %j on an eval node",
    (runtime) => {
      expect(runtimeAllowedOnNode("eval", runtime)).toBe(false);
    },
  );

  // A gpu node hosts anything; the role only ever *restricts*.
  test.prop([fc.string()])("permits every runtime on a gpu node", (runtime) => {
    expect(runtimeAllowedOnNode("gpu", runtime)).toBe(true);
  });

  /** Invariant: the only role that restricts anything is "eval". */
  test.prop([fc.string().filter((r) => r !== "eval"), fc.string()])(
    "no role other than eval restricts any runtime",
    (role, runtime) => {
      expect(runtimeAllowedOnNode(role, runtime)).toBe(true);
    },
  );

  /** Invariant: on an eval node, allowed === membership of the allow-list. */
  test.prop([fc.string()])("eval allows exactly the allow-list", (runtime) => {
    expect(runtimeAllowedOnNode("eval", runtime)).toBe(
      RUNTIMES_ALLOWED_ON_EVAL.includes(runtime),
    );
  });
});

describe("evalNodeRejectionMessage", () => {
  it("names the node and the runtime, and says what is allowed", () => {
    const m = evalNodeRejectionMessage("agenthost", "vllm");
    expect(m).toContain("agenthost");
    expect(m).toContain("vllm");
    expect(m).toContain("ollama");
  });
});
