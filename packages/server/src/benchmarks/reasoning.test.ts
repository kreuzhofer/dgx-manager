import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { stripReasoning } from "./reasoning.js";

describe("stripReasoning", () => {
  it("removes a complete <think>…</think> block and keeps the trailing answer", () => {
    expect(stripReasoning("<think>lots of thinking</think>The answer is B")).toBe("The answer is B");
  });

  it("handles a template-injected open tag (only the closing tag is echoed)", () => {
    expect(stripReasoning("reasoning text</think>\n\nThe answer is 42")).toBe("The answer is 42");
  });

  it("yields empty output when the model was cut off mid-think (no close tag)", () => {
    expect(stripReasoning("<think>never finished reasoning")).toBe("");
  });

  it("is a no-op (modulo trim) when there are no reasoning tags", () => {
    expect(stripReasoning("The answer is 7")).toBe("The answer is 7");
  });

  it("is case-insensitive on the tags", () => {
    expect(stripReasoning("<THINK>x</THINK>done")).toBe("done");
  });

  // Invariant: output never contains a think tag, for any input.
  test.prop([fc.string()])("output never contains a think tag", (s) => {
    const out = stripReasoning(s).toLowerCase();
    expect(out.includes("<think>")).toBe(false);
    expect(out.includes("</think>")).toBe(false);
  });

  // Invariant: idempotent.
  test.prop([fc.string()])("is idempotent", (s) => {
    expect(stripReasoning(stripReasoning(s))).toBe(stripReasoning(s));
  });

  // Invariant: a think block prepended to a tag-free answer is fully removed.
  test.prop([
    fc.stringMatching(/^[a-zA-Z0-9 .,!?]+$/),
    fc.stringMatching(/^[a-zA-Z0-9 .,!?]+$/),
  ])("strips a wrapped think block down to the answer", (noise, answer) => {
    expect(stripReasoning(`<think>${noise}</think>${answer}`)).toBe(answer.trim());
  });
});
