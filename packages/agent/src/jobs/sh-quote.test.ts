import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { execFileSync } from "child_process";
import { shQuote } from "./sh-quote.js";

describe("shQuote", () => {
  it("wraps a plain word", () => {
    expect(shQuote("hello")).toBe("'hello'");
  });

  it("escapes an embedded single quote", () => {
    expect(shQuote("it's")).toBe("'it'\\''s'");
  });

  /**
   * Invariant: a quoted string, passed through `sh -c`, comes back byte-for-byte.
   * This is the property that matters — the wrapper script interpolates argv, and
   * an escaping bug there is a shell-injection bug.
   */
  test.prop([fc.string({ minLength: 1 }).filter((s) => !s.includes("\0"))])(
    "round-trips through sh",
    (s) => {
      const out = execFileSync("sh", ["-c", `printf %s ${shQuote(s)}`], { encoding: "utf8" });
      expect(out).toBe(s);
    },
  );

  /** Invariant: shell metacharacters can never escape the quoting. */
  test.prop([fc.constantFrom(";", "&&", "|", "$(id)", "`id`", "\n", ">out")])(
    "neutralises metacharacters",
    (evil) => {
      const out = execFileSync("sh", ["-c", `printf %s ${shQuote(evil)}`], { encoding: "utf8" });
      expect(out).toBe(evil);
    },
  );
});
