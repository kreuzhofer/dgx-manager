import { afterEach, describe, expect, it } from "vitest";
import { acquire, outstandingFor, outstandingSnapshot, resetOutstanding } from "./inflight.js";

afterEach(resetOutstanding);

describe("in-flight accounting", () => {
  it("counts nothing before anything starts", () => {
    expect(outstandingFor("d1")).toBe(0);
  });

  it("counts concurrent requests to one deployment", () => {
    acquire("d1");
    acquire("d1");
    expect(outstandingFor("d1")).toBe(2);
  });

  it("counts each deployment separately", () => {
    acquire("d1");
    expect(outstandingFor("d2")).toBe(0);
  });

  it("returns to zero once released", () => {
    const release = acquire("d1");
    release();
    expect(outstandingFor("d1")).toBe(0);
  });

  // A caller may release from a success path and again from a finally; that
  // must not push the count below what is actually in flight, which would make
  // a busy member look idle.
  it("ignores a repeated release", () => {
    const first = acquire("d1");
    acquire("d1");
    first();
    first();
    first();
    expect(outstandingFor("d1")).toBe(1);
  });

  it("drops an idle deployment from the snapshot rather than keeping a zero", () => {
    const release = acquire("d1");
    expect(outstandingSnapshot()).toEqual({ d1: 1 });
    release();
    expect(outstandingSnapshot()).toEqual({});
  });

  it("reports every busy deployment in the snapshot", () => {
    acquire("d1");
    acquire("d2");
    acquire("d2");
    expect(outstandingSnapshot()).toEqual({ d1: 1, d2: 2 });
  });
});
