import { describe, it, expect, vi } from "vitest";
import { CapRegistry } from "./registry.js";

const noopCtx = { emitChunk: () => {} };

describe("CapRegistry", () => {
  it("dispatches to a registered capability", async () => {
    const r = new CapRegistry();
    r.register({ name: "echo", handle: async (i) => ({ got: i }) });
    expect(await r.dispatch("echo", 42, noopCtx)).toEqual({ ok: true, data: { got: 42 } });
  });
  it("unknown capability -> ok:false", async () => {
    expect(await new CapRegistry().dispatch("nope", null, noopCtx))
      .toEqual({ ok: false, error: "unknown capability: nope" });
  });
  it("handler throw -> ok:false with message", async () => {
    const r = new CapRegistry();
    r.register({ name: "boom", handle: async () => { throw new Error("kaboom"); } });
    expect(await r.dispatch("boom", null, noopCtx)).toEqual({ ok: false, error: "kaboom" });
  });
});
