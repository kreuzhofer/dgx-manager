import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { Readable } from "node:stream";
import type { Request } from "express";
import {
  BodyTooLargeError,
  MAX_BODY_BYTES,
  forwardableHeaders,
  readBodyWithLimit,
} from "./proxy.js";

/** A request-like stream carrying `chunks`. */
function fakeRequest(chunks: (string | Buffer)[]): Request {
  return Readable.from(chunks.map((c) => (typeof c === "string" ? Buffer.from(c) : c))) as unknown as Request;
}

describe("readBodyWithLimit", () => {
  it("returns the body as the exact bytes sent", async () => {
    const body = await readBodyWithLimit(fakeRequest(["{\"mo", "del\":\"m\"}"]));
    expect(body.toString()).toBe('{"model":"m"}');
  });

  it("accepts a body exactly at the limit", async () => {
    const body = await readBodyWithLimit(fakeRequest(["x".repeat(64)]), 64);
    expect(body.byteLength).toBe(64);
  });

  // The failure mode the cap exists for: a body that would otherwise be
  // accumulated in manager memory is refused instead.
  it("refuses a body over the limit", async () => {
    await expect(readBodyWithLimit(fakeRequest(["x".repeat(65)]), 64)).rejects.toBeInstanceOf(
      BodyTooLargeError,
    );
  });

  it("refuses as soon as the limit is passed, not after the whole body arrives", async () => {
    let chunksPulled = 0;
    // A small highWaterMark so the count reflects what we consumed rather than
    // Node's read-ahead buffer, which would otherwise pull ~16KB regardless.
    const stream = new Readable({
      highWaterMark: 100,
      read() {
        chunksPulled += 1;
        // Bounded so a regression pauses the run rather than hanging it.
        if (chunksPulled > 500) return void this.push(null);
        this.push(Buffer.alloc(100, 0x61));
      },
    });
    await expect(readBodyWithLimit(stream as unknown as Request, 150)).rejects.toBeInstanceOf(
      BodyTooLargeError,
    );
    // Two chunks pass 150 bytes; an endless stream must not be drained past that.
    expect(chunksPulled).toBeLessThan(10);
  });

  it("carries the limit on the error so the refusal can say what it was", async () => {
    const err = await readBodyWithLimit(fakeRequest(["xx"]), 1).catch((e) => e);
    expect(err).toBeInstanceOf(BodyTooLargeError);
    expect((err as BodyTooLargeError).limit).toBe(1);
  });

  it("propagates a stream error", async () => {
    const stream = new Readable({ read() { this.destroy(new Error("ECONNRESET")); } });
    await expect(readBodyWithLimit(stream as unknown as Request)).rejects.toThrow("ECONNRESET");
  });

  it("defaults to a limit far above a long-context prompt", () => {
    expect(MAX_BODY_BYTES).toBeGreaterThan(8 * 1024 * 1024);
  });

  /**
   * Invariant: whatever the chunking, the collected body is the concatenation
   * of what was sent — a proxy that reassembled bytes wrongly would corrupt
   * every prompt that arrived in more than one packet.
   */
  test.prop([fc.array(fc.string(), { maxLength: 20 })])(
    "reassembles any chunking into the original bytes",
    async (parts) => {
      const body = await readBodyWithLimit(fakeRequest(parts.length ? parts : [""]));
      expect(body.toString()).toBe(parts.join(""));
    },
  );
});

describe("forwardableHeaders", () => {
  // A client's key is meaningless to a node, and relaying a credential onward
  // is worse than ignoring it.
  it("drops credentials and connection-scoped headers", () => {
    const out = forwardableHeaders({
      authorization: "Bearer sk-secret",
      cookie: "session=abc",
      "proxy-authorization": "Basic xyz",
      host: "manager:4000",
      "content-length": "12",
      connection: "keep-alive",
      "transfer-encoding": "chunked",
      "content-type": "application/json",
      accept: "text/event-stream",
    });
    expect(out).toEqual({ "content-type": "application/json", accept: "text/event-stream" });
  });

  it("joins repeated headers rather than dropping them", () => {
    expect(forwardableHeaders({ "x-thing": ["a", "b"] })["x-thing"]).toBe("a, b");
  });

  it("ignores headers with no value", () => {
    expect(forwardableHeaders({ "x-absent": undefined })).toEqual({});
  });

  /**
   * Invariant: no header the gateway forwards may ever be one of the
   * never-forward set, whatever casing a client uses to smuggle it.
   */
  test.prop([
    fc.dictionary(
      fc.constantFrom("Authorization", "AUTHORIZATION", "Cookie", "HOST", "x-safe", "Accept"),
      fc.string({ minLength: 1 }),
    ),
  ])("never forwards a credential header, whatever its casing", (headers) => {
    const out = forwardableHeaders(headers);
    for (const key of Object.keys(out)) {
      expect(["authorization", "cookie", "host"]).not.toContain(key.toLowerCase());
    }
  });
});
