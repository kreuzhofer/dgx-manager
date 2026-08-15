import { describe, it, expect, afterEach } from "vitest";
import http from "node:http";
import { startReasoningProxy, applyNoTimeouts } from "./reasoning-proxy.js";

/**
 * The reasoning proxy sits between lm-eval and a vLLM endpoint for the whole of a
 * multi-hour accuracy run, so every default timeout in the path is a live grenade.
 *
 * A GPQA-Diamond run against Muse Glimmer died at 126/198 after 2h28m because BOTH
 * of Node's 300 s defaults fired on the slow tail (mean 70.6 s/item):
 *
 *   inbound   server.requestTimeout = 300_000  -> client saw ServerDisconnectedError
 *   outbound  undici fetch headersTimeout      -> "reasoning-proxy: fetch failed" 502
 *
 * A non-streaming completion sends no bytes at all until generation finishes, so
 * "time to first byte" is the whole generation. There is no safe default here —
 * the only correct value is "no timeout", which is what gateway/proxy.ts already
 * does deliberately.
 */

const servers: http.Server[] = [];
const closers: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.all(closers.splice(0).map((c) => c()));
  await Promise.all(servers.splice(0).map((s) => new Promise<void>((r) => s.close(() => r()))));
});

/** A stub upstream that waits `delayMs` before replying — stands in for a long generation. */
function stubUpstream(delayMs: number, payload: unknown): Promise<string> {
  return new Promise((resolve) => {
    const s = http.createServer((_req, res) => {
      setTimeout(() => {
        res.setHeader("content-type", "application/json");
        res.end(JSON.stringify(payload));
      }, delayMs);
    });
    servers.push(s);
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address() as { port: number };
      resolve(`http://127.0.0.1:${port}/v1`);
    });
  });
}

describe("applyNoTimeouts", () => {
  // Node's defaults are the bug. Assert the disabling explicitly rather than
  // waiting 300 s to observe it, so a future Node upgrade that reintroduces a
  // default fails this test instead of a benchmark two hours in.
  it("disables every inbound timeout that could cut a long generation", () => {
    const s = http.createServer(() => {});
    servers.push(s);
    expect(s.requestTimeout).toBeGreaterThan(0); // default is live before the fix
    applyNoTimeouts(s);
    expect(s.requestTimeout).toBe(0);
    expect(s.headersTimeout).toBe(0);
    expect(s.timeout).toBe(0);
  });
});

describe("reasoning proxy forwarding", () => {
  it("strips reasoning from chat completions and preserves status", async () => {
    const upstream = await stubUpstream(0, {
      choices: [{ message: { content: "<think>hidden working</think>The answer is (C)." } }],
    });
    const proxy = await startReasoningProxy(upstream);
    closers.push(proxy.close);

    const r = await fetch(`${proxy.url}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [] }),
    });
    expect(r.status).toBe(200);
    const body = await r.json() as { choices: Array<{ message: { content: string } }> };
    expect(body.choices[0].message.content).toBe("The answer is (C).");
  });

  // The regression proper: an upstream slower than the proxy would previously
  // tolerate must still come back 200. The delay here is short so the suite stays
  // fast — the timeouts it guards are asserted directly above.
  it("survives an upstream that replies slowly", async () => {
    const upstream = await stubUpstream(1200, { choices: [{ message: { content: "ok" } }] });
    const proxy = await startReasoningProxy(upstream);
    closers.push(proxy.close);

    const r = await fetch(`${proxy.url}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [] }),
    });
    expect(r.status).toBe(200);
    expect((await r.json() as any).choices[0].message.content).toBe("ok");
  });

  it("passes non-chat paths through untouched", async () => {
    const upstream = await stubUpstream(0, { object: "list", data: [{ id: "served-model" }] });
    const proxy = await startReasoningProxy(upstream);
    closers.push(proxy.close);

    const r = await fetch(`${proxy.url}/models`);
    expect(r.status).toBe(200);
    expect((await r.json() as any).data[0].id).toBe("served-model");
  });
});

// Binding behaviour — a remote eval node cannot reach the manager's loopback, so
// the advertised host is what makes a remote run work at all (see 1b8be53).
describe("startReasoningProxy advertiseHost", () => {
  it("uses loopback by default (local runs)", async () => {
    const proxy = await startReasoningProxy("http://example.invalid/v1");
    closers.push(proxy.close);
    expect(proxy.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v1$/);
  });

  it("advertises the given host so a remote eval node can reach it", async () => {
    const proxy = await startReasoningProxy("http://example.invalid/v1", "192.168.44.14");
    closers.push(proxy.close);
    expect(proxy.url).toMatch(/^http:\/\/192\.168\.44\.14:\d+\/v1$/);
  });
});
