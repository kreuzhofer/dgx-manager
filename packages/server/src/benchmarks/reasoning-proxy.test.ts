import { describe, expect, it } from "vitest";
import http from "node:http";
import { startReasoningProxy } from "./reasoning-proxy.js";

// Spin a fake upstream that echoes a canned response for a given path.
function fakeUpstream(handler: (req: http.IncomingMessage, res: http.ServerResponse) => void) {
  return new Promise<{ v1Url: string; close: () => Promise<void> }>((resolve) => {
    const server = http.createServer(handler);
    server.listen(0, "127.0.0.1", () => {
      const addr = server.address() as { port: number };
      resolve({
        v1Url: `http://127.0.0.1:${addr.port}/v1`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}

describe("startReasoningProxy", () => {
  it("strips <think>…</think> from chat-completion responses", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({
        choices: [{ message: { role: "assistant", content: "<think>secret</think>Answer: B" } }],
      }));
    });
    const proxy = await startReasoningProxy(upstream.v1Url);

    const resp = await fetch(`${proxy.url}/chat/completions`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ model: "m", messages: [] }),
    });
    const body = await resp.json();
    expect(body.choices[0].message.content).toBe("Answer: B");

    await proxy.close();
    await upstream.close();
  });

  it("passes non-chat responses (e.g. /v1/models) through unchanged", async () => {
    const upstream = await fakeUpstream((_req, res) => {
      res.setHeader("content-type", "application/json");
      res.end(JSON.stringify({ data: [{ id: "served-model" }] }));
    });
    const proxy = await startReasoningProxy(upstream.v1Url);

    const resp = await fetch(`${proxy.url}/models`);
    const body = await resp.json();
    expect(body.data[0].id).toBe("served-model");

    await proxy.close();
    await upstream.close();
  });
});
