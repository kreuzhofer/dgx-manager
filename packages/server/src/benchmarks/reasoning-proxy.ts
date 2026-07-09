import http from "node:http";
import { AddressInfo } from "node:net";
import { stripReasoning } from "./reasoning.js";

export type ReasoningProxy = {
  url: string;             // .../v1 base to hand to lm-eval
  close: () => Promise<void>;
};

function readBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(c));
    req.on("end", () => resolve(Buffer.concat(chunks)));
    req.on("error", reject);
  });
}

// Rewrite a chat-completion JSON body, stripping reasoning from each choice's
// message content. Returns the original text unchanged if it isn't the expected
// shape (so /v1/models and errors pass through untouched).
function rewriteChatBody(text: string): string {
  try {
    const parsed = JSON.parse(text) as {
      choices?: Array<{ message?: { content?: unknown } }>;
    };
    if (!Array.isArray(parsed.choices)) return text;
    for (const c of parsed.choices) {
      if (c.message && typeof c.message.content === "string") {
        c.message.content = stripReasoning(c.message.content);
      }
    }
    return JSON.stringify(parsed);
  } catch {
    return text;
  }
}

// Localhost proxy in front of `targetV1Url` (a .../v1 base) that strips
// <think>…</think> from /v1/chat/completions responses before returning them, so
// lm-eval scores the final answer. Non-streaming only (lm-eval uses
// non-streaming completions).
export function startReasoningProxy(targetV1Url: string): Promise<ReasoningProxy> {
  return new Promise((resolve) => {
    const server = http.createServer(async (req, res) => {
      try {
        const suffix = (req.url ?? "").replace(/^\/v1/, "");
        const targetUrl = `${targetV1Url}${suffix}`;
        const body = await readBody(req);

        const headers: Record<string, string> = {};
        for (const [k, v] of Object.entries(req.headers)) {
          if (typeof v === "string" && k.toLowerCase() !== "host" && k.toLowerCase() !== "content-length") {
            headers[k] = v;
          }
        }

        const upstream = await fetch(targetUrl, {
          method: req.method,
          headers,
          body: req.method === "GET" || req.method === "HEAD" ? undefined : body,
        });

        const text = await upstream.text();
        const isChat = suffix.includes("/chat/completions");
        const out = isChat ? rewriteChatBody(text) : text;

        res.statusCode = upstream.status;
        const ct = upstream.headers.get("content-type");
        if (ct) res.setHeader("content-type", ct);
        res.end(out);
      } catch (e) {
        res.statusCode = 502;
        res.end(JSON.stringify({ error: `reasoning-proxy: ${(e as Error).message}` }));
      }
    });

    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        url: `http://127.0.0.1:${port}/v1`,
        close: () => new Promise((r) => server.close(() => r())),
      });
    });
  });
}
