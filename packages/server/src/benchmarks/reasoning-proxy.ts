import http from "node:http";
import { AddressInfo } from "node:net";
import { stripReasoning } from "./reasoning.js";

/**
 * Disable every inbound timeout on a proxy server.
 *
 * Node applies a 300 s `requestTimeout` and a 60 s `headersTimeout` by default.
 * Both are wrong here: a non-streaming completion emits nothing until generation
 * finishes, so "time to first byte" is the entire generation, and a slow
 * accuracy item legitimately exceeds five minutes. A GPQA run against Muse
 * Glimmer died at 126/198 when these fired on the tail — the client saw
 * `ServerDisconnectedError` and its connector collapsed, taking the run with it.
 *
 * There is no safe non-zero value: the correct bound is the upstream's, not
 * ours. gateway/proxy.ts takes the same position for the same reason.
 */
export function applyNoTimeouts(server: http.Server): void {
  server.requestTimeout = 0;
  server.headersTimeout = 0;
  server.setTimeout(0);
}

/** Forward a request with no timeout of any kind, collecting the full response. */
function forwardNoTimeout(
  targetUrl: string,
  method: string,
  headers: Record<string, string>,
  body: Buffer,
): Promise<{ status: number; contentType: string | null; text: string }> {
  return new Promise((resolve, reject) => {
    const u = new URL(targetUrl);
    const send = method === "GET" || method === "HEAD" ? undefined : body;
    const req = http.request(
      {
        hostname: u.hostname,
        port: u.port,
        path: `${u.pathname}${u.search}`,
        method,
        headers: send ? { ...headers, "content-length": String(send.length) } : headers,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on("data", (c: Buffer) => chunks.push(c));
        res.on("end", () =>
          resolve({
            status: res.statusCode ?? 502,
            contentType: res.headers["content-type"] ?? null,
            text: Buffer.concat(chunks).toString(),
          }),
        );
        res.on("error", reject);
      },
    );
    // Deliberately no req.setTimeout(): undici's fetch defaults are exactly what
    // broke this, and node:http imposes none unless asked.
    req.on("error", reject);
    if (send) req.write(send);
    req.end();
  });
}

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
// `advertiseHost`: when set (remote eval runs), bind on all interfaces and hand
// the runner the manager's LAN IP — a job on the eval node can't reach the
// manager's 127.0.0.1. Omit for local runs (loopback only, unchanged). The proxy
// only strips <think> and forwards to the (already LAN-exposed) model endpoint,
// so 0.0.0.0 on the internal fabric adds no new exposure.
export function startReasoningProxy(
  targetV1Url: string,
  advertiseHost?: string,
): Promise<ReasoningProxy> {
  const bindHost = advertiseHost ? "0.0.0.0" : "127.0.0.1";
  const urlHost = advertiseHost || "127.0.0.1";
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

        const upstream = await forwardNoTimeout(targetUrl, req.method ?? "GET", headers, body);

        const isChat = suffix.includes("/chat/completions");
        const out = isChat ? rewriteChatBody(upstream.text) : upstream.text;

        res.statusCode = upstream.status;
        if (upstream.contentType) res.setHeader("content-type", upstream.contentType);
        res.end(out);
      } catch (e) {
        res.statusCode = 502;
        res.end(JSON.stringify({ error: `reasoning-proxy: ${(e as Error).message}` }));
      }
    });

    applyNoTimeouts(server);

    server.listen(0, bindHost, () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        url: `http://${urlHost}:${port}/v1`,
        close: () => new Promise((r) => {
          server.close(() => r());
          server.closeAllConnections?.();
        }),
      });
    });
  });
}
