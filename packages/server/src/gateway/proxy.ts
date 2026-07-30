import http from "node:http";
import type { Request, Response } from "express";

/**
 * The forwarding hop.
 *
 * Built on the Node HTTP client rather than fetch, for two concrete reasons:
 *
 *   - fetch (undici) applies a headers timeout by default, which a long prefill
 *     can exceed — a 300K-token prompt surfaces as an abrupt socket error
 *     instead of a response. Nothing here imposes a timeout; the upstream
 *     decides how long it needs.
 *   - the request body is forwarded as the exact bytes the client sent, never
 *     re-serialized. The gateway parses it only to read `model`.
 *
 * The gateway is mounted ahead of the server's global JSON parser, so the body
 * arrives here unread. It is buffered — the routing key lives in it — under an
 * explicit cap, so an oversized request fails loudly rather than consuming
 * manager memory.
 */

/** Bodies above this are refused. Well past a 300K-token prompt (~1.2 MB). */
export const MAX_BODY_BYTES = 64 * 1024 * 1024;

export class BodyTooLargeError extends Error {
  constructor(readonly limit: number) {
    super(`Request body exceeds ${limit} bytes`);
    this.name = "BodyTooLargeError";
  }
}

/**
 * Collect the raw request body, refusing anything over `limit`.
 *
 * Rejects as soon as the limit is passed rather than after the whole body has
 * arrived, so an oversized request is cheap to refuse.
 */
export function readBodyWithLimit(req: Request, limit = MAX_BODY_BYTES): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;

    const fail = (err: Error) => {
      if (settled) return;
      settled = true;
      req.removeAllListeners("data");
      req.removeAllListeners("end");
      req.removeAllListeners("error");
      reject(err);
    };

    req.on("data", (chunk: Buffer) => {
      if (settled) return;
      size += chunk.length;
      if (size > limit) {
        fail(new BodyTooLargeError(limit));
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks));
    });
    req.on("error", fail);
  });
}

/**
 * Headers a client sends that must not reach a node.
 *
 * `authorization` is dropped deliberately: clients always send some key, the
 * gateway does not authenticate, and forwarding a client's credential to a node
 * is worse than ignoring it. `host` and `content-length` are recomputed for the
 * upstream request; hop-by-hop headers do not survive a hop by definition.
 */
const HEADERS_NEVER_FORWARDED = new Set([
  "authorization",
  "cookie",
  "host",
  "content-length",
  "connection",
  "keep-alive",
  "transfer-encoding",
  "upgrade",
  "proxy-authorization",
]);

export function forwardableHeaders(headers: http.IncomingHttpHeaders): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (value === undefined) continue;
    if (HEADERS_NEVER_FORWARDED.has(key.toLowerCase())) continue;
    out[key] = Array.isArray(value) ? value.join(", ") : value;
  }
  return out;
}

export interface ForwardTarget {
  /** `http://<ip>:<port>` of the chosen member. */
  baseUrl: string;
  /** OpenAI path to append, e.g. `/v1/chat/completions`. */
  path: string;
}

/**
 * Forward `body` to the target and stream the response straight back.
 *
 * Streaming matters: a chat completion with `stream: true` is a stream of SSE
 * events, and buffering it to completion would turn an incremental UI into a
 * long pause. Response bytes are piped, not collected.
 *
 * Resolves when the response has been fully relayed; rejects if the upstream
 * could not be reached, leaving the caller to shape the error.
 */
export function forwardToMember(
  target: ForwardTarget,
  body: Buffer,
  clientHeaders: http.IncomingHttpHeaders,
  res: Response,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = new URL(target.path, target.baseUrl);

    const upstream = http.request(
      {
        hostname: url.hostname,
        port: url.port,
        path: url.pathname + url.search,
        method: "POST",
        headers: {
          ...forwardableHeaders(clientHeaders),
          "content-length": String(body.byteLength),
        },
      },
      (upstreamRes) => {
        res.status(upstreamRes.statusCode ?? 502);
        for (const [key, value] of Object.entries(upstreamRes.headers)) {
          if (value === undefined) continue;
          if (key.toLowerCase() === "transfer-encoding") continue;
          res.setHeader(key, value as string | string[]);
        }

        upstreamRes.pipe(res);
        upstreamRes.on("end", resolve);
        upstreamRes.on("error", reject);
      },
    );

    // No timeout is set anywhere on purpose: a long-context prefill can take
    // minutes before the first byte, and cutting it off would cap context
    // length by accident.
    upstream.on("error", reject);

    // If the client hangs up, stop asking the node to keep working.
    res.on("close", () => {
      if (!res.writableEnded) upstream.destroy();
    });

    upstream.end(body);
  });
}
