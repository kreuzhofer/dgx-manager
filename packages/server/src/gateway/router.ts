import { Router, type Request, type Response } from "express";
import { prisma } from "../prisma.js";
import { HEALTHY_STATUS } from "../ws/deployment-status.js";
import { toModelList } from "./models.js";
import { selectEligibleMembers } from "./eligibility.js";
import { BodyTooLargeError, forwardToMember, readBodyWithLimit } from "./proxy.js";

/**
 * The inference **gateway**: one OpenAI-compatible surface fronting every
 * running deployment. See docs/adr/0001-inference-gateway.md and CONTEXT.md.
 *
 * The served surface is an **allowlist**, not a path prefix. Anything not
 * listed here is refused by the gateway and never forwarded to a node — so a
 * runtime that gains new endpoints in a future version can never silently widen
 * what the cluster exposes, and a runtime's own management API (pull, delete,
 * push) is unreachable through the gateway by construction.
 *
 * A client cannot tell which runtime serves a name. That is deliberate: it is
 * what lets a model move between runtimes without breaking callers.
 *
 * Mounted ahead of the server's global JSON body parser, whose default size
 * limit would otherwise reject long-context prompts at the manager that succeed
 * when sent to a node directly. The management API keeps that defensive limit.
 */
export const gatewayRouter = Router();

/** OpenAI-shaped error body, so clients can parse failures the way they expect. */
function openAiError(
  res: Response,
  status: number,
  message: string,
  type: string,
  code: string,
): Response {
  return res.status(status).json({ error: { message, type, code } });
}

/**
 * The cluster's model list, synthesized from what the manager deployed —
 * never proxied. Asking a node what it holds would enumerate models nobody
 * deployed, which is precisely what must not be discoverable.
 */
gatewayRouter.get("/models", async (_req: Request, res: Response) => {
  try {
    const deployments = await prisma.deployment.findMany({
      where: { status: HEALTHY_STATUS, publishedName: { not: null } },
      select: { publishedName: true, createdAt: true },
    });
    res.json(toModelList(deployments));
  } catch (err) {
    // Express would otherwise hand this to its default handler, which answers
    // an OpenAI client with an HTML 500 — the one exit that would break the
    // promise that this surface speaks the OpenAI API and nothing else.
    console.error("[gateway] could not read the model list:", err);
    openAiError(
      res,
      503,
      "The gateway could not read the cluster's model list.",
      "api_error",
      "model_list_unavailable",
    );
  }
});

/**
 * The inference operations. Both take the same shape: read the body, find who
 * serves the requested name, forward the exact bytes, stream the answer back.
 */
async function proxyInference(req: Request, res: Response): Promise<void> {
  let body: Buffer;
  try {
    body = await readBodyWithLimit(req);
  } catch (err) {
    if (err instanceof BodyTooLargeError) {
      openAiError(
        res,
        413,
        `Request body exceeds the gateway's ${err.limit}-byte limit.`,
        "invalid_request_error",
        "request_too_large",
      );
      return;
    }
    openAiError(res, 400, "Could not read the request body.", "invalid_request_error", "bad_request");
    return;
  }

  let requested: unknown;
  try {
    requested = (JSON.parse(body.toString("utf8")) as { model?: unknown }).model;
  } catch {
    openAiError(res, 400, "Request body is not valid JSON.", "invalid_request_error", "bad_request");
    return;
  }
  if (typeof requested !== "string" || requested.length === 0) {
    openAiError(
      res,
      400,
      "Missing required field: 'model'.",
      "invalid_request_error",
      "missing_model",
    );
    return;
  }

  // Candidates are the deployments publishing this exact name — a pool. An
  // unknown name never reaches a node: it is refused here.
  const candidates = await prisma.deployment.findMany({
    where: { publishedName: requested },
    select: { id: true, status: true, port: true, node: { select: { id: true, ipAddress: true } } },
  });
  if (candidates.length === 0) {
    openAiError(
      res,
      404,
      `The model '${requested}' does not exist.`,
      "invalid_request_error",
      "model_not_found",
    );
    return;
  }

  const agentHub = req.app.get("agentHub") as { isAgentOnline(nodeId: string): boolean } | undefined;
  const members = selectEligibleMembers(
    candidates.map((c) => ({
      id: c.id,
      status: c.status,
      port: c.port,
      nodeId: c.node.id,
      nodeIp: c.node.ipAddress,
    })),
    (nodeId) => agentHub?.isAgentOnline(nodeId) ?? false,
  );
  if (members.length === 0) {
    openAiError(
      res,
      503,
      `No deployment serving '${requested}' can take a request right now.`,
      "api_error",
      "no_eligible_member",
    );
    return;
  }

  // Choosing BETWEEN eligible members is a separate concern; for now the first
  // eligible member takes the request.
  const target = members[0];

  try {
    await forwardToMember({ baseUrl: target.baseUrl, path: req.originalUrl }, body, req.headers, res);
  } catch (err) {
    console.error(`[gateway] forwarding '${requested}' to ${target.baseUrl} failed:`, err);
    if (res.headersSent) {
      // Mid-stream: the client already has a status and some bytes, so the only
      // honest thing left is to stop rather than append an error to a payload.
      res.end();
      return;
    }
    openAiError(
      res,
      502,
      `The deployment serving '${requested}' could not be reached.`,
      "api_error",
      "upstream_unreachable",
    );
  }
}

gatewayRouter.post("/chat/completions", proxyInference);
gatewayRouter.post("/embeddings", proxyInference);

/**
 * Everything else. Refused here, never forwarded — including every native
 * runtime path and any OpenAI operation the gateway does not serve.
 */
gatewayRouter.use((req: Request, res: Response) =>
  openAiError(
    res,
    404,
    `Unknown or unsupported operation: ${req.method} ${req.originalUrl}. ` +
      `This gateway serves the OpenAI API only.`,
    "invalid_request_error",
    "unknown_operation",
  ),
);
