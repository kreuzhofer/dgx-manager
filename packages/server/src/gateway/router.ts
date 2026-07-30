import { Router, type Request, type Response } from "express";
import { prisma } from "../prisma.js";
import { HEALTHY_STATUS } from "../ws/deployment-status.js";
import { toModelList } from "./models.js";

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
