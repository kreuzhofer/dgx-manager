export type EndpointDeployment = {
  port: number | null;
  node: { ipAddress: string | null } | null;
};

export function deploymentEndpointUrl(d: EndpointDeployment): string {
  if (!d.node) throw new Error("deployment.node is required");
  if (!d.node.ipAddress) throw new Error("deployment.node.ipAddress is required");
  if (!d.port) throw new Error("deployment.port is required");
  return `http://${d.node.ipAddress}:${d.port}`;
}

/**
 * Parse the first served model id out of an OpenAI `/v1/models` response body.
 * vLLM serves the model under the recipe's `--served-model-name`, which is not
 * necessarily the deployment's displayName or the catalog model name — so the
 * benchmark must ask the running endpoint what it actually serves. Returns null
 * for a missing/empty/malformed body so the caller can fall back.
 */
export function firstServedModelId(body: string): string | null {
  try {
    const parsed = JSON.parse(body) as { data?: Array<{ id?: unknown }> };
    const id = parsed?.data?.[0]?.id;
    return typeof id === "string" && id.length > 0 ? id : null;
  } catch {
    return null;
  }
}

/**
 * Resolve the model name to pass to llama-benchy / tool-eval-bench: query the
 * live deployment's `{endpointUrl}/models` and use vLLM's actual served id.
 * Falls back to `fallback` if the endpoint is unreachable or returns nothing —
 * so a not-yet-ready deployment still produces a run (that will surface the real
 * error) rather than a silent mismatch 404.
 */
export async function resolveServedModelName(
  endpointUrl: string,
  fallback: string,
  fetchImpl: typeof fetch = fetch,
): Promise<string> {
  try {
    const res = await fetchImpl(`${endpointUrl}/models`, {
      signal: AbortSignal.timeout(3000),
    });
    if (res.ok) {
      const id = firstServedModelId(await res.text());
      if (id) return id;
    }
  } catch {
    // endpoint not ready / unreachable — fall back below
  }
  return fallback;
}
