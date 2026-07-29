import { resolveServedModelName } from "../benchmarks/endpoint.js";

/**
 * A deployment's **published name** — the name a client puts in the `model`
 * field to reach it.
 *
 * The name is never rewritten in flight, so it must be a name the upstream
 * runtime itself answers to. That cannot be assumed, and how it is discovered
 * depends on the kind of runtime:
 *
 *   - A **pinned runtime** (vLLM / sparkrun) serves one resident model and is
 *     authoritative about its name. A recipe's `served_model_name` may be
 *     neither the deployment's displayName nor the catalog model name, so we
 *     ask the endpoint. Guessing wrong produces an unexplained 404 for the
 *     caller — the failure this resolution exists to prevent.
 *   - An **allocation-inducing runtime** (Ollama) hosts every model pulled on
 *     that node behind one shared address. Its model list describes the node,
 *     not this deployment, so probing would return an arbitrary sibling. The
 *     deployment's model tag is the name the runtime answers to.
 *
 * See CONTEXT.md ("Published name") and docs/adr/0001-inference-gateway.md.
 */
export interface PublishedNameInput {
  /** The catalog model's runtime, e.g. `vllm` | `ollama`. */
  runtime: string | null;
  /** Catalog model name. For an allocation-inducing runtime this is the tag. */
  modelName: string;
  /** Operator-chosen alias, passed to a pinned runtime as its served name. */
  displayName: string | null;
  /** `http://<ip>:<port>` of the running deployment, or null when not serving. */
  endpointUrl: string | null;
}

/**
 * Runtimes an unauthenticated caller can induce into allocating — they load a
 * model on demand. Kept as a set so a future on-demand runtime inherits the
 * behaviour by being listed here rather than by a scattered `=== "ollama"`.
 */
const ALLOCATION_INDUCING_RUNTIMES = new Set(["ollama"]);

export function isAllocationInducingRuntime(runtime: string | null): boolean {
  return ALLOCATION_INDUCING_RUNTIMES.has((runtime ?? "").toLowerCase());
}

/**
 * Resolve the published name for a deployment that has started serving.
 *
 * `fetchImpl` is injected for tests, matching the shape `resolveServedModelName`
 * already uses. It is never called for an allocation-inducing runtime, nor when
 * the deployment has no endpoint yet.
 */
export async function resolvePublishedName(
  input: PublishedNameInput,
  fetchImpl: typeof fetch = fetch,
): Promise<string> {
  if (isAllocationInducingRuntime(input.runtime)) return input.modelName;

  const fallback = input.displayName ?? input.modelName;
  if (!input.endpointUrl) return fallback;

  return resolveServedModelName(`${input.endpointUrl}/v1`, fallback, fetchImpl);
}
