/**
 * A node's role decides what may be *deployed* onto it. It never affects
 * metrics, provisioning, or agent management.
 *
 * `eval` exists for agenthost: a benchmark runner with no CUDA VRAM. It may
 * serve small Ollama models (embeddings) but must never host a vLLM/dgxrun
 * deployment. VRAM and arch admission would usually reject those anyway, but
 * relying on that is accidental, not intentional — this makes it explicit.
 */
export type NodeRole = "gpu" | "eval";

/** Runtimes an `eval` node is permitted to host. Everything else is refused. */
export const RUNTIMES_ALLOWED_ON_EVAL: readonly string[] = ["ollama"];

/** Only the exact string "eval" restricts a node. Legacy rows have no role. */
export function isEvalNode(role: string | null | undefined): boolean {
  return role === "eval";
}

/** True when `runtime` may be deployed onto a node with this `role`. */
export function runtimeAllowedOnNode(
  role: string | null | undefined,
  runtime: string,
): boolean {
  if (!isEvalNode(role)) return true;
  return RUNTIMES_ALLOWED_ON_EVAL.includes(runtime);
}

export function evalNodeRejectionMessage(nodeName: string, runtime: string): string {
  return (
    `Node "${nodeName}" has role "eval" and cannot host a "${runtime}" deployment. ` +
    `Allowed runtimes: ${RUNTIMES_ALLOWED_ON_EVAL.join(", ")}. ` +
    `Eval nodes run benchmarks; GPU nodes host models.`
  );
}
