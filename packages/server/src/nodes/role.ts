/**
 * A node's role decides what may be *deployed* onto it. It never affects
 * metrics, provisioning, or agent management.
 *
 * `eval` exists for agenthost: a benchmark runner with no CUDA VRAM. It hosts
 * NOTHING. VRAM and arch admission would usually reject a vLLM/dgxrun deploy
 * anyway, but relying on that is accidental, not intentional — this makes it
 * explicit.
 *
 * It used to permit small Ollama models, because agenthost was the only sensible
 * place for embeddings: a Spark runs Ollama on its ARM CPU with no GPU, measured
 * ~12x slower per embedding, and burns a GPU node to do it. The owner's decision
 * on 2026-09-27 is that agenthost is a benchmark runner only, which means
 * embeddings have no good home on this fleet until one is chosen deliberately.
 */
export type NodeRole = "gpu" | "eval";

/**
 * Runtimes an `eval` node is permitted to host. Everything else is refused.
 *
 * Deliberately empty: an eval node hosts nothing. Kept as a list rather than
 * collapsed into `return false` so the rule stays one editable fact, and so
 * `runtimeAllowedOnNode` remains the single place that answers the question —
 * anything restating it elsewhere drifts (see the auto-selection filter in
 * routes/deployments.ts, which did).
 */
export const RUNTIMES_ALLOWED_ON_EVAL: readonly string[] = [];

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
  // With an empty allow-list, "Allowed runtimes: ." would be worse than saying
  // nothing — so the empty case gets its own sentence rather than a blank list.
  const allowed = RUNTIMES_ALLOWED_ON_EVAL.length > 0
    ? `Allowed runtimes: ${RUNTIMES_ALLOWED_ON_EVAL.join(", ")}.`
    : "It cannot host any deployment.";
  return (
    `Node "${nodeName}" has role "eval" and cannot host a "${runtime}" deployment. ` +
    `${allowed} ` +
    `Eval nodes run benchmarks; GPU nodes host models.`
  );
}
