// Deploy lifecycle phases in FORWARD order. The agent classifies container log
// lines into these and reports them as the deployment's status. Progression is
// MONOTONIC — the dashboard must never flip backward. Concretely: after the
// ~13-min weight load, vLLM logs "Prefetching checkpoint files into page cache"
// (matches the download heuristic) and then compiles; without a forward-only
// guard the status flapped loading → downloading → (stuck) through capture,
// which read as "downloading while it's compiling". Ranking + a max-rank guard
// make a late lower-ranked match a no-op instead of a confusing regression.
export const PHASE_ORDER = [
  "starting",
  "building",
  "launching",
  "downloading",
  "loading",
  "compiling",
  "running",
] as const;

export type Phase = (typeof PHASE_ORDER)[number];

/** Rank of a phase in the lifecycle, or -1 for a non-progression status
 *  (pending/failed/stopped/removing) which is not subject to the forward guard. */
export function phaseRank(phase: string): number {
  return (PHASE_ORDER as readonly string[]).indexOf(phase);
}

/** True when `to` is a known progression phase strictly ahead of `from`. */
export function isForwardPhase(from: string, to: string): boolean {
  const t = phaseRank(to);
  return t >= 0 && t > phaseRank(from);
}

/**
 * Classify a single container log line into a lifecycle phase, or null when the
 * line carries no phase signal. Case-insensitive substring match. Order is
 * lifecycle order, but the caller's forward-only guard is what actually prevents
 * regressions — this only needs to name the phase a line belongs to.
 */
export function detectPhase(line: string): Phase | null {
  const l = line.toLowerCase();
  if (l.includes("building") || l.includes("=== building")) return "building";
  if (l.includes("copying") && l.includes("image to")) return "building";
  if (l.includes("downloading model") || l.includes("=== downloading")) return "downloading";
  if (l.includes("fetching") && l.includes("files")) return "downloading";
  if (l.includes("starting head node") || l.includes("applying mod")) return "launching";
  if (l.includes("starting ray") || l.includes("ray worker") || l.includes("starting worker node")) return "launching";
  if (l.includes("loading safetensors") || l.includes("loading model") || l.includes("loading weights")) return "loading";
  // torch.compile / inductor / Triton + CUDA-graph capture are the long tail
  // between load and serving (the DCP stack spends ~1-16 min here).
  if (l.includes("torch.compile") || l.includes("compiling a graph") || l.includes("torch_compile_cache")) return "compiling";
  if (l.includes("capturing cuda graph") || l.includes("graph capturing") || l.includes("capturing model for speculator")) return "compiling";
  if (l.includes("application startup complete")) return "running";
  return null;
}
