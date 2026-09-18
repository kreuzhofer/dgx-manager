/**
 * Wording for the peer-throughput verdict the gateway view attaches to each
 * pool member (#88).
 *
 * The server decides; this only turns a verdict into the sentence an operator
 * reads. Kept out of the component so it can be tested without rendering.
 */

export type ThroughputState = "ok" | "suspect" | "not-comparable";

export type NotComparableReason =
  | "single-member"
  | "insufficient-samples"
  | "idle"
  | "multi-deployment-node"
  | "model-mismatch";

export interface ThroughputVerdict {
  state: ThroughputState;
  rate: number | null;
  peerRates: { node: string; rate: number }[];
  ratio: number | null;
  windowMinutes: number;
  reason?: NotComparableReason;
}

export interface ThroughputView {
  tone: "ok" | "suspect" | "muted";
  label: string;
  detail?: string;
}

/**
 * Why a comparison was declined, in the words an operator would use. Never
 * show the bare enum: "idle" alone reads like a fault.
 */
const REASONS: Record<NotComparableReason, string> = {
  "single-member": "nothing to compare against — this pool has one member",
  "insufficient-samples": "too little traffic in the window to take a median",
  idle: "served nothing in the window",
  "multi-deployment-node": "another deployment shares this node, so its rate is a sum",
  "model-mismatch": "pool members are serving different models",
};

const median = (xs: number[]): number | null => {
  if (xs.length === 0) return null;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length / 2;
  return s.length % 2 ? s[(s.length - 1) / 2] : (s[mid - 1] + s[mid]) / 2;
};

export function describeThroughput(verdict: ThroughputVerdict | null): ThroughputView | null {
  if (!verdict) return null;
  const { state, rate, ratio, peerRates, windowMinutes } = verdict;

  if (state === "not-comparable") {
    return {
      tone: "muted",
      label: "not compared",
      detail: verdict.reason ? REASONS[verdict.reason] : undefined,
    };
  }

  const pct = ratio === null ? null : `${Math.round(ratio * 100)}%`;
  const peerMedian = median(peerRates.map((p) => p.rate));

  if (state === "suspect") {
    return {
      tone: "suspect",
      label: "behind peers",
      detail:
        `${rate?.toFixed(1)} t/s against ${peerMedian?.toFixed(1)} t/s across ` +
        `${peerRates.length} ${peerRates.length === 1 ? "peer" : "peers"} ` +
        `over ${windowMinutes} min (${pct})`,
    };
  }

  return {
    tone: "ok",
    label: `${rate?.toFixed(1)} t/s`,
    detail: pct ? `${pct} of the peer median over ${windowMinutes} min` : undefined,
  };
}

/** Just enough of the gateway pool view to find the members behind their peers. */
export interface PoolLike {
  publishedName: string;
  members: { node: string; throughput: ThroughputVerdict | null }[];
}

/**
 * Nodes with at least one pool member behind its peers, keyed by node name.
 *
 * The overview shows one card per node, so several members collapse to one
 * badge: being behind in any pool is reason enough to look at the machine.
 */
export function suspectedNodes(pools: PoolLike[]): Map<string, ThroughputVerdict> {
  const found = new Map<string, ThroughputVerdict>();
  for (const pool of pools) {
    for (const m of pool.members) {
      if (m.throughput?.state === "suspect") found.set(m.node, m.throughput);
    }
  }
  return found;
}
