/**
 * Decide whether to (re)send an agent:deployment:status message.
 *
 * A status TRANSITION (e.g. starting -> running) is always reported, even when
 * VRAM has stabilized — the old code gated the send on a >1% VRAM change, which
 * dropped the starting->running flip once weights finished loading and VRAM went
 * flat, leaving deploys stuck at "starting". VRAM changes still trigger a resend
 * (to refresh vramActual), but only when VRAM is actually readable (>0).
 */
export function shouldReportStatus(args: {
  lastStatus: string | undefined;
  status: string;
  lastVram: number | undefined;
  vramUsed: number;
}): boolean {
  if (args.status !== args.lastStatus) return true;
  if (args.vramUsed > 0) {
    if (args.lastVram == null) return true;
    if (Math.abs(args.vramUsed - args.lastVram) > args.lastVram * 0.01) return true;
  }
  return false;
}
