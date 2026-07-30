/**
 * Tie-break rotation, kept per published name.
 *
 * A single counter shared by the whole gateway looked sufficient — it only has
 * to vary — but it is not. Consumed on every selection, including pools of one,
 * it aliases with other pools' traffic: a two-member pool interleaved 1:1 with
 * any other name only ever sees one parity, so one of its members is never
 * chosen. Reproduced before this was split out.
 *
 * One counter per pool removes the coupling: a pool's rotation advances only
 * with its own traffic.
 */

const rotations = new Map<string, number>();

/** Next rotation value for this published name. */
export function nextRotation(publishedName: string): number {
  const next = (rotations.get(publishedName) ?? 0) + 1;
  // Wrap well below Number.MAX_SAFE_INTEGER so the value stays an exact
  // integer forever; the modulo downstream only cares that it advances.
  rotations.set(publishedName, next % 1_000_000);
  return next;
}

/** Test seam: forget every pool's rotation. Never called in production. */
export function resetRotations(): void {
  rotations.clear();
}
