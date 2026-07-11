/**
 * Decide which byte range of the job log to return, given what the caller has
 * already consumed.
 *
 * The manager persists its offset, so this is also the reattach path after a
 * manager restart: it asks from the byte it last stored. If the file is smaller
 * than the stored offset the log was rotated or the job dir recreated — restart
 * from zero rather than slice past the end and emit garbage.
 */
export function planRead(
  prevOffset: number,
  size: number,
): { from: number; to: number; truncated: boolean } {
  const truncated = !Number.isFinite(prevOffset) || prevOffset < 0 || prevOffset > size;
  const from = truncated ? 0 : prevOffset;
  return { from, to: size, truncated };
}
