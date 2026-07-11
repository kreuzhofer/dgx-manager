/** Job dirs whose mtime is older than the retention window. Pure. */
export const JOB_RETENTION_MS = 14 * 86_400_000;

export function stalePaths(
  entries: { path: string; mtimeMs: number }[],
  nowMs: number,
  retentionMs: number = JOB_RETENTION_MS,
): string[] {
  return entries.filter((e) => nowMs - e.mtimeMs > retentionMs).map((e) => e.path);
}
