/**
 * Order selected cluster node ids with the chosen head first, so the deploy
 * API (which treats nodeIds[0] as the rank-0 head) launches the operator's
 * chosen head. Deduped; head absent/unknown -> selection order unchanged.
 */
export function buildClusterNodeIds(
  headId: string | null | undefined,
  selected: Iterable<string>,
): string[] {
  const ids = Array.from(new Set(selected));
  if (!headId || !ids.includes(headId)) return ids;
  return [headId, ...ids.filter((id) => id !== headId)];
}
