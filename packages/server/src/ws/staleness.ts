export interface StaleNodeRow { id: string; status: string; lastSeen: Date | null; }

/**
 * Ids of nodes that claim to be "online" but whose last heartbeat is older than
 * thresholdMs — i.e. a half-open/dead socket the WS 'close' event hasn't caught.
 * Skips already-offline nodes and nodes that never heartbeated (null lastSeen).
 */
export function selectStaleNodes(nodes: StaleNodeRow[], nowMs: number, thresholdMs: number): string[] {
  return nodes
    .filter((n) => n.status === "online" && n.lastSeen != null && nowMs - n.lastSeen.getTime() > thresholdMs)
    .map((n) => n.id);
}
