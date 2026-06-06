/**
 * Node management-IP resolution (pure).
 *
 * The server normally derives a node's management IP from the agent's WebSocket
 * source address. That breaks when the agent is co-located with the server on a
 * docker bridge (e.g. the manager host also being a cluster node): the observed
 * source is the bridge gateway (172.19.0.1), not the host's real NIC, which then
 * makes Ray bind the wrong IP and the cluster fails to form.
 *
 * The agent may therefore send an explicit `advertiseIp` (from NODE_ADVERTISE_IP).
 * When valid, it wins over the socket source.
 */

export function isValidIpv4(ip: string | null | undefined): ip is string {
  if (!ip) return false;
  const m = ip.match(/^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/);
  if (!m) return false;
  return m.slice(1, 5).every((o) => {
    const n = Number(o);
    return n >= 0 && n <= 255;
  });
}

/**
 * Prefer an agent-supplied advertise IP over the WebSocket source IP.
 * Returns the chosen IPv4, or null if neither is a valid IPv4.
 */
export function resolveNodeIp(
  advertiseIp: string | null | undefined,
  remoteIp: string | null | undefined,
): string | null {
  if (isValidIpv4(advertiseIp)) return advertiseIp;
  if (isValidIpv4(remoteIp)) return remoteIp;
  return null;
}
