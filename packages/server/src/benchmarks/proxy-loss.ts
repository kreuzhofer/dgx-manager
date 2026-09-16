/**
 * Recognising a run that died because the MANAGER restarted underneath it.
 *
 * An accuracy run's lm-eval job is a systemd unit on the eval node and survives
 * a manager restart perfectly well. What does not survive is the reasoning
 * proxy: `startReasoningProxy` binds an ephemeral port INSIDE the manager
 * process, and recreating the container takes the socket with it. The job then
 * fails against a host/port that no longer exists.
 *
 * The run record said only `lm-eval exited with code 1`. The cause was findable
 * solely by reading a urllib3 traceback in the log and correlating it against
 * docker events — for a failure that a routine `docker compose up --build`
 * causes (#22).
 */

/** What the log proves about where the connection was refused. */
export type ProxyLossEvidence = {
  host: string;
  port: number;
  /** The matched log line, kept so the finding can be justified, not just asserted. */
  line: string;
};

// urllib3's message, which carries the host and port we need to attribute the
// failure. Matched on the structural part, not the whole sentence, so a wording
// change upstream degrades to "no evidence" rather than a wrong attribution.
const REFUSED =
  /HTTPConnection\(host='([^']+)',\s*port=(\d+)\):\s*Failed to establish a new connection/;

function hostPortOf(url: string): { host: string; port: string } | null {
  try {
    const u = new URL(url);
    return { host: u.hostname, port: u.port || (u.protocol === "https:" ? "443" : "80") };
  } catch {
    return null;
  }
}

/**
 * Find evidence that the reasoning proxy vanished mid-run.
 *
 * `proxyHosts` are the addresses the proxy could have been advertised on —
 * MANAGER_ADVERTISE_HOST for a remote run, 127.0.0.1 for a local one.
 *
 * `endpointUrl` is the MODEL endpoint, and a refusal there is excluded
 * deliberately: that is the deployment dying, which is a different incident
 * with a different fix. Reporting one as the other is the misattribution this
 * codebase has already been bitten by (#92), so the check is explicit rather
 * than relying on the hosts happening to differ.
 */
export function detectProxyLoss(
  logText: string,
  opts: { proxyHosts: (string | undefined | null)[]; endpointUrl?: string | null },
): ProxyLossEvidence | null {
  if (!logText) return null;
  const hosts = new Set(opts.proxyHosts.filter((h): h is string => !!h));
  if (hosts.size === 0) return null;
  const endpoint = opts.endpointUrl ? hostPortOf(opts.endpointUrl) : null;

  for (const line of logText.split("\n")) {
    const m = REFUSED.exec(line);
    if (!m) continue;
    const host = m[1];
    const port = m[2];
    if (!hosts.has(host)) continue;
    // The model endpoint going away is a different failure. Same host is
    // possible (a local run, or a manager that also serves models), so compare
    // the port too rather than assuming the hosts distinguish them.
    if (endpoint && endpoint.host === host && endpoint.port === port) continue;
    return { host, port: Number(port), line: line.trim() };
  }
  return null;
}

/** The message a human should get instead of "lm-eval exited with code 1". */
export function proxyLossReason(e: ProxyLossEvidence): string {
  return (
    `the manager restarted mid-run: lm-eval lost the reasoning proxy at ` +
    `${e.host}:${e.port}, which lives inside the manager process on an ephemeral ` +
    `port and does not survive a container restart. The lm-eval job itself was ` +
    `fine — only its target went away. Avoid rebuilding the server while an ` +
    `accuracy run is in flight (#22).`
  );
}

/**
 * Upgrade a failure reason when the log explains it better than the exit code.
 *
 * Returns the original reason unchanged when there is no evidence, so this can
 * sit on the failure path without inventing causes.
 */
export function explainFailure(
  reason: string,
  logText: string,
  opts: { proxyHosts: (string | undefined | null)[]; endpointUrl?: string | null },
): string {
  const e = detectProxyLoss(logText, opts);
  return e === null ? reason : `${reason} — ${proxyLossReason(e)}`;
}

/**
 * The message for a run we ended OURSELVES at boot, having established that its
 * proxy died with the previous container.
 *
 * Separate from {@link proxyLossReason} because here we have no log evidence and
 * therefore no port — and inventing one to reuse that message would put a number
 * in the record that nothing measured.
 */
export function proxyLostAtRestartReason(): string {
  return (
    "ended by the manager: this accuracy run's reasoning proxy lived inside the " +
    "manager process and did not survive the restart, so the lm-eval job on the " +
    "eval node was still running but had nothing to talk to. The job was " +
    "cancelled rather than left to burn GPU until it failed. Avoid rebuilding " +
    "the server while an accuracy run is in flight (#22)."
  );
}

/**
 * Did this run depend on a proxy hosted inside the manager process?
 *
 * Only accuracy runs do, and only when `reasoning` is set — `runAccuracy` starts
 * the proxy on exactly that condition. Parsed defensively: an unreadable config
 * yields false, because claiming a run was proxied when we cannot tell would
 * fail it for a reason we have not established.
 */
export function usesManagerProxy(row: { kind: string; config: string }): boolean {
  if (row.kind !== "accuracy") return false;
  try {
    const c: unknown = JSON.parse(row.config);
    return !!c && typeof c === "object" && (c as { reasoning?: unknown }).reasoning === true;
  } catch {
    return false;
  }
}
