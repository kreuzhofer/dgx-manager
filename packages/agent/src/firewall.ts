import { execFile } from "node:child_process";
import { promisify } from "node:util";
import type { SelfAuditCheck } from "./self-audit.js";

const execFileAsync = promisify(execFile);

/**
 * Ollama firewall — restrict the unauthenticated Ollama HTTP API (:11434)
 * to the dgx-manager host + loopback.
 *
 * Why: Ollama has no auth. An external client's embed call once loaded a
 * 15 GB model onto a node mid-vLLM-startup and killed a 4-node deployment.
 * The agent applies these rules at every startup (rules are intentionally
 * NON-persistent — reapplying at boot self-heals after a node reboot).
 *
 * Rule design (iptables-nft, own chain so we never clobber other rules):
 *   DGX_AGENT_OLLAMA: ACCEPT from manager IP (v4 only), ACCEPT -i lo, DROP
 *   INPUT: -p tcp --dport 11434 -j DGX_AGENT_OLLAMA (inserted once, guarded by -C)
 * IPv6 mirror is loopback-only — the manager is IPv4-only. Ollama itself
 * binds IPv4, so the v6 chain is defense-in-depth: a v6-only failure does
 * NOT expose the API.
 */

const CHAIN = "DGX_AGENT_OLLAMA";
const OLLAMA_PORT = "11434";

/** Hostname / IPv4 charset. Anything outside this (spaces, `;`, `$`,
 *  quotes, brackets, …) is rejected so a hostile MANAGER_URL can never
 *  smuggle extra iptables arguments or shell syntax. Note this also rejects
 *  bracketed IPv6 literals (`[::1]`) — the manager is IPv4-only by design. */
const HOST_RE = /^[0-9a-zA-Z_.:-]+$/;

/**
 * One step of the firewall install. The applier interprets the kinds:
 *  - "run": must succeed; a failure aborts the family's install and is
 *    logged loudly.
 *  - "tolerate": failure is expected sometimes (`-N` when the chain already
 *    exists) and is silently accepted.
 *  - "check-then": run `check`; only if it exits non-zero, run `then`
 *    (which must succeed). Encodes "insert the INPUT jump exactly once".
 */
export type FirewallStep =
  | { kind: "run"; argv: string[] }
  | { kind: "tolerate"; argv: string[] }
  | { kind: "check-then"; check: string[]; then: string[] };

/** Per-family apply outcome. "pending" = startup apply not finished yet. */
export type FamilyResult = "pending" | "applied" | "failed";
export interface FirewallState {
  v4: FamilyResult;
  v6: FamilyResult;
}

// Module state so the self-audit (fired on every WS open) can report the
// firewall's CURRENT state. The apply is fire-and-forget at boot, so the
// first audit of a fresh process may legitimately see "pending"; subsequent
// reconnect audits deliver the final state.
let state: FirewallState = { v4: "pending", v6: "pending" };

/** Current firewall state (copy). Feeds the self-audit check. */
export function getFirewallState(): FirewallState {
  return { ...state };
}

/**
 * Map firewall state to a dashboard self-audit check.
 *  - green: both families applied.
 *  - yellow: apply still in progress, OR v4 applied but v6 failed (v6 is
 *    defense-in-depth only — Ollama binds IPv4).
 *  - red: v4 not applied (bad MANAGER_URL, sudo denied, iptables failure)
 *    → :11434 may actually be exposed.
 */
export function firewallAuditCheck(s: FirewallState = getFirewallState()): SelfAuditCheck {
  const name = "Ollama Firewall";
  if (s.v4 === "applied" && s.v6 === "applied") {
    return { name, status: "green", detail: ":11434 restricted to manager + loopback (IPv6: loopback-only)" };
  }
  if (s.v4 === "pending" || s.v6 === "pending") {
    return { name, status: "yellow", detail: "Startup apply still in progress — re-audit on next reconnect" };
  }
  if (s.v4 === "applied") {
    return { name, status: "yellow", detail: "IPv4 rules active; IPv6 mirror failed (defense-in-depth only, Ollama binds IPv4)" };
  }
  return { name, status: "red", detail: "IPv4 rules NOT applied — Ollama :11434 may be exposed. Check agent journal for [firewall] errors." };
}

function assertHostLike(managerHost: string): void {
  if (!HOST_RE.test(managerHost)) {
    throw new Error(`manager host is not host-like: "${managerHost}"`);
  }
}

/**
 * Extract the host from a MANAGER_URL value like
 * `ws://192.168.44.14:4000/ws/agent` (ws://, wss://, hostnames all fine —
 * the host is returned as-is). Throws on anything unparseable or hostless:
 * a bad MANAGER_URL is a config error and must surface, not be guessed at.
 */
export function parseManagerHost(managerUrl: string): string {
  let url: URL;
  try {
    url = new URL(managerUrl);
  } catch {
    throw new Error(`unparseable MANAGER_URL: "${managerUrl}"`);
  }
  if (!url.hostname) {
    throw new Error(`MANAGER_URL has no host: "${managerUrl}"`);
  }
  return url.hostname;
}

/** Steps for one address family. `managerHost` present → v4 manager ACCEPT. */
function familySteps(tool: "iptables" | "ip6tables", managerHost?: string): FirewallStep[] {
  const jump = ["INPUT", "-p", "tcp", "--dport", OLLAMA_PORT, "-j", CHAIN];
  return [
    { kind: "tolerate", argv: [tool, "-N", CHAIN] },
    { kind: "run", argv: [tool, "-F", CHAIN] },
    ...(managerHost !== undefined
      ? [{ kind: "run", argv: [tool, "-A", CHAIN, "-s", managerHost, "-j", "ACCEPT"] } satisfies FirewallStep]
      : []),
    { kind: "run", argv: [tool, "-A", CHAIN, "-i", "lo", "-j", "ACCEPT"] },
    { kind: "run", argv: [tool, "-A", CHAIN, "-j", "DROP"] },
    { kind: "check-then", check: [tool, "-C", ...jump], then: [tool, "-I", ...jump] },
  ];
}

/**
 * Pure builder: the ordered steps that idempotently install the Ollama
 * firewall for the given manager host (v4: manager + loopback; v6:
 * loopback-only). Throws if the host contains anything outside the
 * hostname/IP charset (argv injection guard).
 */
export function buildFirewallSteps(managerHost: string): FirewallStep[] {
  assertHostLike(managerHost);
  return [...familySteps("iptables", managerHost), ...familySteps("ip6tables")];
}

/** Injectable exec: rejects (with optional `.stderr`) on non-zero exit. */
export type FirewallExec = (cmd: string, args: string[]) => Promise<void>;

const defaultExec: FirewallExec = async (cmd, args) => {
  await execFileAsync(cmd, args, { timeout: 15_000 });
};

/** Run one argv through `sudo -n` (never hangs on a password prompt). */
async function sudo(exec: FirewallExec, argv: string[]): Promise<void> {
  await exec("sudo", ["-n", ...argv]);
}

function stderrOf(err: unknown): string {
  const s = (err as { stderr?: unknown })?.stderr;
  return typeof s === "string" && s.trim() ? s.trim() : String(err);
}

/**
 * Apply one family's steps. On a hard failure, logs via `failMsg` and stops
 * that family (remaining steps skipped) — the other family still runs.
 */
async function applyFamily(
  exec: FirewallExec,
  steps: FirewallStep[],
  failMsg: (argv: string[], stderr: string) => string,
): Promise<boolean> {
  for (const step of steps) {
    if (step.kind === "tolerate") {
      try {
        await sudo(exec, step.argv);
      } catch {
        // Expected when the chain already exists — idempotent reapply.
      }
      continue;
    }

    let argv: string[];
    if (step.kind === "check-then") {
      try {
        await sudo(exec, step.check);
        continue; // rule already present — nothing to insert
      } catch {
        argv = step.then; // -C exited non-zero → insert the jump
      }
    } else {
      argv = step.argv;
    }

    try {
      await sudo(exec, argv);
    } catch (err) {
      console.error(failMsg(argv, stderrOf(err)));
      return false;
    }
  }
  return true;
}

/**
 * Apply the Ollama firewall for the given MANAGER_URL. Never throws: any
 * failure is logged loudly (agent journal) with the failing command +
 * stderr — metrics and deploy duties must survive a firewall failure.
 * Tracks per-family success in module state (see getFirewallState /
 * firewallAuditCheck): a v6-only failure is NOT an exposure (Ollama binds
 * IPv4) and is reported as such.
 */
export async function applyOllamaFirewall(
  managerUrl: string,
  exec: FirewallExec = defaultExec,
): Promise<FirewallState> {
  state = { v4: "pending", v6: "pending" };

  let host: string;
  try {
    host = parseManagerHost(managerUrl);
    assertHostLike(host);
  } catch (err) {
    console.error(
      `[firewall] NOT applied — Ollama :11434 may be exposed. Cannot derive manager host: ${err instanceof Error ? err.message : err}`,
    );
    state = { v4: "failed", v6: "failed" };
    return getFirewallState();
  }

  const cmdStr = (argv: string[]) => `sudo -n ${argv.join(" ")}`;

  const v4ok = await applyFamily(exec, familySteps("iptables", host), (argv, stderr) =>
    `[firewall] FAILED — Ollama :11434 may be exposed (IPv4 rules not applied). Command \`${cmdStr(argv)}\` failed: ${stderr}`,
  );
  state.v4 = v4ok ? "applied" : "failed";

  const v6ok = await applyFamily(exec, familySteps("ip6tables"), (argv, stderr) =>
    v4ok
      ? `[firewall] IPv4 rules active; IPv6 mirror failed (defense-in-depth only, Ollama binds IPv4). Command \`${cmdStr(argv)}\` failed: ${stderr}`
      : `[firewall] IPv6 mirror also failed. Command \`${cmdStr(argv)}\` failed: ${stderr}`,
  );
  state.v6 = v6ok ? "applied" : "failed";

  if (v4ok && v6ok) {
    console.log(`[firewall] Ollama :11434 restricted to ${host} + loopback (IPv6: loopback-only)`);
  }
  return getFirewallState();
}
