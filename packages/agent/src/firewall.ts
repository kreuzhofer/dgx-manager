import { execFile } from "node:child_process";
import { promisify } from "node:util";

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
 * IPv6 mirror is loopback-only — the manager is IPv4-only.
 */

const CHAIN = "DGX_AGENT_OLLAMA";
const OLLAMA_PORT = "11434";

/** Hostname / IPv4 / IPv6 charset. Anything outside this (spaces, `;`, `$`,
 *  quotes, …) is rejected so a hostile MANAGER_URL can never smuggle extra
 *  iptables arguments or shell syntax. */
const HOST_RE = /^[0-9a-zA-Z_.:-]+$/;

/**
 * One step of the firewall install. The applier interprets the kinds:
 *  - "run": must succeed; a failure aborts the install and is logged loudly.
 *  - "tolerate": failure is expected sometimes (`-N` when the chain already
 *    exists) and is silently accepted.
 *  - "check-then": run `check`; only if it exits non-zero, run `then`
 *    (which must succeed). Encodes "insert the INPUT jump exactly once".
 */
export type FirewallStep =
  | { kind: "run"; argv: string[] }
  | { kind: "tolerate"; argv: string[] }
  | { kind: "check-then"; check: string[]; then: string[] };

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

/**
 * Pure builder: the ordered steps that idempotently install the Ollama
 * firewall for the given manager host. Throws if the host contains anything
 * outside the hostname/IP charset (argv injection guard).
 */
export function buildFirewallSteps(managerHost: string): FirewallStep[] {
  if (!HOST_RE.test(managerHost)) {
    throw new Error(`manager host is not host-like: "${managerHost}"`);
  }
  const jump = ["INPUT", "-p", "tcp", "--dport", OLLAMA_PORT, "-j", CHAIN];
  return [
    // IPv4: manager + loopback allowed, everything else dropped
    { kind: "tolerate", argv: ["iptables", "-N", CHAIN] },
    { kind: "run", argv: ["iptables", "-F", CHAIN] },
    { kind: "run", argv: ["iptables", "-A", CHAIN, "-s", managerHost, "-j", "ACCEPT"] },
    { kind: "run", argv: ["iptables", "-A", CHAIN, "-i", "lo", "-j", "ACCEPT"] },
    { kind: "run", argv: ["iptables", "-A", CHAIN, "-j", "DROP"] },
    { kind: "check-then", check: ["iptables", "-C", ...jump], then: ["iptables", "-I", ...jump] },
    // IPv6: loopback-only (the manager is IPv4-only)
    { kind: "tolerate", argv: ["ip6tables", "-N", CHAIN] },
    { kind: "run", argv: ["ip6tables", "-F", CHAIN] },
    { kind: "run", argv: ["ip6tables", "-A", CHAIN, "-i", "lo", "-j", "ACCEPT"] },
    { kind: "run", argv: ["ip6tables", "-A", CHAIN, "-j", "DROP"] },
    { kind: "check-then", check: ["ip6tables", "-C", ...jump], then: ["ip6tables", "-I", ...jump] },
  ];
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
 * Apply the Ollama firewall for the given MANAGER_URL. Never throws: any
 * failure is logged loudly (agent journal) with the failing command +
 * stderr, and `false` is returned — metrics and deploy duties must survive
 * a firewall failure. Returns `true` when all rules are in place.
 */
export async function applyOllamaFirewall(
  managerUrl: string,
  exec: FirewallExec = defaultExec,
): Promise<boolean> {
  let steps: FirewallStep[];
  let host: string;
  try {
    host = parseManagerHost(managerUrl);
    steps = buildFirewallSteps(host);
  } catch (err) {
    console.error(`[firewall] NOT applied — cannot derive manager host: ${err instanceof Error ? err.message : err}`);
    return false;
  }

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
      console.error(
        `[firewall] FAILED — Ollama :11434 may be exposed. Command \`sudo -n ${argv.join(" ")}\` failed: ${stderrOf(err)}`,
      );
      return false;
    }
  }

  console.log(`[firewall] Ollama :11434 restricted to ${host} + loopback (IPv6: loopback-only)`);
  return true;
}
