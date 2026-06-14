import { isValidIpv4 } from "../ws/node-ip.js";

export type PowerAction = "reboot" | "shutdown" | "sleep";

const COMMANDS: Record<PowerAction, string> = {
  // --no-block returns immediately without waiting for the systemd job, so the
  // SSH exec gets a clean exit code before the machine actually goes down.
  reboot: "sudo systemctl --no-block reboot",
  shutdown: "sudo systemctl --no-block poweroff",
  // suspend is interactive-safe and returns once the node has entered S3.
  sleep: "sudo systemctl suspend",
};

export function powerCommand(action: PowerAction): string {
  const cmd = COMMANDS[action];
  if (!cmd) throw new Error(`Unknown power action: ${action}`);
  return cmd;
}

/**
 * Shell command (run over SSH) that resolves the network interface whose IPv4
 * address matches `ip`, then prints that interface's MAC. Output is a single
 * MAC line (or empty if not found).
 *
 * `ip` is interpolated into a remote shell command, so it is validated as a
 * strict IPv4 literal first — this is the only place a node-supplied address
 * reaches an executed command, and a non-IPv4 value could carry shell
 * metacharacters. Callers run this inside try/catch (MAC capture is
 * best-effort), so a throw here simply skips capture rather than failing.
 */
export function macCaptureCmd(ip: string): string {
  if (!isValidIpv4(ip)) throw new Error(`Invalid IPv4 for MAC capture: ${ip}`);
  return (
    `ifc=$(ip -o -4 addr show | awk -v ip="${ip}" '$4 ~ "^"ip"/" {print $2; exit}'); ` +
    `cat /sys/class/net/"$ifc"/address 2>/dev/null`
  );
}

const MAC_RE = /^([0-9a-f]{2}:){5}[0-9a-f]{2}$/;

/** Trim+lowercase a captured MAC, or null if it is not a valid MAC. */
export function normalizeMac(raw: string): string | null {
  const m = raw.trim().toLowerCase();
  return MAC_RE.test(m) ? m : null;
}
