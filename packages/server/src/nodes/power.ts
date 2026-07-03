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

/**
 * Shell command (run over SSH) that resolves the interface bound to `ip` and
 * arms magic-packet Wake-on-LAN on it via ethtool (`wol g`). NICs commonly ship
 * with WOL disabled, and the setting does not persist across reboots, so we arm
 * it right before a shutdown/suspend — otherwise a later /wake sends a magic
 * packet the NIC ignores.
 *
 * Best-effort by design: ethtool may be absent or the driver may reject the
 * mode; callers run this inside try/catch. `ip` is interpolated into the remote
 * command, so it is validated as a strict IPv4 literal first (see macCaptureCmd).
 */
export function wolArmCmd(ip: string): string {
  if (!isValidIpv4(ip)) throw new Error(`Invalid IPv4 for WOL arm: ${ip}`);
  return (
    `ifc=$(ip -o -4 addr show | awk -v ip="${ip}" '$4 ~ "^"ip"/" {print $2; exit}'); ` +
    `[ -n "$ifc" ] && sudo ethtool -s "$ifc" wol g`
  );
}

const MAC_RE = /^([0-9a-f]{2}:){5}[0-9a-f]{2}$/;

/** Trim+lowercase a captured MAC, or null if it is not a valid MAC. */
export function normalizeMac(raw: string): string | null {
  const m = raw.trim().toLowerCase();
  return MAC_RE.test(m) ? m : null;
}
