/**
 * Outcome of `systemctl show <unit> -p LoadState -p ActiveState -p ExecMainStatus`.
 *
 * The four cases are NOT interchangeable. `missing` means systemd positively said
 * the unit does not exist; `unknown` means we failed to ask (timeout, dead bus,
 * empty output). Collapsing `unknown` into anything else is the bug that tore down
 * four healthy dgxrun ranks on 2026-07-09 — here it would either kill an
 * 80-minute eval or mark it complete with no result.
 */
export type JobStatus =
  | { kind: "active" }
  | { kind: "exited"; code: number }
  | { kind: "missing" }
  | { kind: "unknown"; reason: string };

const ACTIVE_STATES = new Set(["active", "activating", "reloading"]);
const FINISHED_STATES = new Set(["inactive", "failed", "deactivating"]);

function parseProps(stdout: string): Map<string, string> {
  const m = new Map<string, string>();
  for (const line of stdout.split("\n")) {
    const eq = line.indexOf("=");
    if (eq > 0) m.set(line.slice(0, eq).trim(), line.slice(eq + 1).trim());
  }
  return m;
}

export function parseSystemctlShow(
  status: number | null,
  stdout: string,
  stderr: string,
): JobStatus {
  if (status !== 0) {
    return { kind: "unknown", reason: `systemctl exited ${status}: ${stderr.trim().slice(0, 200)}` };
  }
  const props = parseProps(stdout);
  const load = props.get("LoadState");
  const active = props.get("ActiveState");
  if (!load && !active) {
    return { kind: "unknown", reason: "systemctl produced no recognisable properties" };
  }
  if (load === "not-found") return { kind: "missing" };
  if (active && ACTIVE_STATES.has(active)) return { kind: "active" };
  if (active && FINISHED_STATES.has(active)) {
    const raw = props.get("ExecMainStatus");
    // Number("") and Number("   ") are 0, not NaN — a truncated/empty ExecMainStatus
    // must be "unknown", never a false exited(0). Require an explicit integer string.
    const code = raw !== undefined && /^-?\d+$/.test(raw) ? Number(raw) : NaN;
    if (!Number.isInteger(code)) {
      return { kind: "unknown", reason: `ActiveState=${active} but ExecMainStatus=${raw}` };
    }
    return { kind: "exited", code };
  }
  return { kind: "unknown", reason: `unrecognised ActiveState=${active ?? "<absent>"}` };
}
