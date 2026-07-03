export type PowerAction = "reboot" | "shutdown" | "sleep";

/**
 * The local shell command the agent runs to reboot/shut down/suspend its own
 * node. Mirrors the server's SSH-side power command (packages/server/src/nodes/
 * power.ts) so the agent-primary and SSH-fallback paths behave identically.
 *
 * Force applies to reboot and shutdown: the graceful --no-block variants ask
 * systemd to stop all services first, which can itself hang on a wedged node.
 * Double --force skips service shutdown AND unmounting and issues the reboot(2)
 * syscall at once. Not meaningful for suspend, so ignored there.
 */
export function powerCommand(action: PowerAction, opts?: { force?: boolean }): string {
  if (opts?.force && (action === "reboot" || action === "shutdown")) {
    const verb = action === "reboot" ? "reboot" : "poweroff";
    return `sudo systemctl --force --force ${verb}`;
  }
  switch (action) {
    case "reboot":
      return "sudo systemctl --no-block reboot";
    case "shutdown":
      return "sudo systemctl --no-block poweroff";
    case "sleep":
      return "sudo systemctl suspend";
    default:
      throw new Error(`Unknown power action: ${action}`);
  }
}
