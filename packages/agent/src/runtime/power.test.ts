import { describe, it, expect } from "vitest";
import { powerCommand } from "./power.js";

describe("agent powerCommand", () => {
  it("maps graceful reboot/shutdown/sleep to the systemd variants", () => {
    expect(powerCommand("reboot")).toBe("sudo systemctl --no-block reboot");
    expect(powerCommand("shutdown")).toBe("sudo systemctl --no-block poweroff");
    expect(powerCommand("sleep")).toBe("sudo systemctl suspend");
  });
  // Force = immediate hard reset for a wedged node; only reboot/shutdown.
  it("maps forced reboot/shutdown to --force --force", () => {
    expect(powerCommand("reboot", { force: true })).toBe(
      "sudo systemctl --force --force reboot",
    );
    expect(powerCommand("shutdown", { force: true })).toBe(
      "sudo systemctl --force --force poweroff",
    );
  });
  it("ignores force for suspend", () => {
    expect(powerCommand("sleep", { force: true })).toBe("sudo systemctl suspend");
  });
  // Must match the server's command strings exactly so agent/SSH paths agree.
  it("matches the server's graceful reboot string", () => {
    expect(powerCommand("reboot")).toBe("sudo systemctl --no-block reboot");
  });
});
