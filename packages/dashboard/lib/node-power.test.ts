import { describe, it, expect } from "vitest";
import { canWake, isNodeInactive, nodeMetricsPlaceholder, showWakeButton } from "./node-power.js";

describe("canWake", () => {
  it("a node with a captured MAC can be woken", () => {
    expect(canWake("74:56:3c:5a:c7:b0")).toBe(true);
  });
  it("a node whose MAC was never captured cannot be woken", () => {
    expect(canWake(null)).toBe(false);
    expect(canWake(undefined)).toBe(false);
  });
  it("an empty or blank MAC is not a captured MAC", () => {
    expect(canWake("")).toBe(false);
    expect(canWake("   ")).toBe(false);
  });
});

describe("isNodeInactive", () => {
  it("off and asleep nodes are inactive", () => {
    expect(isNodeInactive("off")).toBe(true);
    expect(isNodeInactive("asleep")).toBe(true);
  });
  it("a node still waking is inactive — the agent has not reconnected yet", () => {
    expect(isNodeInactive("waking")).toBe(true);
  });
  it("on and rebooting nodes are active", () => {
    expect(isNodeInactive("on")).toBe(false);
    expect(isNodeInactive("rebooting")).toBe(false);
  });
  it("an absent power state is treated as active", () => {
    expect(isNodeInactive(undefined)).toBe(false);
    expect(isNodeInactive(null)).toBe(false);
  });
});

describe("nodeMetricsPlaceholder", () => {
  const MAC = "74:56:3c:5a:c7:b0";

  it("an active node with no samples yet just says so", () => {
    expect(nodeMetricsPlaceholder("on", MAC)).toBe("No metrics yet");
    expect(nodeMetricsPlaceholder(undefined, null)).toBe("No metrics yet");
  });

  it("a wakeable powered-off node points at the Wake button", () => {
    expect(nodeMetricsPlaceholder("off", MAC)).toBe("Powered off — Wake to bring it back");
    expect(nodeMetricsPlaceholder("asleep", MAC)).toBe("Powered off — Wake to bring it back");
  });

  it("a wakeable node still waking offers a retry", () => {
    expect(nodeMetricsPlaceholder("waking", MAC)).toBe(
      "Waking… — click Wake to retry if it doesn't come back",
    );
  });

  // The dead end this issue introduces: with Wake gated on the MAC, an inactive
  // node that never had one captured shows no Wake button, so the copy must not
  // send the operator looking for it.
  it("an inactive node with no captured MAC explains why it cannot be woken", () => {
    for (const state of ["off", "asleep", "waking"]) {
      const copy = nodeMetricsPlaceholder(state, null);
      expect(copy).toMatch(/MAC/);
      expect(copy).not.toMatch(/click Wake|Wake to bring/);
    }
  });

  it("names the actual power state when no MAC was captured", () => {
    expect(nodeMetricsPlaceholder("off", null)).toMatch(/^Powered off/);
    expect(nodeMetricsPlaceholder("waking", "")).toMatch(/^Waking/);
  });
});

describe("showWakeButton", () => {
  const MAC = "74:56:3c:5a:c7:b0";

  it("an inactive node with a captured MAC offers Wake", () => {
    expect(showWakeButton("off", MAC)).toBe(true);
    expect(showWakeButton("asleep", MAC)).toBe(true);
    expect(showWakeButton("waking", MAC)).toBe(true);
  });

  // A magic packet aimed at a live host is a no-op, so the button would be a
  // control that does nothing where it appears.
  it("a running node does not offer Wake, even with a captured MAC", () => {
    expect(showWakeButton("on", MAC)).toBe(false);
    expect(showWakeButton("rebooting", MAC)).toBe(false);
    expect(showWakeButton(undefined, MAC)).toBe(false);
  });

  // The half #76 added, kept: no MAC means no packet can be addressed, so the
  // button must not appear only to fail with a 409 after the click.
  it("a node with no captured MAC never offers Wake", () => {
    for (const state of ["on", "off", "asleep", "waking", "rebooting"]) {
      expect(showWakeButton(state, null)).toBe(false);
      expect(showWakeButton(state, "  ")).toBe(false);
    }
  });

  it("agrees with the placeholder copy: the button shows iff the copy points at it", () => {
    for (const state of ["on", "off", "asleep", "waking", "rebooting"]) {
      for (const mac of [MAC, null]) {
        const pointsAtButton = /click Wake|Wake to bring/.test(nodeMetricsPlaceholder(state, mac));
        expect(pointsAtButton).toBe(showWakeButton(state, mac));
      }
    }
  });
});
