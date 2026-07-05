import { describe, it, expect } from "vitest";
import { shouldReportStatus } from "./deploy-report.js";

describe("shouldReportStatus", () => {
  it("reports a status change even when VRAM is flat (the bug this fixes)", () => {
    expect(shouldReportStatus({ lastStatus: "starting", status: "running", lastVram: 103000, vramUsed: 103000 })).toBe(true);
  });
  it("reports a status change even when VRAM is 0/unreadable", () => {
    expect(shouldReportStatus({ lastStatus: "starting", status: "running", lastVram: 103000, vramUsed: 0 })).toBe(true);
  });
  it("reports the first time (lastStatus undefined)", () => {
    expect(shouldReportStatus({ lastStatus: undefined, status: "starting", lastVram: undefined, vramUsed: 50000 })).toBe(true);
  });
  it("reports when VRAM moved >1% at the same status", () => {
    expect(shouldReportStatus({ lastStatus: "starting", status: "starting", lastVram: 100000, vramUsed: 102000 })).toBe(true);
  });
  it("does NOT report when status and VRAM are both stable (throttle preserved)", () => {
    expect(shouldReportStatus({ lastStatus: "running", status: "running", lastVram: 103000, vramUsed: 103200 })).toBe(false);
  });
  it("does NOT report when VRAM is 0 and status is unchanged", () => {
    expect(shouldReportStatus({ lastStatus: "running", status: "running", lastVram: 103000, vramUsed: 0 })).toBe(false);
  });
});
