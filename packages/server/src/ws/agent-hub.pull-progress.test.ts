import { describe, expect, it, vi } from "vitest";

vi.mock("../sse.js", () => ({
  broadcast: vi.fn(),
}));
vi.mock("../prisma.js", () => ({
  prisma: { /* not touched in this test */ },
}));
vi.mock("../metrics-buffer.js", () => ({
  metricsBuffer: { push: vi.fn() },
}));

import { broadcast as sseBroadcast } from "../sse.js";
import { handleOllamaPullProgress } from "./agent-hub.js";

describe("handleOllamaPullProgress", () => {
  it("translates a pull-progress payload to a deployment:progress SSE event", () => {
    handleOllamaPullProgress({
      deploymentId: "dep-1",
      status: "downloading",
      percent: 42,
      current: 4200000,
      total: 10000000,
    });
    expect(sseBroadcast).toHaveBeenCalledWith({
      type: "deployment:progress",
      payload: {
        deploymentId: "dep-1",
        phase: "downloading",
        phaseProgress: 42,
        current: 4200000,
        total: 10000000,
      },
    });
  });

  it("passes through non-downloading statuses verbatim as the phase", () => {
    handleOllamaPullProgress({
      deploymentId: "dep-2",
      status: "pulling manifest",
      percent: null,
      current: null,
      total: null,
    });
    expect(sseBroadcast).toHaveBeenLastCalledWith({
      type: "deployment:progress",
      payload: {
        deploymentId: "dep-2",
        phase: "pulling manifest",
        phaseProgress: 0,
        current: null,
        total: null,
      },
    });
  });
});
