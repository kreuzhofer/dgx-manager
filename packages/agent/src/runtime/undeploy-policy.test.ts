import { describe, it, expect } from "vitest";
import { shouldForceStopSharedContainer, selectContainerOwnerId } from "./undeploy-policy.js";

describe("shouldForceStopSharedContainer", () => {
  // All vLLM deployments on a node share ONE `vllm_node` container. Force-stop
  // it ONLY when the deployment being undeployed actually owns the running one.

  it("stops when the target owns the running container", () => {
    expect(shouldForceStopSharedContainer("A", ["A"], true)).toBe(true);
    expect(shouldForceStopSharedContainer("A", ["A", "ignored-extra"], true)).toBe(true);
  });

  // The bug this fixes: deleting deployment B must NOT tear down deployment A's
  // live container just because they share the `vllm_node` name.
  it("does NOT stop when a different deployment owns the running container", () => {
    expect(shouldForceStopSharedContainer("B", ["A"], true)).toBe(false);
  });

  // Conservative: a running container with no tracked owner is not attributable
  // to the target, so an unrelated undeploy must not kill it.
  it("does NOT stop when the container is running but no owner is tracked", () => {
    expect(shouldForceStopSharedContainer("B", [], true)).toBe(false);
  });

  // Nothing to stop if no container is running, regardless of ownership.
  it("does NOT stop when no container is running", () => {
    expect(shouldForceStopSharedContainer("A", ["A"], false)).toBe(false);
    expect(shouldForceStopSharedContainer("B", [], false)).toBe(false);
  });
});

describe("selectContainerOwnerId", () => {
  // One container per node → the owner is the most-recently-started tracked entry.
  it("returns undefined when nothing is tracked", () => {
    expect(selectContainerOwnerId([])).toBeUndefined();
  });

  it("returns the only tracked deployment", () => {
    expect(selectContainerOwnerId([{ deploymentId: "A", startedAt: "2026-06-11T00:00:00Z" }])).toBe("A");
  });

  it("returns the most-recently-started when several are tracked", () => {
    const tracked = [
      { deploymentId: "old", startedAt: "2026-06-11T08:00:00Z" },
      { deploymentId: "new", startedAt: "2026-06-11T11:30:00Z" },
      { deploymentId: "mid", startedAt: "2026-06-11T10:00:00Z" },
    ];
    expect(selectContainerOwnerId(tracked)).toBe("new");
  });
});
