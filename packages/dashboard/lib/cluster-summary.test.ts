import { describe, expect, it } from "vitest";
import { formatClusterSummary } from "./cluster-summary";

describe("formatClusterSummary", () => {
  it("returns the single node's name for jobs without clusterNodes", () => {
    expect(
      formatClusterSummary({
        nodeId: "n1",
        node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" },
        clusterNodes: [],
      }),
    ).toBe("dgx-spark-01");
  });

  it("falls back to nodeId when node relation is missing", () => {
    expect(
      formatClusterSummary({
        nodeId: "node-abc-12345",
        node: null,
        clusterNodes: [],
      }),
    ).toBe("node-abc-12345");
  });

  it("lists multi-node clusters head-first", () => {
    expect(
      formatClusterSummary({
        nodeId: "n2",
        node: { name: "dgx-spark-02", ipAddress: "10.0.0.2" },
        clusterNodes: [
          { node: { name: "dgx-spark-03", ipAddress: "10.0.0.3" }, role: "worker" },
          { node: { name: "dgx-spark-02", ipAddress: "10.0.0.2" }, role: "head" },
          { node: { name: "dgx-spark-04", ipAddress: "10.0.0.4" }, role: "worker" },
        ],
      }),
    ).toBe("3 nodes: dgx-spark-02 (head), dgx-spark-03, dgx-spark-04");
  });

  it("renders all-worker clusters without a head label", () => {
    expect(
      formatClusterSummary({
        nodeId: "n1",
        node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" },
        clusterNodes: [
          { node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" }, role: "worker" },
          { node: { name: "dgx-spark-02", ipAddress: "10.0.0.2" }, role: "worker" },
        ],
      }),
    ).toBe("2 nodes: dgx-spark-01, dgx-spark-02");
  });

  it("handles a single clusterNodes entry as N=1", () => {
    expect(
      formatClusterSummary({
        nodeId: "n1",
        node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" },
        clusterNodes: [
          { node: { name: "dgx-spark-01", ipAddress: "10.0.0.1" }, role: "head" },
        ],
      }),
    ).toBe("1 node: dgx-spark-01 (head)");
  });
});
