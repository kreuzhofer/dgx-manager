import { describe, it, expect, vi } from "vitest";
import * as push from "./push.js";

function fakeSink() {
  const sent: { nodeId: string; msg: Record<string, unknown> }[] = [];
  return {
    sent,
    sendToAgent: (nodeId: string, msg: Record<string, unknown>) => sent.push({ nodeId, msg }),
    getConnectedNodeIds: () => ["node-a", "node-b"],
  };
}

describe("pushRegistriesToConnectedAgents", () => {
  it("sends cmd:set-registries to every connected node", async () => {
    vi.spyOn(push, "loadRegistryWire").mockResolvedValue([
      { name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" },
    ]);
    const sink = fakeSink();
    await push.pushRegistriesToConnectedAgents(sink);
    expect(sink.sent.map((s) => s.nodeId)).toEqual(["node-a", "node-b"]);
    expect(sink.sent[0].msg).toEqual({
      type: "cmd:set-registries",
      payload: { registries: [{ name: "rtx", url: "https://github.com/kreuzhofer/rtx-recipe-registry.git", subpath: "recipes" }] },
    });
  });
});
