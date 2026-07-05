import { describe, it, expect, vi } from "vitest";
import { CapClient } from "./cap-client.js";

describe("CapClient", () => {
  it("correlates a result back to the invoke promise", async () => {
    const sent: any[] = [];
    const c = new CapClient((_n, m) => sent.push(m));
    const p = c.invoke("node1", "diag.collect", null);
    const id = sent[0].payload.id;
    c.onResult({ id, ok: true, data: { memory: { totalMb: 124546 } } });
    await expect(p).resolves.toEqual({
      ok: true,
      data: { memory: { totalMb: 124546 } },
    });
  });

  it("times out when no result arrives", async () => {
    vi.useFakeTimers();
    const c = new CapClient(() => {}, { timeoutMs: 1000 });
    const p = c.invoke("n", "exec", {});
    vi.advanceTimersByTime(1001);
    await expect(p).resolves.toEqual({ ok: false, error: "cap timeout" });
    vi.useRealTimers();
  });
});
