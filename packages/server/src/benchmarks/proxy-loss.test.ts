import { describe, expect, it } from "vitest";
import { detectProxyLoss, explainFailure, proxyLossReason, proxyLostAtRestartReason, usesManagerProxy } from "./proxy-loss.js";

// Verbatim from the 2026-08-29 run named in #22 (remote runner, manager on .14)
// and from a local run, which differ only in the advertised host.
const REMOTE =
  "urllib3.exceptions.NewConnectionError: HTTPConnection(host='192.168.44.14', port=42443): " +
  "Failed to establish a new connection: [Errno 111] Connection refused";
const LOCAL =
  "urllib3.exceptions.NewConnectionError: HTTPConnection(host='127.0.0.1', port=39429): " +
  "Failed to establish a new connection: [Errno 111] Connection refused";

const log = (...lines: string[]) =>
  ["INFO Running requests", ...lines, "lm-eval exited with code 1"].join("\n");

describe("detectProxyLoss", () => {
  it("attributes a refusal at the advertised manager host to the proxy", () => {
    const e = detectProxyLoss(log(REMOTE), {
      proxyHosts: ["192.168.44.14"],
      endpointUrl: "http://192.168.44.36:8000/v1",
    });
    expect(e).toMatchObject({ host: "192.168.44.14", port: 42443 });
    expect(e!.line).toContain("Connection refused");
  });

  it("handles the local-run form too", () => {
    const e = detectProxyLoss(log(LOCAL), { proxyHosts: [undefined, "127.0.0.1"] });
    expect(e).toMatchObject({ host: "127.0.0.1", port: 39429 });
  });

  /**
   * The discriminator that matters. A refusal at the MODEL endpoint means the
   * deployment died — a different incident with a different fix. Calling that a
   * manager restart would be the same misattribution that made #92 expensive,
   * so it is excluded on host AND port rather than trusting the hosts to differ.
   */
  it("does NOT blame the manager when the MODEL endpoint refused", () => {
    const modelDown =
      "urllib3.exceptions.NewConnectionError: HTTPConnection(host='127.0.0.1', port=8000): " +
      "Failed to establish a new connection: [Errno 111] Connection refused";
    expect(
      detectProxyLoss(log(modelDown), {
        proxyHosts: ["127.0.0.1"],
        endpointUrl: "http://127.0.0.1:8000/v1",
      }),
    ).toBeNull();
  });

  it("still blames the proxy when it shares a host with the model but not the port", () => {
    // Same host, ephemeral port — that is the proxy, not the endpoint.
    const e = detectProxyLoss(log(LOCAL), {
      proxyHosts: ["127.0.0.1"],
      endpointUrl: "http://127.0.0.1:8000/v1",
    });
    expect(e).toMatchObject({ port: 39429 });
  });

  it("ignores a refusal at some unrelated host", () => {
    const other =
      "urllib3.exceptions.NewConnectionError: HTTPConnection(host='10.9.9.9', port=1234): " +
      "Failed to establish a new connection: [Errno 111] Connection refused";
    expect(detectProxyLoss(log(other), { proxyHosts: ["192.168.44.14"] })).toBeNull();
  });

  it("finds nothing in a clean log, an empty log, or with no candidate hosts", () => {
    expect(detectProxyLoss(log("INFO done"), { proxyHosts: ["192.168.44.14"] })).toBeNull();
    expect(detectProxyLoss("", { proxyHosts: ["192.168.44.14"] })).toBeNull();
    expect(detectProxyLoss(log(REMOTE), { proxyHosts: [] })).toBeNull();
    expect(detectProxyLoss(log(REMOTE), { proxyHosts: [undefined, null] })).toBeNull();
  });

  it("degrades to no-evidence if urllib3 rewords the message", () => {
    // Better to say nothing than to attribute a failure on a guess.
    const reworded = "ConnectionError: could not connect to 192.168.44.14:42443";
    expect(detectProxyLoss(log(reworded), { proxyHosts: ["192.168.44.14"] })).toBeNull();
  });
});

describe("proxyLossReason", () => {
  it("names the cause, the address, and that the job itself was fine", () => {
    const r = proxyLossReason({ host: "192.168.44.14", port: 42443, line: REMOTE });
    expect(r).toMatch(/manager restarted/i);
    expect(r).toContain("192.168.44.14:42443");
    expect(r).toMatch(/job itself was fine/i);
  });
});

describe("explainFailure", () => {
  it("augments the bare exit-code reason when the log explains it", () => {
    const out = explainFailure("lm-eval exited with code 1", log(REMOTE), {
      proxyHosts: ["192.168.44.14"],
      endpointUrl: "http://192.168.44.36:8000/v1",
    });
    expect(out).toContain("lm-eval exited with code 1");
    expect(out).toMatch(/manager restarted/i);
  });

  it("leaves the reason alone when there is no evidence — never invents a cause", () => {
    const reason = "lm-eval exited with code 1";
    expect(explainFailure(reason, log("INFO done"), { proxyHosts: ["192.168.44.14"] })).toBe(reason);
  });
});

describe("usesManagerProxy", () => {
  it("is true only for an accuracy run with reasoning on", () => {
    expect(usesManagerProxy({ kind: "accuracy", config: '{"reasoning":true}' })).toBe(true);
    expect(usesManagerProxy({ kind: "accuracy", config: '{"reasoning":false}' })).toBe(false);
    // runAccuracy starts the proxy on exactly this condition; throughput and
    // tool-eval runs never touch it.
    expect(usesManagerProxy({ kind: "throughput", config: '{"reasoning":true}' })).toBe(false);
    expect(usesManagerProxy({ kind: "tool-eval", config: "{}" })).toBe(false);
  });

  it("is false when the config cannot be read, rather than assumed", () => {
    // Claiming a run was proxied when we cannot tell would fail it for a reason
    // we have not established.
    expect(usesManagerProxy({ kind: "accuracy", config: "not json" })).toBe(false);
    expect(usesManagerProxy({ kind: "accuracy", config: "null" })).toBe(false);
  });
});

describe("proxyLostAtRestartReason", () => {
  it("explains the cause without inventing a port it never observed", () => {
    const r = proxyLostAtRestartReason();
    expect(r).toMatch(/did not survive the restart/i);
    expect(r).toMatch(/cancelled/i);
    // No log evidence exists in this path, so no address may appear.
    expect(r).not.toMatch(/\d+\.\d+\.\d+\.\d+/);
    expect(r).not.toMatch(/:\d{2,}/);
  });
});
