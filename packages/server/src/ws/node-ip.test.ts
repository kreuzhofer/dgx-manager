import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import { fc } from "@fast-check/vitest";
import { isValidIpv4, resolveNodeIp } from "./node-ip.js";

describe("isValidIpv4", () => {
  it("accepts well-formed dotted-quad addresses", () => {
    expect(isValidIpv4("192.168.44.36")).toBe(true);
    expect(isValidIpv4("10.0.0.1")).toBe(true);
    expect(isValidIpv4("0.0.0.0")).toBe(true);
    expect(isValidIpv4("255.255.255.255")).toBe(true);
  });

  it("rejects empty, malformed, out-of-range, and non-IPv4 values", () => {
    expect(isValidIpv4(null)).toBe(false);
    expect(isValidIpv4(undefined)).toBe(false);
    expect(isValidIpv4("")).toBe(false);
    expect(isValidIpv4("nope")).toBe(false);
    expect(isValidIpv4("256.1.1.1")).toBe(false);
    expect(isValidIpv4("1.2.3")).toBe(false);
    expect(isValidIpv4("1.2.3.4.5")).toBe(false);
    expect(isValidIpv4("::1")).toBe(false);
  });
});

describe("resolveNodeIp", () => {
  // The agent-supplied advertise IP (NODE_ADVERTISE_IP) wins when valid — that's
  // the whole point: a node co-located with the server on a docker bridge would
  // otherwise be registered with the bridge gateway as its WS source IP.
  it("prefers a valid advertise IP over the socket source IP", () => {
    expect(resolveNodeIp("192.168.44.36", "172.19.0.1")).toBe("192.168.44.36");
  });

  it("falls back to the socket IP when advertise is missing or invalid", () => {
    expect(resolveNodeIp(null, "192.168.44.37")).toBe("192.168.44.37");
    expect(resolveNodeIp(undefined, "192.168.44.37")).toBe("192.168.44.37");
    expect(resolveNodeIp("garbage", "192.168.44.37")).toBe("192.168.44.37");
    expect(resolveNodeIp("999.0.0.1", "192.168.44.37")).toBe("192.168.44.37");
  });

  it("returns null when neither source is a valid IPv4", () => {
    expect(resolveNodeIp(null, null)).toBeNull();
    expect(resolveNodeIp("", "bad")).toBeNull();
  });

  // Invariant: a valid advertise IP is returned verbatim regardless of the
  // socket IP (so the docker-bridge source can never override an explicit pin).
  fcIt.prop([
    fc.tuple(fc.nat({ max: 255 }), fc.nat({ max: 255 }), fc.nat({ max: 255 }), fc.nat({ max: 255 })),
    fc.option(fc.string(), { nil: undefined }),
  ])("valid advertise IP always wins", (octets, remote) => {
    const adv = octets.join(".");
    expect(resolveNodeIp(adv, remote)).toBe(adv);
  });
});
