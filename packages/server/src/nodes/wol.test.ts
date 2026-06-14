import { describe, it, expect } from "vitest";
import { it as fcit } from "@fast-check/vitest";
import * as fc from "fast-check";
import { isValidMac, buildMagicPacket, broadcastFor } from "./wol.js";

describe("isValidMac", () => {
  it("accepts lowercase colon-separated MACs", () => {
    expect(isValidMac("aa:bb:cc:dd:ee:ff")).toBe(true);
  });
  it("rejects junk", () => {
    expect(isValidMac("nope")).toBe(false);
    expect(isValidMac("aa:bb:cc:dd:ee")).toBe(false);
  });
});

describe("broadcastFor", () => {
  it("computes the /24 broadcast by default", () => {
    expect(broadcastFor("192.168.44.41")).toBe("192.168.44.255");
  });
  it("supports an explicit /16 prefix", () => {
    expect(broadcastFor("192.168.44.41", 16)).toBe("192.168.255.255");
  });
});

// PROPERTY: a WOL magic packet is always exactly 102 bytes — a 6-byte 0xFF
// sync stream followed by the 6-byte MAC repeated 16 times.
fcit.prop([
  fc.array(fc.integer({ min: 0, max: 255 }), { minLength: 6, maxLength: 6 }),
])("buildMagicPacket is 6x0xFF + 16x MAC", (octets) => {
  const mac = octets.map((o) => o.toString(16).padStart(2, "0")).join(":");
  const pkt = buildMagicPacket(mac);
  expect(pkt.length).toBe(102);
  for (let i = 0; i < 6; i++) expect(pkt[i]).toBe(0xff);
  for (let rep = 0; rep < 16; rep++) {
    for (let b = 0; b < 6; b++) {
      expect(pkt[6 + rep * 6 + b]).toBe(octets[b]);
    }
  }
});
