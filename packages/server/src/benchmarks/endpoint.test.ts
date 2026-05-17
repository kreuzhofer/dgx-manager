import { describe, expect, it } from "vitest";
import { deploymentEndpointUrl } from "./endpoint.js";

describe("deploymentEndpointUrl", () => {
  it("returns http://<ip>:<port> for a running deployment", () => {
    expect(
      deploymentEndpointUrl({
        port: 8000,
        node: { ipAddress: "192.168.1.10" },
      }),
    ).toBe("http://192.168.1.10:8000");
  });

  it("throws when port is missing", () => {
    expect(() =>
      deploymentEndpointUrl({
        port: null,
        node: { ipAddress: "192.168.1.10" },
      }),
    ).toThrow(/port/);
  });

  it("throws when node ipAddress is missing", () => {
    expect(() =>
      deploymentEndpointUrl({
        port: 8000,
        node: { ipAddress: null },
      }),
    ).toThrow(/ipAddress/);
  });

  it("throws when node is null", () => {
    expect(() =>
      deploymentEndpointUrl({ port: 8000, node: null }),
    ).toThrow(/node/);
  });
});
