import { describe, expect, it } from "vitest";
import {
  deploymentEndpointUrl,
  firstServedModelId,
  resolveServedModelName,
} from "./endpoint.js";

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

describe("firstServedModelId", () => {
  it("returns the first model id from a vLLM /v1/models body", () => {
    const body = JSON.stringify({
      object: "list",
      data: [{ id: "glm-5.2", object: "model" }],
    });
    expect(firstServedModelId(body)).toBe("glm-5.2");
  });

  it("returns null for empty data", () => {
    expect(firstServedModelId(JSON.stringify({ data: [] }))).toBeNull();
  });

  it("returns null for a non-string / missing id", () => {
    expect(firstServedModelId(JSON.stringify({ data: [{ id: 123 }] }))).toBeNull();
    expect(firstServedModelId(JSON.stringify({ data: [{}] }))).toBeNull();
  });

  it("returns null for malformed JSON", () => {
    expect(firstServedModelId("not json")).toBeNull();
    expect(firstServedModelId("")).toBeNull();
  });
});

describe("resolveServedModelName", () => {
  const ok = (body: string): typeof fetch =>
    (async () => new Response(body, { status: 200 })) as unknown as typeof fetch;

  it("uses the endpoint's actual served id, not the fallback", async () => {
    const fetchImpl = ok(JSON.stringify({ data: [{ id: "glm-5.2" }] }));
    await expect(
      resolveServedModelName("http://x:8000/v1", "@catalog/name", fetchImpl),
    ).resolves.toBe("glm-5.2");
  });

  it("falls back when the endpoint errors", async () => {
    const fetchImpl = (async () => {
      throw new Error("ECONNREFUSED");
    }) as unknown as typeof fetch;
    await expect(
      resolveServedModelName("http://x:8000/v1", "@catalog/name", fetchImpl),
    ).resolves.toBe("@catalog/name");
  });

  it("falls back on a non-2xx response", async () => {
    const fetchImpl = (async () =>
      new Response("nope", { status: 503 })) as unknown as typeof fetch;
    await expect(
      resolveServedModelName("http://x:8000/v1", "@catalog/name", fetchImpl),
    ).resolves.toBe("@catalog/name");
  });

  it("falls back when the body has no usable id", async () => {
    const fetchImpl = ok(JSON.stringify({ data: [] }));
    await expect(
      resolveServedModelName("http://x:8000/v1", "@catalog/name", fetchImpl),
    ).resolves.toBe("@catalog/name");
  });
});
