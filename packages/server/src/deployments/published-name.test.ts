import { describe, expect, it, vi } from "vitest";
import { resolvePublishedName } from "./published-name.js";

/**
 * A fetch stub that answers an OpenAI `/v1/models` request with one served id.
 * Mirrors the shape `resolveServedModelName` already expects.
 */
function servingFetch(id: string) {
  return vi.fn(async () =>
    new Response(JSON.stringify({ data: [{ id }] }), { status: 200 }),
  ) as unknown as typeof fetch;
}

const unreachableFetch = vi.fn(async () => {
  throw new Error("ECONNREFUSED");
}) as unknown as typeof fetch;

describe("resolvePublishedName", () => {
  // A pinned runtime is authoritative about its own name: a recipe may set
  // `served_model_name` to something that is neither the displayName nor the
  // catalog name, and guessing produces an unexplained 404 for the caller.
  it("asks a pinned runtime what it serves, over any local guess", async () => {
    const name = await resolvePublishedName(
      {
        runtime: "vllm",
        modelName: "@dgxrun/glm-5.2-quanttrio-unpruned-dcp2-320k",
        displayName: "glm-5.2",
        endpointUrl: "http://192.168.44.36:8000",
      },
      servingFetch("something-else-entirely"),
    );
    expect(name).toBe("something-else-entirely");
  });

  it("falls back to the display name when a pinned endpoint is unreachable", async () => {
    const name = await resolvePublishedName(
      {
        runtime: "vllm",
        modelName: "@dgxrun/glm-5.2-long",
        displayName: "glm-5.2",
        endpointUrl: "http://192.168.44.36:8000",
      },
      unreachableFetch,
    );
    expect(name).toBe("glm-5.2");
  });

  it("falls back to the catalog name when a pinned deployment has no display name", async () => {
    const name = await resolvePublishedName(
      {
        runtime: "vllm",
        modelName: "@dgxrun/glm-5.2-long",
        displayName: null,
        endpointUrl: "http://192.168.44.36:8000",
      },
      unreachableFetch,
    );
    expect(name).toBe("@dgxrun/glm-5.2-long");
  });

  // An allocation-inducing runtime hosts every model pulled on that node behind
  // one address, so its /v1/models list is not about this deployment at all —
  // probing it would return an arbitrary sibling. The model tag is the name.
  it("takes an allocation-inducing runtime's name from the model tag, without probing", async () => {
    const probe = servingFetch("some-other-model-on-the-node");
    const name = await resolvePublishedName(
      {
        runtime: "ollama",
        modelName: "qwen3-embedding:8b",
        displayName: null,
        endpointUrl: "http://192.168.44.15:11434",
      },
      probe,
    );
    expect(name).toBe("qwen3-embedding:8b");
    expect(probe).not.toHaveBeenCalled();
  });

  // The tag is what the runtime answers to; an operator alias cannot change
  // that, so it must not be published as though it could.
  it("ignores a display name on an allocation-inducing runtime", async () => {
    const probe = servingFetch("unused");
    const name = await resolvePublishedName(
      {
        runtime: "ollama",
        modelName: "qwen3-embedding:8b",
        displayName: "embed",
        endpointUrl: "http://192.168.44.15:11434",
      },
      probe,
    );
    expect(name).toBe("qwen3-embedding:8b");
    expect(probe).not.toHaveBeenCalled();
  });

  it("does not probe when the deployment has no endpoint yet", async () => {
    const probe = servingFetch("nope");
    const name = await resolvePublishedName(
      { runtime: "vllm", modelName: "m", displayName: "alias", endpointUrl: null },
      probe,
    );
    expect(name).toBe("alias");
    expect(probe).not.toHaveBeenCalled();
  });
});
