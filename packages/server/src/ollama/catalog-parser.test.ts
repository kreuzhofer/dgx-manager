import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import fc from "fast-check";
import { parseCatalogHtml } from "./catalog-parser.js";

/**
 * Fixture mirrors the real markup of ollama.com/library (sampled 2026-05).
 * Four cards cover the cases the parser must handle:
 *
 *   - llama3.1     : plain chat model with sizes + a "tools" capability.
 *   - nomic-embed-text : embedding model — has the "embedding" capability
 *                      badge and a single parameter size.
 *   - gemma3       : has BOTH local sizes AND a cloud marker — must be kept,
 *                    cloud span must be ignored, sizes intact.
 *   - kimi-k2      : cloud-only — zero x-test-size entries. Must be dropped.
 */
const SAMPLE_LIBRARY_HTML = `
<html><body>
  <ul role="list">
    <li>
      <a href="/library/llama3.1" class="group w-full space-y-5">
        <div x-test-model-title title="llama3.1" class="flex flex-col">
          <h2><div class="flex"><span class="truncate">llama3.1</span></div></h2>
          <p>Meta's Llama 3.1 family.</p>
        </div>
        <div>
          <span x-test-capability>tools</span>
          <span x-test-size>8b</span>
          <span x-test-size>70b</span>
          <span x-test-size>405b</span>
          <span class="flex items-center" title="Nov 30, 2024 10:34 PM UTC">
            <span x-test-updated>11 months ago</span>
          </span>
        </div>
      </a>
    </li>
    <li>
      <a href="/library/nomic-embed-text" class="group w-full space-y-5">
        <div x-test-model-title title="nomic-embed-text">
          <h2><span>nomic-embed-text</span></h2>
          <p>Open embedding model.</p>
        </div>
        <div>
          <span x-test-capability>embedding</span>
          <span x-test-size>137m</span>
          <span class="flex items-center" title="Feb 21, 2024 5:26 PM UTC">
            <span x-test-updated>2 years ago</span>
          </span>
        </div>
      </a>
    </li>
    <li>
      <a href="/library/gemma3" class="group w-full space-y-5">
        <div x-test-model-title title="gemma3">
          <h2><span>gemma3</span></h2>
          <p>Gemma 3, Google's open model.</p>
        </div>
        <div>
          <span x-test-capability>vision</span>
          <span x-test-size>270m</span>
          <span x-test-size>1b</span>
          <span x-test-size>4b</span>
          <span x-test-size>12b</span>
          <span x-test-size>27b</span>
          <span class="bg-cyan-50 text-cyan-500">cloud</span>
        </div>
      </a>
    </li>
    <li>
      <a href="/library/kimi-k2" class="group w-full space-y-5">
        <div x-test-model-title title="kimi-k2">
          <h2><span>kimi-k2</span></h2>
          <p>Cloud-only Kimi K2.</p>
        </div>
        <div>
          <span x-test-capability>tools</span>
          <span class="bg-cyan-50 text-cyan-500">cloud</span>
        </div>
      </a>
    </li>
    <li>
      <a href="/library/wizardlm" class="group w-full space-y-5">
        <div x-test-model-title title="wizardlm">
          <h2><span>wizardlm</span></h2>
          <p>Older local model without a size badge on the index card.</p>
        </div>
        <div>
        </div>
      </a>
    </li>
  </ul>
</body></html>`;

describe("parseCatalogHtml", () => {
  it("extracts locally-pullable entries from a library page", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    const names = entries.map((e) => e.name);
    expect(names).toEqual(["llama3.1", "nomic-embed-text", "gemma3", "wizardlm"]);

    const llama = entries.find((e) => e.name === "llama3.1")!;
    expect(llama.description).toBe("Meta's Llama 3.1 family.");
    expect(llama.type).toBe("chat");
    expect(llama.sizes).toEqual(["8b", "70b", "405b"]);
    expect(llama.capabilities).toEqual(["tools"]);
  });

  it("classifies the embedding capability as type=embedding", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    const nomic = entries.find((e) => e.name === "nomic-embed-text")!;
    expect(nomic.type).toBe("embedding");
    expect(nomic.capabilities).toContain("embedding");
    expect(nomic.sizes).toEqual(["137m"]);
  });

  it("keeps models that have BOTH local sizes and a cloud marker", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    const gemma = entries.find((e) => e.name === "gemma3")!;
    expect(gemma).toBeDefined();
    expect(gemma.sizes).toEqual(["270m", "1b", "4b", "12b", "27b"]);
    // The cloud span must NOT leak into sizes or capabilities.
    expect(gemma.sizes).not.toContain("cloud");
    expect(gemma.capabilities).not.toContain("cloud");
  });

  it("drops cloud-only models (cloud marker + no x-test-size)", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    expect(entries.map((e) => e.name)).not.toContain("kimi-k2");
  });

  it("parses updatedAt from the card's date tooltip", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    const llama = entries.find((e) => e.name === "llama3.1")!;
    expect(llama.updatedAt).toBe(new Date("Nov 30, 2024 10:34 PM UTC").toISOString());
    const nomic = entries.find((e) => e.name === "nomic-embed-text")!;
    expect(nomic.updatedAt).toBe(new Date("Feb 21, 2024 5:26 PM UTC").toISOString());
  });

  it("returns updatedAt=null when no date tooltip is present", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    const wizard = entries.find((e) => e.name === "wizardlm")!;
    expect(wizard.updatedAt).toBeNull();
  });

  it("keeps local-only models that happen to have no size badge", () => {
    // Embedding models like nomic-embed-text and older single-size models
    // (wizardlm, openhermes, goliath, …) render no x-test-size span on the
    // library index. They must NOT be dropped — they're locally pullable.
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    const wizard = entries.find((e) => e.name === "wizardlm");
    expect(wizard).toBeDefined();
    expect(wizard!.sizes).toEqual([]);
  });

  it("returns [] on malformed HTML", () => {
    expect(parseCatalogHtml("")).toEqual([]);
    expect(parseCatalogHtml("<p>no models here</p>")).toEqual([]);
  });

  it("skips entries with no /library/{slug} href", () => {
    expect(parseCatalogHtml(`<a href="/library/"><h2></h2></a>`)).toEqual([]);
  });

  /**
   * Property: feeding the parser any string never throws and always returns
   * an array — the catalog refresh path treats parse failure as "no models"
   * not as an error, so the parser must not surface exceptions to callers.
   */
  fcIt.prop([fc.string()])("never throws on arbitrary input", (s) => {
    expect(() => parseCatalogHtml(s)).not.toThrow();
    expect(Array.isArray(parseCatalogHtml(s))).toBe(true);
  });
});
