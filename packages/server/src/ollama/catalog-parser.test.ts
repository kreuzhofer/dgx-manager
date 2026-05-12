import { describe, it, expect } from "vitest";
import { it as fcIt } from "@fast-check/vitest";
import fc from "fast-check";
import { parseCatalogHtml } from "./catalog-parser.js";

const SAMPLE_LIBRARY_HTML = `
<html><body>
  <ul role="list">
    <li>
      <a href="/library/llama3.1">
        <h2>llama3.1</h2>
        <p>Meta's Llama 3.1 family.</p>
        <span x-test="size">4.7GB</span>
        <span x-test="size">40GB</span>
        <span x-test="pulls">1.2M</span>
      </a>
    </li>
    <li>
      <a href="/library/nomic-embed-text">
        <h2>nomic-embed-text</h2>
        <p>A high-performing open embedding model.</p>
        <span x-test="size">274MB</span>
        <span x-test="capability">embedding</span>
      </a>
    </li>
  </ul>
</body></html>`;

describe("parseCatalogHtml", () => {
  it("extracts entries from a library page", () => {
    const entries = parseCatalogHtml(SAMPLE_LIBRARY_HTML);
    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      name: "llama3.1",
      description: "Meta's Llama 3.1 family.",
      type: "chat",
    });
    expect(entries[0].sizes).toEqual(["4.7GB", "40GB"]);
    expect(entries[1]).toMatchObject({ name: "nomic-embed-text", type: "embedding" });
  });

  it("returns [] on malformed HTML", () => {
    expect(parseCatalogHtml("")).toEqual([]);
    expect(parseCatalogHtml("<p>no models here</p>")).toEqual([]);
  });

  it("skips entries with no name", () => {
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
