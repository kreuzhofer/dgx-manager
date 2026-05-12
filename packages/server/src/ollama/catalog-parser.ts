import * as cheerio from "cheerio";

export interface OllamaCatalogEntry {
  /** Bare model name as shown by Ollama, e.g. "llama3.1" or "qwen3-vl". */
  name: string;
  /** One-line description from the library page; "" if missing. */
  description: string;
  /** "embedding" if Ollama tags the card with the embedding capability, otherwise "chat". */
  type: "chat" | "embedding";
  /**
   * Locally-pullable parameter sizes Ollama lists for this model. Each entry
   * is the raw tag suffix as displayed: "8b", "70b", "405b", "e2b", "270m".
   * Combine with `name` as `${name}:${size}` to form the pull tag used by
   * `ollama pull`.
   */
  sizes: string[];
  /**
   * Capability badges Ollama attaches to the card: "tools", "thinking",
   * "vision", "embedding", "audio". Useful for UI badges and filtering.
   */
  capabilities: string[];
  /**
   * ISO timestamp of the last tag update for this model, parsed from the
   * card's `x-test-updated` tooltip ("Nov 30, 2024 10:34 PM UTC"). Null when
   * the card has no recognizable date — the UI sorts those last in newest-
   * first views.
   */
  updatedAt: string | null;
}

/**
 * Parse Ollama's https://ollama.com/library page into a flat list of
 * locally-pullable models.
 *
 * Selectors we rely on (verified against ollama.com markup, 2026-05):
 *   - Each model card is an `<a href="/library/{slug}">` wrapping the card body.
 *   - `<div x-test-model-title title="{slug}">` carries the canonical name.
 *   - `<span x-test-size>` spans hold parameter sizes ("8b", "70b", "e2b").
 *   - `<span x-test-capability>` spans hold capability badges ("tools",
 *     "thinking", "vision", "embedding", "audio").
 *   - A separate `<span class="... bg-cyan-50 ...">cloud</span>` flags cards
 *     that *also* offer a cloud variant — orthogonal to the parameter sizes.
 *
 * Cloud-only filter: we deploy locally, so we drop any card that carries the
 * cloud marker AND has zero `x-test-size` entries. Cards that offer BOTH
 * cloud and local sizes (e.g. `gemma3`, `gpt-oss`) are kept with the cloud
 * marker ignored. Cards with no sizes and no cloud marker are kept too —
 * embedding models like `nomic-embed-text` and older single-size models
 * (`openhermes`, `wizardlm`) don't show parameter sizes on the index card
 * but are still locally pullable.
 *
 * Never throws: malformed input yields []. Returning [] on a 200 response is
 * the fetcher's signal that Ollama changed their markup and the UI should
 * surface a "parser stale" message.
 */
export function parseCatalogHtml(html: string): OllamaCatalogEntry[] {
  try {
    const $ = cheerio.load(html);
    const entries: OllamaCatalogEntry[] = [];
    $('a[href^="/library/"]').each((_, el) => {
      const href = $(el).attr("href") || "";
      const slug = href.replace(/^\/library\//, "").split(/[/?#]/)[0].trim();
      if (!slug) return;

      const titleAttr = $(el).find("[x-test-model-title]").first().attr("title")?.trim();
      const name = titleAttr || slug;

      const description = $(el).find("p").first().text().trim();

      const sizes: string[] = [];
      $(el).find("[x-test-size]").each((_i, sizeEl) => {
        const t = $(sizeEl).text().trim();
        if (t) sizes.push(t);
      });

      // Cloud-only filter: a card is cloud-only when it carries the cloud
      // marker AND has no `x-test-size` entries. Catches glm-5, kimi-k2,
      // deepseek-v4-pro etc. without dropping embedding models like
      // nomic-embed-text that legitimately have no size badges.
      const hasCloudMarker =
        $(el).find('span.bg-cyan-50, span.text-cyan-500').length > 0 ||
        $(el).find('span').toArray().some((s) => $(s).text().trim().toLowerCase() === "cloud");
      if (sizes.length === 0 && hasCloudMarker) return;

      const capabilities: string[] = [];
      $(el).find("[x-test-capability]").each((_i, capEl) => {
        const t = $(capEl).text().trim().toLowerCase();
        if (t) capabilities.push(t);
      });

      const type: "chat" | "embedding" = capabilities.includes("embedding")
        ? "embedding"
        : "chat";

      // Updated-at: Ollama wraps the date inside the `x-test-updated` span's
      // parent, in a `title="Nov 30, 2024 10:34 PM UTC"` attribute. Walk the
      // card's [title] elements and pick the first one whose value parses as
      // a real date — the x-test-model-title's title is the bare slug and
      // intentionally fails the Date.parse check.
      let updatedAt: string | null = null;
      $(el).find("[title]").each((_i, t) => {
        if (updatedAt) return;
        const v = $(t).attr("title");
        if (!v) return;
        const ms = Date.parse(v);
        if (!Number.isNaN(ms)) updatedAt = new Date(ms).toISOString();
      });

      entries.push({ name, description, type, sizes, capabilities, updatedAt });
    });

    // Dedupe by name (defensive — the library page renders each card once,
    // but the same slug can appear under multiple sort orders or filters).
    const seen = new Set<string>();
    return entries.filter((e) => (seen.has(e.name) ? false : seen.add(e.name)));
  } catch {
    return [];
  }
}
