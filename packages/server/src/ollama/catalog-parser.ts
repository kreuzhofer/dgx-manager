import * as cheerio from "cheerio";

export interface OllamaCatalogEntry {
  /** Bare model name as shown by Ollama, e.g. "llama3.1" or "qwen3-vl". */
  name: string;
  /** One-line description from the library page; "" if missing. */
  description: string;
  /** "embedding" if Ollama tags it as such, otherwise "chat". */
  type: "chat" | "embedding";
  /** Tag-level size strings as displayed ("4.7GB", "274MB"). May be empty. */
  sizes: string[];
}

/**
 * Parse Ollama's https://ollama.com/library page into a flat list.
 *
 * Layout assumption: each model is rendered as an <a href="/library/{name}">
 * inside a list, containing an <h2> with the name, a description <p>, and
 * size/capability spans. We intentionally stay loose — Ollama tweaks markup
 * occasionally, and any heuristic that finds an /library/{slug} anchor with
 * an h2 child is good enough.
 *
 * Never throws: malformed input yields [].
 */
export function parseCatalogHtml(html: string): OllamaCatalogEntry[] {
  try {
    const $ = cheerio.load(html);
    const entries: OllamaCatalogEntry[] = [];
    $('a[href^="/library/"]').each((_, el) => {
      const href = $(el).attr("href") || "";
      const name = href.replace(/^\/library\//, "").split(/[/?#]/)[0].trim();
      if (!name) return;
      const heading = $(el).find("h2").first().text().trim();
      if (!heading) return;
      const description = $(el).find("p").first().text().trim();
      const sizes: string[] = [];
      $(el).find("*").each((_i, child) => {
        const t = $(child).text().trim();
        if (/^\d+(\.\d+)?\s*(GB|MB|KB|TB)$/i.test(t)) sizes.push(t);
      });
      const text = $(el).text().toLowerCase();
      const type: "chat" | "embedding" = /\bembedding\b/.test(text) ? "embedding" : "chat";
      entries.push({ name: heading || name, description, type, sizes });
    });
    // Dedupe by name (the library page sometimes lists the same slug twice).
    const seen = new Set<string>();
    return entries.filter((e) => (seen.has(e.name) ? false : seen.add(e.name)));
  } catch {
    return [];
  }
}
