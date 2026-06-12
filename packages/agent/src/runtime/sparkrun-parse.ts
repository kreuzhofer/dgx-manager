/** A recipe as the dashboard picker needs it: a launchable ref + display info. */
export interface SparkrunRecipeSummary {
  ref: string;          // what `sparkrun run <ref>` accepts, e.g. "@sparkrun-transitional/qwen3-1.7b-vllm"
  name: string;         // display name (falls back to ref)
  description?: string;
  runtime?: string;     // declared runtime; opaque
  registry?: string;    // e.g. "official" | "community" | custom name
}

/**
 * Parse `sparkrun list` output. If sparkrun emits JSON (the captured fixture is
 * a JSON array), the JSON branch is used; otherwise the tabular text branch
 * parses columns. Both paths are exercised by fixture tests.
 */
export function parseSparkrunList(raw: string): SparkrunRecipeSummary[] {
  const text = raw.trim();
  if (!text) return [];
  if (text.startsWith("[") || text.startsWith("{")) {
    const data = JSON.parse(text);
    const arr: any[] = Array.isArray(data) ? data : (data.recipes ?? []);
    return arr.map((r) => ({
      ref: String(r.ref ?? r.id ?? r.name),
      name: String(r.name ?? r.ref ?? r.id),
      description: r.description || undefined,
      runtime: r.runtime,
      registry: r.registry,
    }));
  }
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !/^name\b/i.test(l) && !/^-+$/.test(l))
    .map((l) => {
      const [ref, ...rest] = l.split(/\s{2,}|\t/);
      return { ref: ref.trim(), name: ref.trim(), description: rest.join(" ").trim() || undefined };
    });
}
