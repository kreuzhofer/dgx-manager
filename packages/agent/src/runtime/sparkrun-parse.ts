/** A recipe as the dashboard picker needs it: a launchable ref + display info. */
export interface SparkrunRecipeSummary {
  ref: string;          // what `sparkrun run <ref>` accepts, e.g. "@sparkrun-transitional/qwen3-1.7b-vllm"
  name: string;         // display name (falls back to ref)
  description?: string;
  runtime?: string;     // declared runtime; opaque
  registry?: string;    // e.g. "official" | "community" | custom name
  model?: string;       // HF model id (from recipe)
  minNodes: number;     // minimum node count required; defaults to 1
  tpDefault?: number;   // tensor-parallelism default; undefined when not set
  gpuMemDefault?: number; // GPU memory fraction default; undefined when not set
}

/** Coerce sparkrun's `"" | number | null | undefined` to a finite number or undefined. */
function numOrUndef(v: unknown): number | undefined {
  if (v === "" || v == null) return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
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
    try {
      const data = JSON.parse(text);
      const arr: any[] = Array.isArray(data) ? data : (data.recipes ?? []);
      return arr.map((r) => ({
        ref: String(r.ref ?? r.id ?? r.name),
        name: String(r.name ?? r.ref ?? r.id),
        description: r.description || undefined,
        runtime: r.runtime,
        registry: r.registry,
        model: r.model ? String(r.model) : undefined,
        minNodes: Number(r.min_nodes ?? 1),
        tpDefault: numOrUndef(r.tp),
        gpuMemDefault: numOrUndef(r.gpu_mem),
      }));
    } catch {
      // fall through to text-line parsing below
    }
  }
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l && !/^name\b/i.test(l) && !/^-+$/.test(l))
    .map((l) => {
      const [ref, ...rest] = l.split(/\s{2,}|\t/);
      return { ref: ref.trim(), name: ref.trim(), description: rest.join(" ").trim() || undefined, minNodes: 1 };
    });
}
