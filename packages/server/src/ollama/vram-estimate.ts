/**
 * Parse an Ollama parameter-size tag ("8b", "70b", "e2b", "270m") into a VRAM
 * estimate in MB. Ollama defaults to Q4_K_M quantization which lands around
 * 0.55 GB per billion params; we use that as the upper-bound estimate for
 * admission. The agent reports `vramActual` once nvidia-smi has post-load
 * truth, so the estimate is only load-bearing for the refuse-before-launch
 * check.
 *
 * Recognized suffixes:
 *   - "b"  — billions of params
 *   - "m"  — millions of params (e.g. "270m")
 *   - "e2b"/"e4b" — Gemma "effective N billion" notation; treated as Nb.
 *
 * Returns null if the input doesn't match the expected shape.
 */

const Q4_GB_PER_BILLION = 0.55;
const MB_PER_GB = 1024;

export function ollamaVramEstimateMB(rawSize: string | null | undefined): number | null {
  if (!rawSize || typeof rawSize !== "string") return null;
  const m = rawSize.trim().toLowerCase().match(/^e?(\d+(?:\.\d+)?)\s*([bm])$/);
  if (!m) return null;
  const value = parseFloat(m[1]);
  if (!Number.isFinite(value)) return null;
  const billionsParams = m[2] === "b" ? value : value / 1000;
  return Math.round(billionsParams * Q4_GB_PER_BILLION * MB_PER_GB);
}
