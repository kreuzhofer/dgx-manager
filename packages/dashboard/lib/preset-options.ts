/**
 * Preset ids available to filter by, derived from the runs on screen.
 *
 * The benchmarks page previously hardcoded this list as static <option> tags, and
 * it drifted: acc-gpqa-diamond-full-longgen existed server-side but could never
 * be selected, because nobody updated the markup when the preset was added.
 *
 * Deriving it from the runs is both drift-proof and the correct domain for a
 * filter - the options are exactly the values that can match something. It also
 * keeps working for runs whose preset has since been renamed or removed, which a
 * list fetched from the presets endpoint would silently drop.
 */
export function presetOptions(runs: Array<{ presetId: string | null }>): string[] {
  return [...new Set(runs.map((r) => r.presetId).filter((p): p is string => !!p))].sort();
}
