export interface RegistryWire {
  name: string;
  url: string;
  subpath: string;
  description?: string;
  visible?: boolean;
  tuning_subpath?: string;
  benchmark_subpath?: string;
  mods_subpath?: string;
}

/** Double-quote a scalar with YAML-safe escaping (backslash, quote, newline). */
function q(s: string): string {
  return '"' + s.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n") + '"';
}

/** Render a sparkrun-compatible registries.yaml. Optional null/undefined fields
 *  are omitted; `visible` is emitted only when false (sparkrun defaults to true). */
export function renderRegistriesYaml(registries: RegistryWire[]): string {
  if (registries.length === 0) return "registries: []\n";
  const lines: string[] = ["registries:"];
  for (const r of registries) {
    lines.push(`- name: ${q(r.name)}`);
    lines.push(`  url: ${q(r.url)}`);
    lines.push(`  subpath: ${q(r.subpath)}`);
    if (r.description != null) lines.push(`  description: ${q(r.description)}`);
    if (r.visible === false) lines.push(`  visible: false`);
    if (r.tuning_subpath != null) lines.push(`  tuning_subpath: ${q(r.tuning_subpath)}`);
    if (r.benchmark_subpath != null) lines.push(`  benchmark_subpath: ${q(r.benchmark_subpath)}`);
    if (r.mods_subpath != null) lines.push(`  mods_subpath: ${q(r.mods_subpath)}`);
  }
  return lines.join("\n") + "\n";
}
