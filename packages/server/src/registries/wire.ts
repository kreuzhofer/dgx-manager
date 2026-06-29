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

interface RegistryRow {
  name: string;
  url: string;
  subpath: string;
  description: string | null;
  visible: boolean;
  tuningSubpath: string | null;
  benchmarkSubpath: string | null;
  modsSubpath: string | null;
  sortOrder: number;
}

export function registryRowsToWire(rows: RegistryRow[]): RegistryWire[] {
  return [...rows]
    .sort((a, b) => a.sortOrder - b.sortOrder || a.name.localeCompare(b.name))
    .map((r) => {
      const w: RegistryWire = { name: r.name, url: r.url, subpath: r.subpath };
      if (r.description != null) w.description = r.description;
      if (r.visible === false) w.visible = false;
      if (r.tuningSubpath != null) w.tuning_subpath = r.tuningSubpath;
      if (r.benchmarkSubpath != null) w.benchmark_subpath = r.benchmarkSubpath;
      if (r.modsSubpath != null) w.mods_subpath = r.modsSubpath;
      return w;
    });
}
