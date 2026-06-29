export interface RegistryInput {
  name: string;
  url: string;
  subpath: string;
  description?: string | null;
  visible?: boolean;
  tuningSubpath?: string | null;
  benchmarkSubpath?: string | null;
  modsSubpath?: string | null;
  sortOrder?: number;
}

export type ValidationResult =
  | { ok: true; value: RegistryInput }
  | { ok: false; error: string };

const NAME_RE = /^[a-z0-9-]+$/;
const URL_RE = /^(https?:\/\/|git:\/\/|ssh:\/\/|git@)/;

function badPath(s: string): boolean {
  return s.length === 0 || s.startsWith("/") || s.split("/").includes("..");
}

export function validateRegistry(input: unknown): ValidationResult {
  if (typeof input !== "object" || input === null) return { ok: false, error: "Body must be an object" };
  const i = input as Record<string, unknown>;

  if (typeof i.name !== "string" || !NAME_RE.test(i.name))
    return { ok: false, error: "name must match ^[a-z0-9-]+$" };
  if (typeof i.url !== "string" || !URL_RE.test(i.url))
    return { ok: false, error: "url must be an http(s)/git/ssh URL" };
  if (typeof i.subpath !== "string" || badPath(i.subpath))
    return { ok: false, error: "subpath must be a non-empty relative path (no '..' or leading '/')" };

  for (const k of ["description", "tuningSubpath", "benchmarkSubpath", "modsSubpath"] as const) {
    if (i[k] != null && typeof i[k] !== "string") return { ok: false, error: `${k} must be a string` };
  }
  for (const k of ["tuningSubpath", "benchmarkSubpath", "modsSubpath"] as const) {
    if (typeof i[k] === "string" && badPath(i[k] as string)) return { ok: false, error: `${k} must be a relative path` };
  }
  if (i.visible != null && typeof i.visible !== "boolean") return { ok: false, error: "visible must be a boolean" };
  if (i.sortOrder != null && typeof i.sortOrder !== "number") return { ok: false, error: "sortOrder must be a number" };

  return {
    ok: true,
    value: {
      name: i.name,
      url: i.url,
      subpath: i.subpath,
      description: (i.description as string | null) ?? null,
      visible: i.visible == null ? true : (i.visible as boolean),
      tuningSubpath: (i.tuningSubpath as string | null) ?? null,
      benchmarkSubpath: (i.benchmarkSubpath as string | null) ?? null,
      modsSubpath: (i.modsSubpath as string | null) ?? null,
      sortOrder: (i.sortOrder as number | undefined) ?? 0,
    },
  };
}
