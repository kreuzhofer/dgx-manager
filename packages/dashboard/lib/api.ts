const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

export async function apiFetch<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    cache: "no-store",
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `API error: ${res.status}`);
  }
  return res.json();
}

export interface ReseedKnownHostsResult {
  trustedIps: string[];
  perNode: Array<{ nodeId: string; host: string; ipsSeeded: number; ok: boolean; error?: string }>;
}

/**
 * POST /api/cluster/reseed-known-hosts — re-seeds the cross-node SSH known_hosts
 * trust mesh. HTTP 502 (no nodes seeded) is treated as a valid report body, not
 * an error. Only throws on other non-OK statuses.
 */
export async function reseedKnownHosts(): Promise<ReseedKnownHostsResult> {
  const res = await fetch(`${API_BASE}/api/cluster/reseed-known-hosts`, {
    method: "POST",
    cache: "no-store",
    headers: { "Content-Type": "application/json" },
  });
  if (!res.ok && res.status !== 502) {
    const body = await res.json().catch(() => ({}));
    throw new Error((body as { error?: string }).error || `Reseed failed: ${res.status}`);
  }
  return res.json();
}

export interface SparkrunRegistry {
  id: string;
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

export type NewRegistry = Pick<SparkrunRegistry, "name" | "url" | "subpath"> &
  Partial<Pick<SparkrunRegistry, "description" | "visible">>;

export const listRegistries = () => apiFetch<SparkrunRegistry[]>("/api/registries");

export const createRegistry = (body: NewRegistry) =>
  apiFetch<SparkrunRegistry>("/api/registries", { method: "POST", body: JSON.stringify(body) });

export const updateRegistry = (id: string, body: Partial<NewRegistry>) =>
  apiFetch<SparkrunRegistry>(`/api/registries/${id}`, { method: "PATCH", body: JSON.stringify(body) });

export const deleteRegistry = (id: string) =>
  apiFetch<{ status: string }>(`/api/registries/${id}`, { method: "DELETE" });

export async function apiUpload<T>(path: string, formData: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: formData,
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error || `API error: ${res.status}`);
  }
  return res.json();
}
