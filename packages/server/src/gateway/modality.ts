import { MODALITIES, isModality, type Modality } from "../deployments/dgxrun-catalog.js";
import { FORWARDED_PATHS, type ForwardedPath } from "./proxy.js";

/**
 * Which OpenAI surface a deployment serves, and which surface a request wants.
 *
 * The gateway used to assume one modality: every deployment answered chat or
 * embeddings, so the published name alone was enough to route. An image model
 * breaks that. It answers `/v1/images/generations` and nothing else — sending
 * it a chat completion is not an error the runtime reports, it is a request it
 * accepts and never completes, which surfaces to the client as a hang rather
 * than a refusal.
 *
 * So modality is part of the routing key, and both halves are pure: what a
 * deployment offers, and what a path requires.
 */

export type { Modality };

/** Recipe-declared modality as persisted on the deployment, mirroring how
 *  `runner` already travels in the same blob. */
interface StoredConfig {
  modality?: unknown;
}

/**
 * The modality a deployment serves, from its stored config blob.
 *
 * Total over whatever is in the column: the blob is written by an older
 * version of this server for every deployment that predates the field, and by
 * hand for `recipeYaml` deploys. Anything unreadable, absent or unrecognised
 * means `text` — the only thing every deployment written before this field
 * existed could have been.
 */
export function deploymentModality(config: string | null | undefined): Modality {
  if (!config) return "text";
  let parsed: StoredConfig;
  try {
    parsed = JSON.parse(config) as StoredConfig;
  } catch {
    return "text";
  }
  const m = parsed?.modality;
  if (m === undefined) return "text";
  if (isModality(m)) return m;
  // Falling back is right — a blob we cannot read must not take the gateway
  // down — but Principle 3 requires the fallback be observable, not silent.
  // Warned once per distinct value so a wedged deployment cannot flood the log
  // on every request it serves.
  warnUnknownModalityOnce(m);
  return "text";
}

const warnedModalities = new Set<string>();
function warnUnknownModalityOnce(value: unknown): void {
  const key = JSON.stringify(value) ?? String(value);
  if (warnedModalities.has(key)) return;
  warnedModalities.add(key);
  console.warn(
    `[gateway] deployment config declares an unrecognised modality ${key}; ` +
      `routing it as text. Expected one of ${MODALITIES.join(" | ")}.`,
  );
}

/** Test seam: the warn-once cache is module state and must not leak between tests. */
export function resetModalityWarnings(): void {
  warnedModalities.clear();
}

/**
 * The OpenAI paths the gateway forwards, and the modality each requires.
 *
 * `satisfies Record<ForwardedPath, Modality>` is the point of this shape: add a
 * fourth entry to FORWARDED_PATHS without one here and this stops compiling.
 * Without it the lookup silently yields `undefined`, every member mismatches,
 * and the caller is told the model "serves text, not undefined".
 */
export const PATH_MODALITY = {
  [FORWARDED_PATHS.chatCompletions]: "text",
  [FORWARDED_PATHS.embeddings]: "text",
  [FORWARDED_PATHS.imagesGenerations]: "image",
} as const satisfies Record<ForwardedPath, Modality>;

/**
 * The path a client should have used for this modality, named in a refusal.
 *
 * Derived from PATH_MODALITY rather than hardcoded, so it cannot drift from the
 * routing table and a modality with no serving path says so instead of guessing
 * chat.
 */
export function servingPathFor(modality: Modality): string {
  const hit = Object.entries(PATH_MODALITY).find(([, m]) => m === modality);
  return hit ? `POST ${hit[0]}` : `no path this gateway serves`;
}

/**
 * Split a pool by whether its members can serve this path.
 *
 * Returned rather than filtered in place so the caller can tell the two
 * refusals apart: a name that exists but under a different modality is an
 * operator mistake worth naming ("that is an image model"), while an empty
 * pool is a 404.
 */
export function partitionByModality<T>(
  members: T[],
  modalityOf: (m: T) => Modality,
  required: Modality,
): { matching: T[]; mismatched: T[] } {
  const matching: T[] = [];
  const mismatched: T[] = [];
  for (const m of members) (modalityOf(m) === required ? matching : mismatched).push(m);
  return { matching, mismatched };
}
