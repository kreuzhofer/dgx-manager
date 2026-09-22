import { MODALITIES, type Modality } from "../deployments/dgxrun-catalog.js";

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
  return (MODALITIES as readonly unknown[]).includes(m) ? (m as Modality) : "text";
}

/** The OpenAI paths the gateway forwards, and the modality each requires. */
export const PATH_MODALITY = {
  "/v1/chat/completions": "text",
  "/v1/embeddings": "text",
  "/v1/images/generations": "image",
} as const satisfies Record<string, Modality>;

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
