/**
 * Per-deployment custom display name validation + normalization.
 *
 * The display name has two consumers:
 *   1. The dashboard deployments list (rendered as `displayName ?? model.name`).
 *   2. vLLM's `--served-model-name` flag, which sets the value that surfaces
 *      via the OpenAI API's `/v1/models` and the `model` field of completions.
 *
 * Both consumers need a value that's safe in URLs and HTTP `model` fields.
 * We restrict to `[A-Za-z0-9._:-]` (letters, digits, dot, dash, underscore,
 * colon) — the same alphabet HuggingFace model ids and Docker image tags use.
 * Spaces and slashes are rejected: spaces break clients that don't quote the
 * model field; slashes collide with REST path segments in the loadbalancer.
 */
export class DisplayNameError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DisplayNameError";
  }
}

const ALLOWED = /^[A-Za-z0-9._:-]+$/;
const MAX_LENGTH = 128;

/**
 * Normalize raw user input.
 *
 * @returns `null` when the input is null/undefined/empty/whitespace-only.
 *          Otherwise returns the trimmed string after validation passes.
 * @throws  {DisplayNameError} when the input is non-empty but contains
 *          disallowed characters or exceeds {@link MAX_LENGTH}.
 */
export function normalizeDisplayName(raw: string | null | undefined): string | null {
  if (raw === null || raw === undefined) return null;
  const trimmed = raw.trim();
  if (trimmed.length === 0) return null;
  if (trimmed.length > MAX_LENGTH) {
    throw new DisplayNameError(
      `Display name must be ${MAX_LENGTH} characters or fewer (got ${trimmed.length}).`,
    );
  }
  if (!ALLOWED.test(trimmed)) {
    throw new DisplayNameError(
      "Display name may only contain letters, digits, dot, dash, underscore, and colon " +
        "(rejected: " + JSON.stringify(trimmed) + ").",
    );
  }
  return trimmed;
}
