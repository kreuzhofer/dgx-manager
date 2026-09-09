/**
 * Deciding whether a finished benchmark process counts as a COMPLETED run.
 *
 * A terminal status is a claim about OUTPUT, not about an exit code: "the
 * process ended" and "the run produced data" are different facts, and only the
 * second is comparable. `finalizeThroughput` used to complete on the exit code
 * alone, so a run whose every request failed — but whose CLI still exited 0 —
 * was recorded as successful with zero rows. Two such runs then compare as
 * perfect agreement, which is indistinguishable from a flawless reproduction
 * at the comparison layer (#95).
 *
 * Pure so the rule can be stated as an invariant and property-tested, rather
 * than inferred from three Prisma-coupled finalizers that disagreed with each
 * other. The Prisma-coupled half lives in execute.ts.
 */

export type FinalizeOutcome =
  | { kind: "complete" }
  | { kind: "fail"; reason: string };

export interface FinalizeInput {
  /** Tool name for the failure message, e.g. "llama-benchy". */
  tool: string;
  /** Process exit code; `null` when the child was killed rather than exiting. */
  exitCode: number | null;
  /** Did the runner produce a summary object at all? */
  hasSummary: boolean;
  /**
   * How many result rows the run produced. Omit for a runner that has no row
   * set — its output is the summary, and an empty-rows check is meaningless.
   */
  rowCount?: number;
  /** A parse failure the runner reported despite exiting 0. */
  parseError?: string | null;
}

/**
 * Complete only when the process exited 0 AND produced something to compare.
 *
 * Precedence matters and is deliberate: a run that has a summary is completed
 * even if the runner also reported a parse error, preserving the behaviour
 * `finalizeAccuracy` already had. The parse error is only surfaced when there
 * is no summary to fall back on.
 */
export function decideFinalize(input: FinalizeInput): FinalizeOutcome {
  const { tool, exitCode, hasSummary, rowCount, parseError } = input;

  if (exitCode !== 0) {
    return { kind: "fail", reason: `${tool} exited with code ${exitCode}` };
  }
  const producedRows = rowCount === undefined || rowCount > 0;
  if (hasSummary && producedRows) return { kind: "complete" };

  // Exited 0 but there is nothing worth recording. Say which, because
  // "exited with code 0" as a failure reason tells the next reader nothing.
  if (parseError) return { kind: "fail", reason: parseError };
  if (!hasSummary) {
    return { kind: "fail", reason: `${tool} exited 0 but produced no summary` };
  }
  return { kind: "fail", reason: `${tool} exited 0 but produced no result rows` };
}
