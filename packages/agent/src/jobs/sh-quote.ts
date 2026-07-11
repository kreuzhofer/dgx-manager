/**
 * POSIX single-quote a string for safe interpolation into a `sh` script.
 *
 * Everything inside single quotes is literal, so the only character needing
 * care is the single quote itself: close the quote, emit an escaped quote,
 * reopen. The wrapper script built in job-spec.ts interpolates a benchmark's
 * argv, so a bug here is a shell-injection bug, not a formatting one.
 */
export function shQuote(s: string): string {
  return `'${s.replaceAll("'", "'\\''")}'`;
}
