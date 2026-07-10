import { shQuote } from "./sh-quote.js";

/** Runtime-safe runId: what a cuid actually is, enforced rather than assumed. */
const SAFE_RUN_ID = /^[A-Za-z0-9]+$/;

function assertSafeRunId(runId: string): void {
  if (typeof runId !== "string" || !SAFE_RUN_ID.test(runId)) {
    throw new Error(`unsafe runId (expected /^[A-Za-z0-9]+$/): ${JSON.stringify(runId)}`);
  }
}

/** systemd transient unit for a benchmark run. */
export function jobUnitName(runId: string): string {
  assertSafeRunId(runId);
  return `dgxbench-${runId}`;
}

/** Per-run state dir on the eval node: log, exit, result.json, cmd.sh. */
export function jobDir(home: string, runId: string): string {
  assertSafeRunId(runId);
  return `${home}/.dgx-agent/jobs/${runId}`;
}

/**
 * The script systemd actually runs.
 *
 * Ordering is load-bearing. The manager polls for the `exit` file and treats its
 * existence as "finished, and result.json is final". So: run the command, capture
 * ITS exit code (not the copy's), resolve and copy the result, and only then write
 * `exit` — atomically, via a temp file, so a partially written code is never read.
 *
 * `resultGlob` exists because the kinds disagree: llama-benchy and tool-eval-bench
 * write <outputDir>/result.json, while lm-eval writes a nested results_*.json. The
 * manager cannot stat a remote filesystem, so resolution happens here.
 */
export function buildWrapperScript(o: {
  argv: string[];
  jobDir: string;
  outputDir: string;
  resultGlob: string;
}): string {
  const cmd = o.argv.map(shQuote).join(" ");
  return [
    "#!/bin/sh",
    `cd ${shQuote(o.jobDir)} || exit 127`,
    `mkdir -p ${shQuote(o.outputDir)}`,
    `${cmd} > log 2>&1`,
    "code=$?",
    `f=$(find ${shQuote(o.outputDir)} -name ${shQuote(o.resultGlob)} -print -quit 2>/dev/null)`,
    `[ -n "$f" ] && cp "$f" result.json`,
    // exit LAST: its presence means "finished, result final".
    `printf '%s' "$code" > exit.tmp && mv exit.tmp exit`,
    "",
  ].join("\n");
}

/**
 * `systemd-run` argv. The agent runs as an unprivileged user (User=daniel in
 * dgx-agent.service), so a *system* transient unit needs `sudo -n`; we then hand
 * privileges straight back with `-p User=`, so the job's caches land in the
 * agent user's home rather than root's.
 *
 * `--collect` is deliberately omitted: it garbage-collects the unit on exit, and
 * we need `systemctl show` to still answer afterwards.
 */
export function buildSystemdRunArgv(o: {
  unit: string;
  jobDir: string;
  user: string;
  scriptPath: string;
}): string[] {
  return [
    "sudo", "-n", "systemd-run",
    `--unit=${o.unit}`,
    "-p", `User=${o.user}`,
    "-p", `WorkingDirectory=${o.jobDir}`,
    "-p", "RemainAfterExit=yes",
    "/bin/sh", o.scriptPath,
  ];
}
