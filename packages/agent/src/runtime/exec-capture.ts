import { spawn } from "node:child_process";

/**
 * Non-blocking replacement for `spawnSync(..., { encoding: "utf8" })` on the
 * agent's hot path.
 *
 * The agent is a single event loop that also answers the manager's WebSocket.
 * Every `spawnSync` on the 5-second health tick freezes that loop for the
 * duration of a `docker` call — and a busy docker daemon makes those calls
 * slow exactly when a deploy is in flight. Three separate incidents trace back
 * to this (#36): a wedged `cmd:update`, a drop-cache loop that froze the tick,
 * and a timed-out inspect that was misread as "container missing" and tore down
 * a healthy four-rank cluster.
 *
 * The RESULT SHAPE deliberately mirrors spawnSync's, because `classifyDockerInspect`
 * keys on it to tell `absent` (docker says the container is gone) apart from
 * `unknown` (docker never answered):
 *
 *   - normal exit      → `status` = exit code, `error` undefined
 *   - timeout / signal → `status` = null, `error` set
 *   - spawn failure    → `status` = null, `error` set
 *
 * Two deliberate improvements over spawnSync:
 *   - The promise settles AT the deadline. spawnSync waits for the child to die
 *     after SIGTERM; a child that ignores it blocks past the timeout. Here the
 *     kill is best-effort (SIGTERM, SIGKILL backstop) and the caller is freed
 *     on time regardless.
 *   - Exceeding `maxBuffer` truncates the captured output rather than failing
 *     the call. Both log-capture callers head-cap their output anyway, and
 *     losing a crash log to a buffer error is worse than losing its tail.
 */
export interface ExecCapture {
  /** Exit code, or `null` when the child was killed or never started. */
  status: number | null;
  stdout: string;
  stderr: string;
  /** Set whenever `status` is null — a timeout, a signal, or a spawn failure. */
  error?: Error;
}

export interface ExecCaptureOptions {
  /** Hard deadline in ms. The promise always settles by then. */
  timeout: number;
  /** Cap on EACH captured stream; output past it is dropped. Default 1 MiB. */
  maxBuffer?: number;
}

/** How long a killed child gets to exit on SIGTERM before SIGKILL. */
const SIGKILL_GRACE_MS = 2_000;

export function execCapture(
  file: string,
  args: string[],
  opts: ExecCaptureOptions,
): Promise<ExecCapture> {
  const maxBuffer = opts.maxBuffer ?? 1024 * 1024;

  return new Promise<ExecCapture>((resolve) => {
    const child = spawn(file, args, { stdio: ["ignore", "pipe", "pipe"] });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const collect = (chunk: Buffer, into: "out" | "err") => {
      if (into === "out") {
        if (stdout.length < maxBuffer) stdout += chunk.toString("utf8");
      } else if (stderr.length < maxBuffer) {
        stderr += chunk.toString("utf8");
      }
    };
    child.stdout?.on("data", (c: Buffer) => collect(c, "out"));
    child.stderr?.on("data", (c: Buffer) => collect(c, "err"));

    const finish = (status: number | null, error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ status, stdout, stderr, ...(error ? { error } : {}) });
    };

    const timer = setTimeout(() => {
      // Best-effort teardown; we do NOT wait for it (see the doc comment).
      try { child.kill("SIGTERM"); } catch { /* already gone */ }
      const hard = setTimeout(() => {
        try { child.kill("SIGKILL"); } catch { /* already gone */ }
      }, SIGKILL_GRACE_MS);
      hard.unref();
      finish(null, new Error(`${file} timed out after ${opts.timeout}ms`));
    }, opts.timeout);

    // ENOENT and friends: the child never ran.
    child.on("error", (err: Error) => finish(null, err));

    child.on("close", (code, signal) => {
      if (signal) finish(null, new Error(`${file} killed with signal ${signal}`));
      else finish(code);
    });
  });
}
