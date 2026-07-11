export type JobStatus =
  | { kind: "active" }
  | { kind: "exited"; code: number }
  | { kind: "missing" }
  | { kind: "unknown"; reason: string };

export type CapInvoker = (
  nodeId: string,
  name: string,
  input: unknown,
) => Promise<{ ok: boolean; data?: unknown; error?: string }>;

export interface RemoteRunOpts {
  runId: string;
  nodeId: string;
  argv: string[];
  resultGlob?: string;
  pollMs?: number;
  invoke: CapInvoker;
  onLog: (line: string) => void;
  /** Called with the new byte offset so the caller can persist it for reattach. */
  onOffset?: (offset: number) => void;
  /** Where to resume from after a manager restart. */
  startOffset?: number;
  /** Reattach: skip job.start (the systemd unit already exists) and go straight to polling. */
  skipStart?: boolean;
}

/**
 * Decide what one poll tick means.
 *
 * `unknown` is NOT a verdict. A cap timeout or a busy box means we failed to ask,
 * not that the job died — an 80-minute eval must survive a slow poll. Only a unit
 * that systemd positively reports gone, with no exit file the wrapper would have
 * written, is a dead job. This is the absent-vs-unknown distinction that tore down
 * four healthy dgxrun ranks on 2026-07-09.
 */
export function nextPollAction(status: JobStatus, hasExitFile: boolean): "continue" | "finish" | "fail" {
  if (status.kind === "exited") return "finish";
  if (status.kind === "active") return "continue";
  if (status.kind === "unknown") return "continue";
  return hasExitFile ? "finish" : "fail"; // missing
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function call<T>(invoke: CapInvoker, nodeId: string, name: string, input: unknown): Promise<T | null> {
  const r = await invoke(nodeId, name, input);
  return r.ok ? (r.data as T) : null;
}

/**
 * Run a benchmark as a systemd job on the eval node.
 *
 * Mirrors `spawnTracked`'s `{exitCode, rawOutput}` contract so the three
 * `runBenchmark`/`runToolEval`/`runAccuracy` wrappers are unchanged above it.
 */
export async function runTrackedRemote(
  o: RemoteRunOpts,
): Promise<{ exitCode: number | null; rawOutput: string | null }> {
  const pollMs = o.pollMs ?? 3_000;

  if (!o.skipStart) {
    const started = await o.invoke(o.nodeId, "job.start", {
      runId: o.runId, argv: o.argv, resultGlob: o.resultGlob ?? "result.json",
    });
    if (!started.ok) throw new Error(`job.start failed: ${started.error}`);
  }

  let offset = o.startOffset ?? 0;

  const drain = async (): Promise<void> => {
    const logs = await call<{ chunk: string; nextOffset: number; truncated: boolean }>(
      o.invoke, o.nodeId, "job.logs", { runId: o.runId, offset },
    );
    if (!logs) return; // inconclusive — try again next tick
    if (logs.truncated) offset = 0;
    if (logs.chunk) {
      for (const line of logs.chunk.split("\n")) if (line) o.onLog(line);
    }
    if (logs.nextOffset !== offset) {
      offset = logs.nextOffset;
      o.onOffset?.(offset);
    }
  };

  for (;;) {
    await drain();
    const status = await call<JobStatus>(o.invoke, o.nodeId, "job.status", { runId: o.runId });
    const s: JobStatus = status ?? { kind: "unknown", reason: "cap call failed" };

    // Belt-and-braces before declaring a run dead. The agent already resolves the
    // exit file inside job.status, so a `missing` here is almost always real — but
    // the wrapper writes result.json just before it exits, and a `job.result` probe
    // is a second, independent check against a race where the unit is gone yet the
    // job genuinely finished. This is the ONLY place `hasExitFile` is non-false.
    let hasResult = false;
    let finishedResult: { raw: string | null } | null = null;
    if (s.kind === "missing") {
      finishedResult = await call<{ raw: string | null }>(o.invoke, o.nodeId, "job.result", { runId: o.runId });
      hasResult = finishedResult?.raw != null;
    }

    const action = nextPollAction(s, hasResult);
    if (action === "fail") {
      throw new Error(`job ${o.runId} vanished on ${o.nodeId} (unit gone, no exit file, no result)`);
    }
    if (action === "finish") {
      await drain(); // final tail
      if (s.kind === "exited") {
        const result = await call<{ raw: string | null }>(o.invoke, o.nodeId, "job.result", { runId: o.runId });
        return { exitCode: s.code, rawOutput: s.code === 0 ? (result?.raw ?? null) : null };
      }
      // Finished via the missing+result path: the unit was gone but a result file
      // exists, so the run completed. Treat as success (the parser validates it).
      return { exitCode: 0, rawOutput: finishedResult?.raw ?? null };
    }
    await sleep(pollMs);
  }
}
