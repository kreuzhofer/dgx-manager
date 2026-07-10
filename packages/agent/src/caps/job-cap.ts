import { spawn as realSpawn } from "child_process";
import {
  mkdirSync,
  writeFileSync,
  readFileSync,
  statSync,
  openSync,
  readSync,
  closeSync,
  readdirSync,
  rmSync,
} from "fs";
import type { Capability } from "./registry.js";
import { jobUnitName, jobDir, buildWrapperScript, buildSystemdRunArgv } from "../jobs/job-spec.js";
import { parseSystemctlShow, type JobStatus } from "../jobs/systemctl-parse.js";
import { planRead } from "../jobs/log-slice.js";
import { stalePaths } from "../jobs/prune.js";

export interface JobCapDeps {
  home: string;
  user: string;
  spawnFn: typeof realSpawn;
  writeFile(path: string, data: string): void;
  mkdir(path: string): void;
  /** Byte range [from,to) of `path`, plus its current size. */
  readFileSlice(path: string, from: number, to: number): { chunk: string; size: number };
  /** Whole file, or null when absent. */
  readTextFile(path: string): string | null;
  now?(): number;
  listJobDirs?(): { path: string; mtimeMs: number }[];
  removeDir?(path: string): void;
}

interface RunOut { code: number | null; stdout: string; stderr: string }

function run(spawnFn: typeof realSpawn, argv: string[]): Promise<RunOut> {
  return new Promise((resolve, reject) => {
    const child = spawnFn(argv[0], argv.slice(1), {});
    let stdout = "";
    let stderr = "";
    child.stdout?.on("data", (b: Buffer) => (stdout += b.toString()));
    child.stderr?.on("data", (b: Buffer) => (stderr += b.toString()));
    child.on("error", reject);
    child.on("close", (code) => resolve({ code, stdout, stderr }));
  });
}

function defaultDeps(): Required<JobCapDeps> {
  const home = process.env.HOME ?? "/home/daniel";
  return {
    home,
    user: process.env.USER ?? "daniel",
    spawnFn: realSpawn,
    writeFile: (p, d) => writeFileSync(p, d, { mode: 0o755 }),
    mkdir: (p) => mkdirSync(p, { recursive: true }),
    readFileSlice: (p, from, to) => {
      const size = statSync(p).size;
      if (to <= from) return { chunk: "", size };
      const fd = openSync(p, "r");
      try {
        const buf = Buffer.alloc(to - from);
        const n = readSync(fd, buf, 0, to - from, from);
        return { chunk: buf.subarray(0, n).toString("utf8"), size };
      } finally {
        closeSync(fd);
      }
    },
    readTextFile: (p) => {
      try { return readFileSync(p, "utf8"); } catch { return null; }
    },
    now: () => Date.now(),
    listJobDirs: () => {
      const jobsRoot = `${home}/.dgx-agent/jobs`;
      let names: string[];
      try {
        names = readdirSync(jobsRoot);
      } catch {
        return [];
      }
      return names.map((name) => {
        const path = `${jobsRoot}/${name}`;
        return { path, mtimeMs: statSync(path).mtimeMs };
      });
    },
    removeDir: (p) => rmSync(p, { recursive: true, force: true }),
  };
}

/**
 * Long-running jobs, owned by systemd rather than by the agent.
 *
 * The agent is a controller, not a parent: it hands the command to a transient
 * unit and returns. That is what lets a benchmark outlive an agent roll AND a
 * manager rebuild. Every call here is short, so it fits inside CapClient's
 * invocation timeout — the manager polls rather than streams.
 */
export function makeJobCaps(depsIn?: Partial<JobCapDeps>): Capability[] {
  const d: Required<JobCapDeps> = { ...defaultDeps(), ...depsIn };

  const paths = (runId: string) => {
    const dir = jobDir(d.home, runId);       // throws on an unsafe runId
    return {
      dir,
      outputDir: `${dir}/out`,
      script: `${dir}/cmd.sh`,
      log: `${dir}/log`,
      exit: `${dir}/exit`,
      result: `${dir}/result.json`,
      unit: jobUnitName(runId),
    };
  };

  const start: Capability = {
    name: "job.start",
    handle: async (input) => {
      const i = input as { runId: string; argv: string[]; resultGlob?: string };
      if (!Array.isArray(i?.argv) || i.argv.length === 0) throw new Error("job.start: argv required");

      // Best-effort housekeeping. A pruning failure must never stop a benchmark.
      try {
        for (const p of stalePaths(d.listJobDirs(), d.now())) d.removeDir(p);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn(`[job.start] prune skipped: ${(e as Error).message}`);
      }

      const p = paths(i.runId);
      d.mkdir(p.outputDir);
      d.writeFile(
        p.script,
        buildWrapperScript({
          argv: i.argv,
          jobDir: p.dir,
          outputDir: p.outputDir,
          resultGlob: i.resultGlob ?? "result.json",
        }),
      );
      const r = await run(d.spawnFn, buildSystemdRunArgv({
        unit: p.unit, jobDir: p.dir, user: d.user, scriptPath: p.script,
      }));
      if (r.code !== 0) {
        throw new Error(`systemd-run failed (exit ${r.code}): ${r.stderr.trim().slice(0, 300)}`);
      }
      return { unit: p.unit, jobDir: p.dir, outputDir: p.outputDir };
    },
  };

  const status: Capability = {
    name: "job.status",
    handle: async (input) => {
      const p = paths((input as { runId: string }).runId);
      const r = await run(d.spawnFn, [
        "systemctl", "show", p.unit,
        "-p", "LoadState", "-p", "ActiveState", "-p", "ExecMainStatus",
      ]);
      const parsed = parseSystemctlShow(r.code, r.stdout, r.stderr);

      // `unknown` is terminal-for-this-tick: we could not ask. Do NOT consult the
      // exit file to "help" — an absent exit file would then read as a dead job.
      if (parsed.kind === "unknown") return parsed;

      // systemd may garbage-collect the unit before we poll. The wrapper writes
      // `exit` LAST, so its presence proves the job finished and result.json is final.
      if (parsed.kind === "missing" || parsed.kind === "active") {
        const raw = d.readTextFile(p.exit);
        if (raw !== null && raw.trim() !== "") {
          const code = Number(raw.trim());
          if (Number.isInteger(code)) return { kind: "exited", code } satisfies JobStatus;
        }
      }
      return parsed;
    },
  };

  const logs: Capability = {
    name: "job.logs",
    handle: async (input) => {
      const i = input as { runId: string; offset?: number };
      const p = paths(i.runId);
      let size = 0;
      try { size = d.readFileSlice(p.log, 0, 0).size; } catch { return { chunk: "", nextOffset: 0, truncated: false }; }
      const plan = planRead(i.offset ?? 0, size);
      const { chunk } = d.readFileSlice(p.log, plan.from, plan.to);
      return { chunk, nextOffset: plan.to, truncated: plan.truncated };
    },
  };

  const cancel: Capability = {
    name: "job.cancel",
    handle: async (input) => {
      const p = paths((input as { runId: string }).runId);
      // Idempotent: stopping an already-gone unit is success, not failure.
      await run(d.spawnFn, ["sudo", "-n", "systemctl", "stop", p.unit]).catch(() => undefined);
      return { stopped: true };
    },
  };

  const result: Capability = {
    name: "job.result",
    handle: async (input) => {
      const p = paths((input as { runId: string }).runId);
      return { raw: d.readTextFile(p.result) };
    },
  };

  return [start, status, logs, cancel, result];
}
