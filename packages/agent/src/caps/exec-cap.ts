import { spawn as realSpawn } from "child_process";
import type { Capability, CapContext } from "./registry.js";

export interface ExecInput {
  cmd: string;
  args?: string[];
  timeoutMs?: number;
  reason?: string;
}

export interface ExecResult {
  code: number | null;
  timedOut: boolean;
}

export interface ExecAudit {
  cmd: string;
  args: string[];
  reason: string;
  code: number | null;
  timedOut: boolean;
  ts: number;
}

type SpawnFn = typeof realSpawn;

export function makeExecCap(
  spawnFn: SpawnFn = realSpawn,
  onAudit?: (a: ExecAudit) => void,
): Capability {
  return {
    name: "exec",
    handle: (input: unknown, ctx: CapContext) =>
      new Promise<ExecResult>((resolve, reject) => {
        const i = input as ExecInput;
        if (!i || typeof i.cmd !== "string" || !i.cmd) {
          return reject(new Error("exec: cmd required"));
        }
        if (!i.reason || !i.reason.trim()) {
          return reject(new Error("exec: reason required (audited)"));
        }
        const timeoutMs = Math.min(Math.max(i.timeoutMs ?? 30_000, 1), 300_000);
        const args = i.args ?? [];
        const child = spawnFn(i.cmd, args, {});
        let timedOut = false;
        const timer = setTimeout(() => {
          timedOut = true;
          try {
            child.kill("SIGKILL");
          } catch {
            /* gone */
          }
        }, timeoutMs);
        child.stdout?.on("data", (d: Buffer) =>
          ctx.emitChunk("stdout", d.toString()),
        );
        child.stderr?.on("data", (d: Buffer) =>
          ctx.emitChunk("stderr", d.toString()),
        );
        const done = (code: number | null): void => {
          clearTimeout(timer);
          onAudit?.({
            cmd: i.cmd,
            args,
            reason: i.reason!,
            code,
            timedOut,
            ts: Date.now(),
          });
          resolve({ code, timedOut });
        };
        child.on("close", (code) => done(code));
        child.on("error", (e) => {
          clearTimeout(timer);
          reject(e);
        });
      }),
  };
}
