# Agent v2 Phase 1 — Incident-Response Core — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the manager fork-free deep observation (`diag.collect`) + rich streaming metrics + an audited break-glass `exec`, driven over the agent WebSocket, so a degraded node (wedged sshd, fork-starved) can be diagnosed and managed without SSH.

**Architecture:** Evolve the existing agent. Add pure `/proc` + `/sys` parsers (`sysinfo/`), a table-driven capability registry (`caps/`) dispatched by the existing WS router, and enrich the metric tick. Server gains a capability client, two REST endpoints, an audit table, and metric-field storage with self-heal. The critical design point: `diag.collect` reads `/proc` in-process (no `fork`), so it works when `fork()` is failing.

**Tech Stack:** TypeScript (strict, ESM), Node 22, Vitest + @fast-check/vitest, Prisma (SQLite), `ws`. Test pattern: pure parsers over fixture strings + injected IO (like `sparkrun-parse.ts` / `dgxrun-args.ts`).

## Global Constraints

- TypeScript strict mode, ES modules; `.js` import extensions in TS source.
- Agent code change ⇒ run `./scripts/bump-agent-version.sh` once at the end (MANDATORY per CLAUDE.md).
- Every task ends `npm test` green; typecheck clean (`npx tsc --noEmit -p packages/agent/tsconfig.json` and `.../server/tsconfig.json`).
- Pure parsers take strings; readers take an injectable `readText(path): string` (default `readFileSync(path,"utf-8")`) so they unit-test without touching real `/proc`.
- Commit after every task. Prefix agent-side `feat(agent-v2):`, server-side `feat(agent-v2-srv):`.
- Do NOT break existing sparkrun/dgxrun/ollama paths — the WS router gains a new case, it doesn't change existing ones.

## File structure

- `packages/agent/src/sysinfo/proc-parse.ts` — pure parsers (meminfo, pressure, loadavg, file-nr, /proc/net/tcp-by-port). No IO.
- `packages/agent/src/sysinfo/proc-parse.test.ts` — parser unit+property tests.
- `packages/agent/src/sysinfo/proc-read.ts` — thin readers: inject `readText`; assemble typed readings; thermals/kmsg/disk/pidcount.
- `packages/agent/src/sysinfo/proc-read.test.ts` — reader tests with stub `readText`.
- `packages/agent/src/sysinfo/diag.ts` — `collectDiag()` bundle (GPU best-effort).
- `packages/agent/src/caps/registry.ts` — capability registry + dispatch.
- `packages/agent/src/caps/registry.test.ts`
- `packages/agent/src/caps/exec-cap.ts` — audited `exec` handler.
- `packages/agent/src/caps/exec-cap.test.ts`
- `packages/agent/src/index.ts` — wire registry into WS router; enrich metrics payload.
- `packages/server/src/caps/cap-client.ts` — server-side request→result correlation + timeout.
- `packages/server/src/caps/cap-client.test.ts`
- `packages/server/src/ws/agent-hub.ts` — handle `agent:cap:result/chunk`, `agent:audit`; store enriched metrics + self-heal.
- `packages/server/src/routes/nodes.ts` — `POST /:id/diag`, `POST /:id/exec`.
- `packages/server/src/__tests__/integration/agent-v2.caps.test.ts`
- `prisma/schema.prisma` — `AuditEvent` model + metric fields.

---

### Task 1: Memory parser (`parseMeminfo`)

**Files:**
- Create: `packages/agent/src/sysinfo/proc-parse.ts`
- Test: `packages/agent/src/sysinfo/proc-parse.test.ts`

**Interfaces:**
- Produces: `export interface MemoryInfo { totalMb:number; availableMb:number; freeMb:number; cachedMb:number; swapTotalMb:number; swapFreeMb:number }` and `export function parseMeminfo(text:string): MemoryInfo`. `/proc/meminfo` reports kB; convert to MB (`Math.round(kB/1024)`).

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { parseMeminfo } from "./proc-parse.js";

describe("parseMeminfo", () => {
  it("converts kB fields to MB", () => {
    const text = [
      "MemTotal:       127535340 kB",
      "MemFree:          869000 kB",
      "MemAvailable:    7100000 kB",
      "Buffers:          100000 kB",
      "Cached:          9900000 kB",
      "SwapTotal:             0 kB",
      "SwapFree:              0 kB",
    ].join("\n");
    const m = parseMeminfo(text);
    expect(m.totalMb).toBe(124546);
    expect(m.availableMb).toBe(6934);
    expect(m.freeMb).toBe(849);
    expect(m.cachedMb).toBe(9668);
    expect(m.swapTotalMb).toBe(0);
  });
  it("missing fields default to 0", () => {
    expect(parseMeminfo("MemTotal: 1024 kB").availableMb).toBe(0);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**
Run: `npx vitest run packages/agent/src/sysinfo/proc-parse.test.ts`
Expected: FAIL — `parseMeminfo` is not exported.

- [ ] **Step 3: Write minimal implementation**
```ts
export interface MemoryInfo {
  totalMb: number; availableMb: number; freeMb: number;
  cachedMb: number; swapTotalMb: number; swapFreeMb: number;
}

/** Parse /proc/meminfo (kB values) into MB. Missing keys default to 0. */
export function parseMeminfo(text: string): MemoryInfo {
  const kb = (key: string): number => {
    const m = text.match(new RegExp(`^${key}:\\s+(\\d+)\\s+kB`, "m"));
    return m ? Math.round(parseInt(m[1], 10) / 1024) : 0;
  };
  return {
    totalMb: kb("MemTotal"), availableMb: kb("MemAvailable"), freeMb: kb("MemFree"),
    cachedMb: kb("Cached"), swapTotalMb: kb("SwapTotal"), swapFreeMb: kb("SwapFree"),
  };
}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `npx vitest run packages/agent/src/sysinfo/proc-parse.test.ts` — Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add packages/agent/src/sysinfo/proc-parse.ts packages/agent/src/sysinfo/proc-parse.test.ts
git commit -m "feat(agent-v2): parseMeminfo /proc/meminfo -> MB"
```

---

### Task 2: Pressure (PSI) parser (`parsePressure`)

**Files:**
- Modify: `packages/agent/src/sysinfo/proc-parse.ts`
- Test: `packages/agent/src/sysinfo/proc-parse.test.ts`

**Interfaces:**
- Produces: `export interface PsiLine { avg10:number; avg60:number; avg300:number; total:number }`, `export interface Pressure { some:PsiLine; full:PsiLine|null }`, `export function parsePressure(text:string): Pressure`. A `/proc/pressure/*` file has a `some ...` line and (except cpu) a `full ...` line.

- [ ] **Step 1: Write the failing test**
```ts
import { parsePressure } from "./proc-parse.js";
describe("parsePressure", () => {
  it("parses some + full lines", () => {
    const p = parsePressure(
      "some avg10=1.23 avg60=4.56 avg300=7.89 total=12345\n" +
      "full avg10=0.10 avg60=0.20 avg300=0.30 total=678\n");
    expect(p.some.avg10).toBe(1.23);
    expect(p.some.total).toBe(12345);
    expect(p.full?.avg60).toBe(0.20);
  });
  it("cpu pressure has no full line", () => {
    expect(parsePressure("some avg10=0 avg60=0 avg300=0 total=0").full).toBeNull();
  });
});
```

- [ ] **Step 2: Run** — Expected FAIL (`parsePressure` undefined).

- [ ] **Step 3: Implement (append to proc-parse.ts)**
```ts
export interface PsiLine { avg10: number; avg60: number; avg300: number; total: number; }
export interface Pressure { some: PsiLine; full: PsiLine | null; }

function psiLine(text: string, kind: "some" | "full"): PsiLine | null {
  const m = text.match(new RegExp(`^${kind}\\s+avg10=([\\d.]+)\\s+avg60=([\\d.]+)\\s+avg300=([\\d.]+)\\s+total=(\\d+)`, "m"));
  if (!m) return null;
  return { avg10: parseFloat(m[1]), avg60: parseFloat(m[2]), avg300: parseFloat(m[3]), total: parseInt(m[4], 10) };
}

/** Parse a /proc/pressure/{cpu,memory,io} file. `full` is null for cpu. */
export function parsePressure(text: string): Pressure {
  return { some: psiLine(text, "some") ?? { avg10: 0, avg60: 0, avg300: 0, total: 0 }, full: psiLine(text, "full") };
}
```

- [ ] **Step 4: Run** — Expected PASS.
- [ ] **Step 5: Commit**
```bash
git add -A && git commit -m "feat(agent-v2): parsePressure PSI some/full"
```

---

### Task 3: Load + fd parsers (`parseLoadavg`, `parseFileNr`)

**Files:** Modify `proc-parse.ts` + its test.

**Interfaces:**
- Produces: `export interface LoadInfo { load1:number; load5:number; load15:number; runnable:number; totalProcs:number }`, `parseLoadavg(text):LoadInfo`; `export interface FdInfo { allocated:number; max:number }`, `parseFileNr(text):FdInfo` (`/proc/sys/fs/file-nr` is `allocated 0 max`).

- [ ] **Step 1: Test**
```ts
import { parseLoadavg, parseFileNr } from "./proc-parse.js";
describe("parseLoadavg", () => {
  it("parses loads + proc counts", () => {
    const l = parseLoadavg("2.15 1.80 1.44 3/1234 99999");
    expect(l.load1).toBe(2.15); expect(l.runnable).toBe(3); expect(l.totalProcs).toBe(1234);
  });
});
describe("parseFileNr", () => {
  it("parses allocated and max", () => {
    const f = parseFileNr("12800\t0\t9223372");
    expect(f.allocated).toBe(12800); expect(f.max).toBe(9223372);
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
export interface LoadInfo { load1: number; load5: number; load15: number; runnable: number; totalProcs: number; }
export function parseLoadavg(text: string): LoadInfo {
  const p = text.trim().split(/\s+/);
  const [run, total] = (p[3] ?? "0/0").split("/");
  return { load1: parseFloat(p[0]) || 0, load5: parseFloat(p[1]) || 0, load15: parseFloat(p[2]) || 0,
           runnable: parseInt(run, 10) || 0, totalProcs: parseInt(total, 10) || 0 };
}
export interface FdInfo { allocated: number; max: number; }
export function parseFileNr(text: string): FdInfo {
  const p = text.trim().split(/\s+/);
  return { allocated: parseInt(p[0], 10) || 0, max: parseInt(p[2], 10) || 0 };
}
```
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): parseLoadavg + parseFileNr"`

---

### Task 4: The money read — `/proc/net/tcp` connections by state for a port

**Files:** Modify `proc-parse.ts` + its test.

**Interfaces:**
- Produces: `export const TCP_STATES: Record<string,string>` (hex→name) and
  `export function parseProcNetTcpByPort(text:string, port:number): Record<string,number>`.
  `/proc/net/tcp` columns: `sl local_address rem_address st ...`; `local_address` is `HEXIP:HEXPORT`; `st` is a 2-hex TCP state. We count rows whose local port == `port`, tallied by state name. Works for `/proc/net/tcp` and `/proc/net/tcp6` (same layout). This distinguishes MaxStartups pileup (many `SYN_RECV`) from fork-starvation (few conns).

- [ ] **Step 1: Test (include a pileup fixture and a starved fixture)**
```ts
import { parseProcNetTcpByPort } from "./proc-parse.js";
describe("parseProcNetTcpByPort", () => {
  // port 22 = 0x16. state 0A=LISTEN, 01=ESTABLISHED, 03=SYN_RECV.
  const header = "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt   uid\n";
  it("tallies :22 rows by TCP state (MaxStartups pileup)", () => {
    const rows = [
      "0: 00000000:0016 00000000:0000 0A 0 0",           // LISTEN
      "1: 0100007F:0016 0200007F:C000 01 0 0",           // ESTABLISHED
      ...Array.from({length:30},(_,i)=>`${i+2}: 0100007F:0016 0300007F:${i} 03 0 0`), // 30 SYN_RECV
    ].join("\n");
    const t = parseProcNetTcpByPort(header + rows, 22);
    expect(t.LISTEN).toBe(1); expect(t.ESTABLISHED).toBe(1); expect(t.SYN_RECV).toBe(30);
  });
  it("ignores other ports and empty input", () => {
    const t = parseProcNetTcpByPort(header + "0: 00000000:0050 00000000:0000 0A 0 0", 22);
    expect(t.LISTEN ?? 0).toBe(0);
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
export const TCP_STATES: Record<string, string> = {
  "01": "ESTABLISHED", "02": "SYN_SENT", "03": "SYN_RECV", "04": "FIN_WAIT1",
  "05": "FIN_WAIT2", "06": "TIME_WAIT", "07": "CLOSE", "08": "CLOSE_WAIT",
  "09": "LAST_ACK", "0A": "LISTEN", "0B": "CLOSING",
};
/** Count rows on `port` (local) in /proc/net/tcp[6] output, tallied by TCP state name. */
export function parseProcNetTcpByPort(text: string, port: number): Record<string, number> {
  const out: Record<string, number> = {};
  const hexPort = port.toString(16).toUpperCase().padStart(4, "0");
  for (const line of text.split("\n")) {
    const cols = line.trim().split(/\s+/);
    if (cols.length < 4 || !cols[1]?.includes(":")) continue;
    const lp = cols[1].split(":")[1]?.toUpperCase();
    if (lp !== hexPort) continue;
    const st = TCP_STATES[cols[3].toUpperCase()] ?? cols[3];
    out[st] = (out[st] ?? 0) + 1;
  }
  return out;
}
```
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): parseProcNetTcpByPort — sshd conns by state"`

---

### Task 5: Readers (`proc-read.ts`) with injected IO

**Files:**
- Create: `packages/agent/src/sysinfo/proc-read.ts`
- Test: `packages/agent/src/sysinfo/proc-read.test.ts`

**Interfaces:**
- Produces: `export interface SysReadings { memory:MemoryInfo; pressure:{cpu:Pressure;memory:Pressure;io:Pressure}; load:LoadInfo; fds:FdInfo; sshd:Record<string,number>; thermalsC:number[] }` and `export function readSysInfo(readText?:(p:string)=>string): SysReadings`. `readText` defaults to `(p)=>readFileSync(p,"utf-8")`; a missing/erroring path yields empty string so parsers return zeroed structs. sshd = merge of tcp + tcp6 tallies for port 22. thermalsC = each `/sys/class/thermal/thermal_zone*/temp` (milli-°C) /1000; glob via `readdirSync` (also injectable — default real).

- [ ] **Step 1: Test with stub readText**
```ts
import { describe, it, expect } from "vitest";
import { readSysInfo } from "./proc-read.js";
describe("readSysInfo", () => {
  it("assembles readings from injected /proc contents", () => {
    const files: Record<string,string> = {
      "/proc/meminfo": "MemTotal: 127535340 kB\nMemAvailable: 7100000 kB",
      "/proc/pressure/cpu": "some avg10=0 avg60=0 avg300=0 total=0",
      "/proc/pressure/memory": "some avg10=5 avg60=1 avg300=0 total=9\nfull avg10=2 avg60=0 avg300=0 total=3",
      "/proc/pressure/io": "some avg10=0 avg60=0 avg300=0 total=0",
      "/proc/loadavg": "1.0 1.0 1.0 2/500 9",
      "/proc/sys/fs/file-nr": "1000 0 900000",
      "/proc/net/tcp": "sl local rem st\n0: 00000000:0016 00000000:0000 0A 0 0",
      "/proc/net/tcp6": "",
    };
    const r = readSysInfo((p) => files[p] ?? "");
    expect(r.memory.totalMb).toBe(124546);
    expect(r.pressure.memory.some.avg10).toBe(5);
    expect(r.sshd.LISTEN).toBe(1);
    expect(r.load.load1).toBe(1.0);
  });
  it("tolerates missing files (readText throws) with zeroed structs", () => {
    const r = readSysInfo(() => { throw new Error("ENOENT"); });
    expect(r.memory.totalMb).toBe(0);
    expect(r.sshd).toEqual({});
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
import { readFileSync, readdirSync } from "fs";
import {
  parseMeminfo, parsePressure, parseLoadavg, parseFileNr, parseProcNetTcpByPort,
  type MemoryInfo, type Pressure, type LoadInfo, type FdInfo,
} from "./proc-parse.js";

export interface SysReadings {
  memory: MemoryInfo;
  pressure: { cpu: Pressure; memory: Pressure; io: Pressure };
  load: LoadInfo; fds: FdInfo; sshd: Record<string, number>; thermalsC: number[];
}
const safe = (read: (p: string) => string, p: string): string => {
  try { return read(p); } catch { return ""; }
};
export function readSysInfo(readText: (p: string) => string = (p) => readFileSync(p, "utf-8")): SysReadings {
  const tcp = parseProcNetTcpByPort(safe(readText, "/proc/net/tcp"), 22);
  const tcp6 = parseProcNetTcpByPort(safe(readText, "/proc/net/tcp6"), 22);
  const sshd: Record<string, number> = { ...tcp };
  for (const [k, v] of Object.entries(tcp6)) sshd[k] = (sshd[k] ?? 0) + v;
  let thermalsC: number[] = [];
  try {
    thermalsC = readdirSync("/sys/class/thermal").filter((d) => d.startsWith("thermal_zone"))
      .map((d) => Math.round((parseInt(safe(readText, `/sys/class/thermal/${d}/temp`), 10) || 0) / 1000))
      .filter((n) => n > 0);
  } catch { /* no thermal zones */ }
  return {
    memory: parseMeminfo(safe(readText, "/proc/meminfo")),
    pressure: {
      cpu: parsePressure(safe(readText, "/proc/pressure/cpu")),
      memory: parsePressure(safe(readText, "/proc/pressure/memory")),
      io: parsePressure(safe(readText, "/proc/pressure/io")),
    },
    load: parseLoadavg(safe(readText, "/proc/loadavg")),
    fds: parseFileNr(safe(readText, "/proc/sys/fs/file-nr")),
    sshd, thermalsC,
  };
}
```
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): readSysInfo — fork-free /proc readings"`

---

### Task 6: `diag.ts` — assemble the bundle (GPU best-effort)

**Files:**
- Create: `packages/agent/src/sysinfo/diag.ts`
- Test: `packages/agent/src/sysinfo/diag.test.ts`

**Interfaces:**
- Consumes: `readSysInfo` (Task 5).
- Produces: `export interface Diag extends SysReadings { pidCount:number; kmsgTail:string[]; gpu:string|null }` and `export function collectDiag(deps?:{ readText?:(p:string)=>string; pidCount?:()=>number; kmsgTail?:()=>string[]; gpu?:()=>string|null }): Diag`. GPU via injected fn (default: `execSync("nvidia-smi ...")` wrapped in try→null) — best-effort so a fork failure doesn't sink the bundle. `pidCount` default counts `/proc/[0-9]+` via readdir; `kmsgTail` default returns `[]` (real impl reads /dev/kmsg — keep out of unit scope, inject in prod).

- [ ] **Step 1: Test**
```ts
import { describe, it, expect } from "vitest";
import { collectDiag } from "./diag.js";
describe("collectDiag", () => {
  it("returns a bundle even when GPU probe fails", () => {
    const d = collectDiag({
      readText: (p) => (p === "/proc/meminfo" ? "MemTotal: 1048576 kB" : ""),
      pidCount: () => 512, kmsgTail: () => ["oom-killer invoked"],
      gpu: () => { throw new Error("fork failed"); },
    });
    expect(d.memory.totalMb).toBe(1024);
    expect(d.pidCount).toBe(512);
    expect(d.gpu).toBeNull();
    expect(d.kmsgTail[0]).toContain("oom");
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
import { readdirSync } from "fs";
import { execSync } from "child_process";
import { readSysInfo, type SysReadings } from "./proc-read.js";

export interface Diag extends SysReadings { pidCount: number; kmsgTail: string[]; gpu: string | null; }
interface DiagDeps {
  readText?: (p: string) => string;
  pidCount?: () => number;
  kmsgTail?: () => string[];
  gpu?: () => string | null;
}
const defaultPidCount = (): number => {
  try { return readdirSync("/proc").filter((d) => /^\d+$/.test(d)).length; } catch { return 0; }
};
const defaultGpu = (): string | null => {
  try {
    return execSync("nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader",
      { timeout: 4000, encoding: "utf-8" }).trim();
  } catch { return null; }
};
export function collectDiag(deps: DiagDeps = {}): Diag {
  const sys = readSysInfo(deps.readText);
  let gpu: string | null = null;
  try { gpu = (deps.gpu ?? defaultGpu)(); } catch { gpu = null; }
  return { ...sys, pidCount: (deps.pidCount ?? defaultPidCount)(), kmsgTail: (deps.kmsgTail ?? (() => []))(), gpu };
}
```
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): collectDiag bundle (GPU best-effort)"`

---

### Task 7: Capability registry

**Files:**
- Create: `packages/agent/src/caps/registry.ts`
- Test: `packages/agent/src/caps/registry.test.ts`

**Interfaces:**
- Produces:
  `export interface CapContext { emitChunk(stream:"stdout"|"stderr", data:string): void }`
  `export interface Capability { name:string; handle(input:unknown, ctx:CapContext): Promise<unknown> }`
  `export class CapRegistry { register(c:Capability):void; async dispatch(name:string, input:unknown, ctx:CapContext): Promise<{ok:true;data:unknown}|{ok:false;error:string}> }`
  Unknown name → `{ok:false, error:"unknown capability: <name>"}`; a throwing handler → `{ok:false, error:<message>}`.

- [ ] **Step 1: Test**
```ts
import { describe, it, expect, vi } from "vitest";
import { CapRegistry } from "./registry.js";
const noopCtx = { emitChunk: () => {} };
describe("CapRegistry", () => {
  it("dispatches to a registered capability", async () => {
    const r = new CapRegistry();
    r.register({ name: "echo", handle: async (i) => ({ got: i }) });
    expect(await r.dispatch("echo", 42, noopCtx)).toEqual({ ok: true, data: { got: 42 } });
  });
  it("unknown capability -> ok:false", async () => {
    expect(await new CapRegistry().dispatch("nope", null, noopCtx))
      .toEqual({ ok: false, error: "unknown capability: nope" });
  });
  it("handler throw -> ok:false with message", async () => {
    const r = new CapRegistry();
    r.register({ name: "boom", handle: async () => { throw new Error("kaboom"); } });
    expect(await r.dispatch("boom", null, noopCtx)).toEqual({ ok: false, error: "kaboom" });
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
export interface CapContext { emitChunk(stream: "stdout" | "stderr", data: string): void; }
export interface Capability { name: string; handle(input: unknown, ctx: CapContext): Promise<unknown>; }
export type CapResult = { ok: true; data: unknown } | { ok: false; error: string };

export class CapRegistry {
  private caps = new Map<string, Capability>();
  register(c: Capability): void { this.caps.set(c.name, c); }
  async dispatch(name: string, input: unknown, ctx: CapContext): Promise<CapResult> {
    const cap = this.caps.get(name);
    if (!cap) return { ok: false, error: `unknown capability: ${name}` };
    try { return { ok: true, data: await cap.handle(input, ctx) }; }
    catch (e) { return { ok: false, error: (e as Error).message }; }
  }
}
```
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): capability registry + dispatch"`

---

### Task 8: `exec` capability (audited, reason-required)

**Files:**
- Create: `packages/agent/src/caps/exec-cap.ts`
- Test: `packages/agent/src/caps/exec-cap.test.ts`

**Interfaces:**
- Consumes: `Capability`, `CapContext` (Task 7).
- Produces: `export interface ExecInput { cmd:string; args?:string[]; timeoutMs?:number; reason?:string }`, `export interface ExecResult { code:number|null; timedOut:boolean }`, and `export function makeExecCap(spawnFn?:SpawnFn, onAudit?:(a:ExecAudit)=>void): Capability`. `SpawnFn` matches child_process `spawn(cmd,args,opts)` returning an object with `.stdout/.stderr` (EventEmitter) and `.on("close"|"error")`. Reject (throw) when `reason` is missing/blank. Hard-cap `timeoutMs` (default 30000, max 300000). Streams chunks via `ctx.emitChunk`. Calls `onAudit({cmd,args,reason,code,timedOut,ts})` on completion.

- [ ] **Step 1: Test with a fake spawn**
```ts
import { describe, it, expect, vi } from "vitest";
import { EventEmitter } from "events";
import { makeExecCap } from "./exec-cap.js";

function fakeSpawn(out: string, code: number) {
  return () => {
    const child: any = new EventEmitter();
    child.stdout = new EventEmitter(); child.stderr = new EventEmitter();
    child.kill = () => {};
    setTimeout(() => { child.stdout.emit("data", Buffer.from(out)); child.emit("close", code); }, 0);
    return child;
  };
}
describe("exec cap", () => {
  it("rejects when reason is missing", async () => {
    const cap = makeExecCap(fakeSpawn("hi", 0));
    await expect(cap.handle({ cmd: "echo", args: ["hi"] }, { emitChunk: () => {} }))
      .rejects.toThrow(/reason/i);
  });
  it("streams output, returns code, and audits", async () => {
    const audits: any[] = [];
    const cap = makeExecCap(fakeSpawn("hello", 0), (a) => audits.push(a));
    const chunks: string[] = [];
    const res: any = await cap.handle(
      { cmd: "echo", args: ["hello"], reason: "debugging .36 sshd" },
      { emitChunk: (_s, d) => chunks.push(d) });
    expect(res.code).toBe(0);
    expect(chunks.join("")).toContain("hello");
    expect(audits[0].reason).toBe("debugging .36 sshd");
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
import { spawn as realSpawn } from "child_process";
import type { Capability, CapContext } from "./registry.js";

export interface ExecInput { cmd: string; args?: string[]; timeoutMs?: number; reason?: string; }
export interface ExecResult { code: number | null; timedOut: boolean; }
export interface ExecAudit { cmd: string; args: string[]; reason: string; code: number | null; timedOut: boolean; ts: number; }
type SpawnFn = typeof realSpawn;

export function makeExecCap(spawnFn: SpawnFn = realSpawn, onAudit?: (a: ExecAudit) => void): Capability {
  return {
    name: "exec",
    handle: (input: unknown, ctx: CapContext) => new Promise<ExecResult>((resolve, reject) => {
      const i = input as ExecInput;
      if (!i || typeof i.cmd !== "string" || !i.cmd) return reject(new Error("exec: cmd required"));
      if (!i.reason || !i.reason.trim()) return reject(new Error("exec: reason required (audited)"));
      const timeoutMs = Math.min(Math.max(i.timeoutMs ?? 30_000, 1), 300_000);
      const args = i.args ?? [];
      const child = spawnFn(i.cmd, args, { });
      let timedOut = false;
      const timer = setTimeout(() => { timedOut = true; try { child.kill("SIGKILL"); } catch { /* gone */ } }, timeoutMs);
      child.stdout?.on("data", (d: Buffer) => ctx.emitChunk("stdout", d.toString()));
      child.stderr?.on("data", (d: Buffer) => ctx.emitChunk("stderr", d.toString()));
      const done = (code: number | null): void => {
        clearTimeout(timer);
        onAudit?.({ cmd: i.cmd, args, reason: i.reason!, code, timedOut, ts: Date.now() });
        resolve({ code, timedOut });
      };
      child.on("close", (code) => done(code));
      child.on("error", (e) => { clearTimeout(timer); reject(e); });
    }),
  };
}
```
Note: `Date.now()` is fine in the agent (not a Workflow script).

- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): audited exec capability"`

---

### Task 9: Wire the registry + diag into the agent WS router; enrich metrics

**Files:**
- Modify: `packages/agent/src/index.ts` (imports; build a `CapRegistry` at startup registering `diag.collect` + `exec`; add a `case "agent:cap:request"` to the message switch; add sysinfo fields to the `agent:metrics` payload).

**Interfaces:**
- Consumes: `CapRegistry` (Task 7), `makeExecCap` (Task 8), `collectDiag` (Task 6), `readSysInfo` (Task 5).
- Produces (wire protocol): inbound `{ type:"agent:cap:request", payload:{ id, name, input } }`; outbound `{ type:"agent:cap:chunk", payload:{ id, stream, data } }` and `{ type:"agent:cap:result", payload:{ id, ok, data?, error? } }`; and `{ type:"agent:audit", payload: ExecAudit & { cap:"exec" } }`. Metrics payload gains `sysinfo: SysReadings`.

- [ ] **Step 1: Add registry construction (near other startup singletons)**
```ts
import { CapRegistry } from "./caps/registry.js";
import { makeExecCap } from "./caps/exec-cap.js";
import { collectDiag } from "./sysinfo/diag.js";
import { readSysInfo } from "./sysinfo/proc-read.js";

const caps = new CapRegistry();
caps.register({ name: "diag.collect", handle: async () => collectDiag() });
caps.register(makeExecCap(undefined, (a) => sendMsg("agent:audit", { cap: "exec", ...a })));
```

- [ ] **Step 2: Add the WS router case** (in the `switch (msg.type)` where `cmd:power` etc. live)
```ts
    case "agent:cap:request": {
      const { id, name, input } = msg.payload as { id: string; name: string; input: unknown };
      const ctx = { emitChunk: (stream: "stdout" | "stderr", data: string) => sendMsg("agent:cap:chunk", { id, stream, data }) };
      const result = await caps.dispatch(name, input, ctx);
      sendMsg("agent:cap:result", { id, ...result });
      break;
    }
```

- [ ] **Step 3: Enrich the metrics payload** — in the metrics tick, add `sysinfo: readSysInfo()` to the `agent:metrics` payload object (alongside `gpuUtil`, `vramUsed`, `vramTotal`, `tps`, …).

- [ ] **Step 4: Typecheck + full test**
Run: `npx tsc --noEmit -p packages/agent/tsconfig.json && npm test`
Expected: clean + green (no existing tests broken).

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2): wire caps into WS router + sysinfo in metrics"`

---

### Task 10: Prisma — AuditEvent + metric fields

**Files:**
- Modify: `prisma/schema.prisma`

**Interfaces:**
- Produces: `model AuditEvent { id String @id @default(cuid()); nodeId String; cap String; cmd String?; reason String?; code Int?; ts DateTime @default(now()); node Node @relation(fields:[nodeId], references:[id]) }` (+ back-relation `auditEvents AuditEvent[]` on `Node`). Add nullable columns to `MetricSnapshot`: `memAvailableMb Int?`, `psiMemSome10 Float?`, `pidCount Int?`, `fdAllocated Int?`, `sshdConns Int?`, `tempC Int?`. (Keep it small — the full `sysinfo` blob is broadcast live via SSE; only these summary scalars are persisted for trends.)

- [ ] **Step 1:** Add the model + fields to `schema.prisma` (exact fields above; add `auditEvents AuditEvent[]` to `model Node`).
- [ ] **Step 2:** Apply: `npm run db:generate && npm run db:push` (dev DB). Expected: no errors.
- [ ] **Step 3: Commit** — `git add prisma/schema.prisma && git commit -m "feat(agent-v2-srv): AuditEvent model + metric trend fields"`

---

### Task 11: Server — capability client (request→result correlation + timeout)

**Files:**
- Create: `packages/server/src/caps/cap-client.ts`
- Test: `packages/server/src/caps/cap-client.test.ts`

**Interfaces:**
- Produces:
  `export class CapClient { constructor(send:(nodeId:string,msg:unknown)=>void, opts?:{timeoutMs?:number}); invoke(nodeId:string, name:string, input:unknown, onChunk?:(c:{stream:string;data:string})=>void): Promise<{ok:boolean;data?:unknown;error?:string}>; onResult(payload:{id:string;ok:boolean;data?:unknown;error?:string}):void; onChunk(payload:{id:string;stream:string;data:string}):void }`
  `invoke` mints an `id` (injectable id fn; default a monotonic counter — NOT Math.random, to keep tests deterministic), calls `send(nodeId,{type:"agent:cap:request",payload:{id,name,input}})`, and returns a promise resolved by a matching `onResult`. A pending request with no result within `timeoutMs` rejects `{ok:false,error:"cap timeout"}`.

- [ ] **Step 1: Test**
```ts
import { describe, it, expect, vi } from "vitest";
import { CapClient } from "./cap-client.js";
describe("CapClient", () => {
  it("correlates a result back to the invoke promise", async () => {
    const sent: any[] = [];
    const c = new CapClient((_n, m) => sent.push(m));
    const p = c.invoke("node1", "diag.collect", null);
    const id = sent[0].payload.id;
    c.onResult({ id, ok: true, data: { memory: { totalMb: 124546 } } });
    await expect(p).resolves.toEqual({ ok: true, data: { memory: { totalMb: 124546 } } });
  });
  it("times out when no result arrives", async () => {
    vi.useFakeTimers();
    const c = new CapClient(() => {}, { timeoutMs: 1000 });
    const p = c.invoke("n", "exec", {});
    vi.advanceTimersByTime(1001);
    await expect(p).resolves.toEqual({ ok: false, error: "cap timeout" });
    vi.useRealTimers();
  });
});
```
- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement**
```ts
interface Pending { resolve: (r: { ok: boolean; data?: unknown; error?: string }) => void; timer: ReturnType<typeof setTimeout>; onChunk?: (c: { stream: string; data: string }) => void; }
export class CapClient {
  private pending = new Map<string, Pending>();
  private seq = 0;
  constructor(private send: (nodeId: string, msg: unknown) => void, private opts: { timeoutMs?: number } = {}) {}
  invoke(nodeId: string, name: string, input: unknown, onChunk?: (c: { stream: string; data: string }) => void):
    Promise<{ ok: boolean; data?: unknown; error?: string }> {
    const id = `cap-${++this.seq}`;
    return new Promise((resolve) => {
      const timer = setTimeout(() => { this.pending.delete(id); resolve({ ok: false, error: "cap timeout" }); },
        this.opts.timeoutMs ?? 30_000);
      this.pending.set(id, { resolve, timer, onChunk });
      this.send(nodeId, { type: "agent:cap:request", payload: { id, name, input } });
    });
  }
  onChunk(payload: { id: string; stream: string; data: string }): void {
    this.pending.get(payload.id)?.onChunk?.({ stream: payload.stream, data: payload.data });
  }
  onResult(payload: { id: string; ok: boolean; data?: unknown; error?: string }): void {
    const p = this.pending.get(payload.id);
    if (!p) return;
    clearTimeout(p.timer); this.pending.delete(payload.id);
    p.resolve({ ok: payload.ok, data: payload.data, error: payload.error });
  }
}
```
- [ ] **Step 4: Run** — PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat(agent-v2-srv): CapClient request/result correlation"`

---

### Task 12: Server — wire CapClient into agent-hub + REST endpoints + audit persist

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (construct a `CapClient` using `sendToAgent`; route `agent:cap:result`→`capClient.onResult`, `agent:cap:chunk`→`capClient.onChunk`; `agent:audit`→`prisma.auditEvent.create`; in `agent:metrics`, persist the new summary fields from `msg.payload.sysinfo` and SSE-broadcast the full `sysinfo`).
- Modify: `packages/server/src/routes/nodes.ts` (add `POST /:id/diag` and `POST /:id/exec`).
- Test: `packages/server/src/__tests__/integration/agent-v2.caps.test.ts`

**Interfaces:**
- Consumes: `CapClient` (Task 11), `AuditEvent` model (Task 10).
- Produces: `POST /api/nodes/:id/diag` → `{ ok, data }` (the bundle) or 502 on timeout/offline; `POST /api/nodes/:id/exec` (body `{ cmd, args?, reason, timeoutMs }`) → `400` if `reason` missing, else `{ ok, code, output }` + an `AuditEvent` row. Expose the `CapClient` on the app (`app.set("capClient", capClient)`) so routes reach it, mirroring the existing `app.set("agentHub", …)` pattern.

- [ ] **Step 1: Integration test (mirror deployments.dgxrun.test.ts harness)**
```ts
// per-suite sqlite; mount only the nodes router; inject a stub capClient
import request from "supertest";
import { describe, it, expect, beforeEach } from "vitest";
// ... standard integration bootstrap (see deployments.dgxrun.test.ts) ...
describe("POST /api/nodes/:id/diag + /exec", () => {
  it("diag returns the agent bundle", async () => {
    const capClient = { invoke: async () => ({ ok: true, data: { memory: { totalMb: 124546 } } }) };
    app.set("capClient", capClient);
    const res = await request(app).post(`/api/nodes/${nodeId}/diag`).send({});
    expect(res.status).toBe(200);
    expect(res.body.data.memory.totalMb).toBe(124546);
  });
  it("exec requires a reason", async () => {
    app.set("capClient", { invoke: async () => ({ ok: true, data: {} }) });
    const res = await request(app).post(`/api/nodes/${nodeId}/exec`).send({ cmd: "ls" });
    expect(res.status).toBe(400);
    expect(res.body.error).toMatch(/reason/i);
  });
});
```
- [ ] **Step 2: Run** — FAIL (routes missing).
- [ ] **Step 3: Implement the routes** (in `routes/nodes.ts`)
```ts
nodesRouter.post("/:id/diag", async (req, res) => {
  const capClient = req.app.get("capClient") as { invoke: (n: string, name: string, i: unknown) => Promise<{ ok: boolean; data?: unknown; error?: string }> };
  const r = await capClient.invoke(req.params.id, "diag.collect", null);
  if (!r.ok) return res.status(502).json({ error: r.error ?? "diag failed" });
  res.json({ ok: true, data: r.data });
});
nodesRouter.post("/:id/exec", async (req, res) => {
  const { cmd, args, reason, timeoutMs } = req.body ?? {};
  if (!reason || !String(reason).trim()) return res.status(400).json({ error: "reason required (audited)" });
  const capClient = req.app.get("capClient") as { invoke: (n: string, name: string, i: unknown) => Promise<{ ok: boolean; data?: unknown; error?: string }> };
  const r = await capClient.invoke(req.params.id, "exec", { cmd, args, reason, timeoutMs });
  if (!r.ok) return res.status(502).json({ error: r.error ?? "exec failed" });
  res.json({ ok: true, result: r.data });
});
```
- [ ] **Step 4:** Wire agent-hub: after constructing the hub, `const capClient = new CapClient((nodeId,msg)=>this.sendToAgent(nodeId,msg)); app.set("capClient", capClient);` and add router cases `agent:cap:result`→`capClient.onResult(msg.payload)`, `agent:cap:chunk`→`capClient.onChunk(msg.payload)`, `agent:audit`→`await prisma.auditEvent.create({ data:{ nodeId, cap:msg.payload.cap, cmd:msg.payload.cmd, reason:msg.payload.reason, code:msg.payload.code } }).catch(()=>{})`. In the `agent:metrics` handler, persist summary fields from `msg.payload.sysinfo` (guarded) and SSE-broadcast the full `sysinfo`.
- [ ] **Step 5: Run** — `npm test` green; both typechecks clean.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(agent-v2-srv): diag/exec REST + cap wiring + audit persist"`

---

### Task 13: Finalize — bump agent version + full green

**Files:** `packages/agent/package.json`

- [ ] **Step 1:** `./scripts/bump-agent-version.sh`
- [ ] **Step 2:** `npm test` (all green) + `npx tsc --noEmit -p packages/agent/tsconfig.json` + `.../server/tsconfig.json`.
- [ ] **Step 3: Commit** — `git add -A && git commit -m "chore(agent-v2): bump agent version for Phase 1"`

---

## Self-review (author checklist — completed)

- **Spec coverage:** capability registry (T7) ✓; fork-free `/proc` diag (T1-6) ✓ incl. the sshd-conn-by-state "money read" (T4); streaming-metrics enrichment + self-heal (T9 payload, T12 persist) ✓; audited reason-required exec (T8, T12) ✓; server cap client + REST + audit table (T10-12) ✓; GPU best-effort (T6) ✓; evolve-not-rewrite (all tasks modify/extend existing files) ✓.
- **Placeholder scan:** all steps carry real code + exact paths/commands. No TBD/TODO.
- **Type consistency:** `SysReadings` (T5) consumed by `Diag` (T6) and the metrics payload (T9); `Capability`/`CapContext` (T7) consumed by `makeExecCap` (T8) and the router (T9); `CapClient.invoke/onResult/onChunk` (T11) consumed by agent-hub + routes (T12). Names match across tasks.
- **Deferred to later phases (not gaps):** kmsg real-read impl (injected in prod; unit-scoped as `()=>[]`), disk deltas, mTLS/exec-arming — all Phase 2+.
