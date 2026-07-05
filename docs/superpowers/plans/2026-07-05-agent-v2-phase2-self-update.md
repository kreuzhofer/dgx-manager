# Agent v2 Phase 2 — Robust Self-Update — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the agent's blocking `execSync` self-update with a detached updater that keeps the agent heartbeating, swaps atomically, health-checks the new agent, and auto-rolls-back — so a slow/failed update never wedges the agent.

**Architecture:** A self-contained `updater.ts` (Node builtins only) runs **detached** (surviving the `dgx-agent` restart) and does download→extract→verify→swap→restart→health-check→rollback. The `cmd:update` handler is reduced to copying the updater to `/tmp` and spawning it detached+unref (non-blocking) — mirroring the existing `cmd:deprovision` pattern. The agent touches a `connected` marker on manager connect (the updater's health signal) and reports any prior update outcome.

**Tech Stack:** TypeScript (strict, ESM), Node 22, Vitest. Test pattern: pure helpers + `runUpdate(args, deps)` with injected IO (like `dgxrun-args`).

## Global Constraints

- TypeScript strict + ESM; `.js` import extensions in TS source.
- `updater.ts` uses **Node builtins only** (`node:child_process`, `node:fs`, `node:https`/`fetch`, `node:path`, `node:timers/promises`) — NO imports from other `packages/agent/src` modules and no npm deps, so it runs standalone from `/tmp` after `/opt/dgx-agent` is swapped out.
- `/opt/dgx-agent*` writes go through `sudo` (NOPASSWD, as the current handler does).
- Server unchanged. Reuse the existing `agent:update-status` WS message for outcome reporting (server already handles it) — do NOT add a new server-handled message.
- Agent change ⇒ `./scripts/bump-agent-version.sh` (once, final task). `npm test` green + `npx tsc --noEmit -p packages/agent/tsconfig.json` clean before each commit.
- Current handler to replace: `packages/agent/src/index.ts:1237-1281`. Detached-spawn precedent: the `cmd:deprovision` handler just below it. Connect path: `index.ts:132` ("Connected to manager"). Agent dir: `AGENT_DIR = join(__dirname, "..")` (`index.ts:34`).
- Commit prefix `feat(self-update):`.

## File structure

- `packages/agent/src/updater.ts` — pure helpers + `runUpdate(args, deps)` orchestration + `main()` real-deps wiring.
- `packages/agent/src/updater.test.ts` — unit (helpers) + integration (`runUpdate` paths).
- `packages/agent/src/index.ts` — rewrite `cmd:update`; connect-handler marker + result report.

---

### Task 1: Pure helpers — `verifyExtractedBundle`, `healthCheckPasses`

**Files:**
- Create: `packages/agent/src/updater.ts`
- Test: `packages/agent/src/updater.test.ts`

**Interfaces:**
- Produces:
  `export interface VerifyResult { ok: boolean; reason?: string }`
  `export function verifyExtractedBundle(dir: string, version: string, readPkg?: (p: string) => string): VerifyResult` — reads `<dir>/package.json`; ok iff it parses and `version` matches the arg. `readPkg` defaults to `readFileSync(p,"utf-8")`; a read/parse error → `{ok:false, reason}`.
  `export function healthCheckPasses(markerMtimeMs: number | null, restartMs: number, windowMs: number): boolean` — true iff `markerMtimeMs != null` and `markerMtimeMs >= restartMs` (the marker was written *after* the restart) and within the window isn't required here (caller bounds the poll loop).

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { verifyExtractedBundle, healthCheckPasses } from "./updater.js";

describe("verifyExtractedBundle", () => {
  it("ok when package.json version matches", () => {
    const r = verifyExtractedBundle("/opt/dgx-agent-new", "0.5.720", () => JSON.stringify({ version: "0.5.720" }));
    expect(r.ok).toBe(true);
  });
  it("fails on version mismatch", () => {
    const r = verifyExtractedBundle("/d", "0.5.720", () => JSON.stringify({ version: "0.5.719" }));
    expect(r.ok).toBe(false); expect(r.reason).toMatch(/version/i);
  });
  it("fails when package.json unreadable/unparseable", () => {
    expect(verifyExtractedBundle("/d", "0.5.720", () => { throw new Error("ENOENT"); }).ok).toBe(false);
    expect(verifyExtractedBundle("/d", "0.5.720", () => "not json").ok).toBe(false);
  });
});

describe("healthCheckPasses", () => {
  it("true when marker written after restart", () => {
    expect(healthCheckPasses(2000, 1000, 90000)).toBe(true);
  });
  it("false when marker missing", () => {
    expect(healthCheckPasses(null, 1000, 90000)).toBe(false);
  });
  it("false when marker is stale (pre-restart)", () => {
    expect(healthCheckPasses(500, 1000, 90000)).toBe(false);
  });
});
```

- [ ] **Step 2: Run** — Expected FAIL (not exported).
Run: `npx vitest run packages/agent/src/updater.test.ts`

- [ ] **Step 3: Implement (top of `updater.ts`)**
```ts
import { readFileSync } from "node:fs";

export interface VerifyResult { ok: boolean; reason?: string; }

/** Verify an extracted bundle dir: package.json parses and version matches. */
export function verifyExtractedBundle(
  dir: string, version: string, readPkg: (p: string) => string = (p) => readFileSync(p, "utf-8"),
): VerifyResult {
  let pkg: { version?: string };
  try { pkg = JSON.parse(readPkg(`${dir}/package.json`)); }
  catch (e) { return { ok: false, reason: `package.json unreadable: ${(e as Error).message}` }; }
  if (pkg.version !== version) return { ok: false, reason: `version mismatch: got ${pkg.version}, want ${version}` };
  return { ok: true };
}

/** New agent is healthy iff it wrote the connected marker AFTER the restart. */
export function healthCheckPasses(markerMtimeMs: number | null, restartMs: number, _windowMs: number): boolean {
  return markerMtimeMs != null && markerMtimeMs >= restartMs;
}
```

- [ ] **Step 4: Run** — Expected PASS.
- [ ] **Step 5: Commit** — `git add packages/agent/src/updater.ts packages/agent/src/updater.test.ts && git commit -m "feat(self-update): verifyExtractedBundle + healthCheckPasses helpers"`

---

### Task 2: `runUpdate` orchestration + `main()` real-deps

**Files:**
- Modify: `packages/agent/src/updater.ts`
- Test: `packages/agent/src/updater.test.ts`

**Interfaces:**
- Consumes: `verifyExtractedBundle`, `healthCheckPasses` (Task 1).
- Produces:
  `export interface UpdateDeps { download(url:string,dest:string):Promise<void>; extract(tarball:string,destDir:string):Promise<void>; verify(dir:string,version:string):VerifyResult; preserveNodeId():void; swap():void; restart():void; checkConnected():number|null; rollback():void; writeResult(r:{version:string;outcome:string;error?:string}):void; log(m:string):void; now():number; sleep(ms:number):Promise<void>; }`
  `export async function runUpdate(args:{bundleUrl:string;version:string}, deps:UpdateDeps): Promise<void>` — the flow below. Aborts pre-swap on download/extract/verify failure (never calls `swap`); after swap, a restart/health failure triggers `rollback`. Always calls `writeResult` exactly once.

- [ ] **Step 1: Write the failing test (integration, injected deps)**
```ts
import { runUpdate, type UpdateDeps } from "./updater.js";

function makeDeps(over: Partial<UpdateDeps> & { connectAt?: number }): { deps: UpdateDeps; calls: string[]; result: any } {
  const calls: string[] = []; let t = 1000; let result: any = null;
  const base: UpdateDeps = {
    download: async () => { calls.push("download"); },
    extract: async () => { calls.push("extract"); },
    verify: () => { calls.push("verify"); return { ok: true }; },
    preserveNodeId: () => calls.push("preserveNodeId"),
    swap: () => calls.push("swap"),
    restart: () => calls.push("restart"),
    checkConnected: () => (over.connectAt != null && t >= over.connectAt ? t : null),
    rollback: () => calls.push("rollback"),
    writeResult: (r) => { result = r; },
    log: () => {},
    now: () => t,
    sleep: async (ms) => { t += ms; },  // advance fake clock
  };
  return { deps: { ...base, ...over }, calls, get result() { return result; } } as any;
}

describe("runUpdate", () => {
  it("happy path: swap+restart, marker fresh -> success (no rollback)", async () => {
    const h = makeDeps({ connectAt: 1000 }); // marker present immediately >= restart time
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).toEqual(["download", "extract", "verify", "preserveNodeId", "swap", "restart"]);
    expect(h.result.outcome).toBe("success");
    expect(h.calls).not.toContain("rollback");
  });
  it("verify fail: aborts BEFORE swap (old agent untouched)", async () => {
    const h = makeDeps({ verify: () => ({ ok: false, reason: "bad" }) });
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).not.toContain("swap");
    expect(h.calls).not.toContain("restart");
    expect(h.result.outcome).toBe("failed");
  });
  it("health fail: swap+restart then no reconnect -> rollback", async () => {
    const h = makeDeps({ checkConnected: () => null }); // never connects
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).toContain("swap");
    expect(h.calls).toContain("rollback");
    expect(h.result.outcome).toBe("rolled-back");
  });
  it("download fail: aborts, no swap, failed result", async () => {
    const h = makeDeps({ download: async () => { throw new Error("net"); } });
    await runUpdate({ bundleUrl: "u", version: "1.0.0" }, h.deps);
    expect(h.calls).not.toContain("swap");
    expect(h.result.outcome).toBe("failed");
  });
});
```

- [ ] **Step 2: Run** — Expected FAIL (`runUpdate` not exported).

- [ ] **Step 3: Implement `runUpdate` (append to `updater.ts`)**
```ts
import { setTimeout as sleepP } from "node:timers/promises";

export interface UpdateDeps {
  download(url: string, dest: string): Promise<void>;
  extract(tarball: string, destDir: string): Promise<void>;
  verify(dir: string, version: string): VerifyResult;
  preserveNodeId(): void;
  swap(): void;
  restart(): void;
  checkConnected(): number | null;
  rollback(): void;
  writeResult(r: { version: string; outcome: string; error?: string }): void;
  log(m: string): void;
  now(): number;
  sleep(ms: number): Promise<void>;
}

const NEW_DIR = "/opt/dgx-agent-new";
const HEALTH_WINDOW_MS = 90_000;
const POLL_MS = 3_000;

export async function runUpdate(args: { bundleUrl: string; version: string }, deps: UpdateDeps): Promise<void> {
  const { bundleUrl, version } = args;
  const tarball = `/tmp/agent-bundle-${version}.tar.gz`;
  let swapped = false;
  deps.log(`[updater] starting update to v${version}`);
  try {
    await deps.download(bundleUrl, tarball);
    await deps.extract(tarball, NEW_DIR);
    const v = deps.verify(NEW_DIR, version);
    if (!v.ok) { deps.log(`[updater] verify failed: ${v.reason}`); deps.writeResult({ version, outcome: "failed", error: `verify: ${v.reason}` }); return; }
    deps.preserveNodeId();
    deps.swap(); swapped = true;
    deps.restart();
    const restartMs = deps.now();
    while (deps.now() - restartMs < HEALTH_WINDOW_MS) {
      await deps.sleep(POLL_MS);
      if (healthCheckPasses(deps.checkConnected(), restartMs, HEALTH_WINDOW_MS)) {
        deps.log("[updater] new agent reconnected — update ok");
        deps.writeResult({ version, outcome: "success" });
        return;
      }
    }
    deps.log("[updater] new agent did not reconnect in 90s — rolling back");
    deps.rollback();
    deps.writeResult({ version, outcome: "rolled-back", error: "new agent did not reconnect within 90s" });
  } catch (e) {
    const err = (e as Error).message;
    if (swapped) { try { deps.rollback(); } catch { /* */ } deps.log(`[updater] post-swap failure, rolled back: ${err}`); deps.writeResult({ version, outcome: "rolled-back", error: err }); }
    else { deps.log(`[updater] pre-swap failure (agent untouched): ${err}`); deps.writeResult({ version, outcome: "failed", error: err }); }
  }
}
```

- [ ] **Step 4: Run** — Expected PASS (4/4 + Task 1 tests).

- [ ] **Step 5: Add `main()` real deps + entrypoint (append).** These are the real IO wrappers; not unit-tested (verified by `runUpdate` tests + live). Marker/result paths match the agent side (Task 4).
```ts
import { execSync } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync, statSync, copyFileSync } from "node:fs";

const RUN_DIR = "/run/dgx-agent";
const MARKER = `${RUN_DIR}/connected`;
const RESULT = `${RUN_DIR}/update-result.json`;
const LOCK = `${RUN_DIR}/updating`;

function realDeps(nodeIdFile: string): UpdateDeps {
  return {
    download: async (url, dest) => { execSync(`curl -sfL -o "${dest}" "${url}"`, { timeout: 600_000 }); },
    extract: async (tarball, destDir) => {
      execSync(`sudo rm -rf "${destDir}" && sudo mkdir -p "${destDir}"`, { timeout: 15_000 });
      execSync(`sudo tar -xzf "${tarball}" -C "${destDir}/"`, { timeout: 300_000 });
    },
    verify: (dir, version) => verifyExtractedBundle(dir, version),
    preserveNodeId: () => { if (existsSync(nodeIdFile)) execSync(`sudo cp "${nodeIdFile}" ${NEW_DIR}/node-id`, { timeout: 5_000 }); },
    swap: () => { execSync("sudo rm -rf /opt/dgx-agent-old && sudo mv /opt/dgx-agent /opt/dgx-agent-old && sudo mv /opt/dgx-agent-new /opt/dgx-agent", { timeout: 15_000 }); },
    restart: () => { execSync("sudo systemctl restart dgx-agent", { timeout: 15_000 }); },
    checkConnected: () => { try { return statSync(MARKER).mtimeMs; } catch { return null; } },
    rollback: () => { execSync("sudo rm -rf /opt/dgx-agent-failed && sudo mv /opt/dgx-agent /opt/dgx-agent-failed && sudo mv /opt/dgx-agent-old /opt/dgx-agent && sudo systemctl restart dgx-agent", { timeout: 20_000 }); },
    writeResult: (r) => { try { mkdirSync(RUN_DIR, { recursive: true }); writeFileSync(RESULT, JSON.stringify(r)); } catch { /* */ } },
    log: (m) => { try { execSync(`logger -t dgx-updater ${JSON.stringify(m)}`); } catch { /* */ } console.log(m); },
    now: () => Date.now(),
    sleep: (ms) => sleepP(ms),
  };
}

// Entrypoint: `node updater.js <bundleUrl> <version> <nodeIdFile>`
if (process.argv[1] && process.argv[1].endsWith("updater.js")) {
  const [, , bundleUrl, version, nodeIdFile] = process.argv;
  const releaseLock = () => { try { execSync(`rm -f ${LOCK}`); } catch { /* */ } };
  runUpdate({ bundleUrl, version }, realDeps(nodeIdFile || "/opt/dgx-agent/node-id"))
    .finally(releaseLock);
}
```
Note: `Date.now()` is allowed here (agent runtime, not a Workflow script). Run `npx tsc --noEmit -p packages/agent/tsconfig.json` (clean) + `npm test` (green).

- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat(self-update): runUpdate orchestration + detached updater main"`

---

### Task 3: Rewrite `cmd:update` handler (non-blocking detached spawn)

**Files:**
- Modify: `packages/agent/src/index.ts:1237-1281`
- Test: `packages/agent/src/index.ts` has no unit harness for the handler; add a focused test file `packages/agent/src/update-launch.test.ts` for the extracted launch helper.

**Interfaces:**
- Produces: `export function launchUpdater(deps: { bundleUrl: string; version: string; updaterPath: string; nodeIdFile: string; copyFile(a:string,b:string):void; spawnDetached(cmd:string,args:string[]):void; lockExists():boolean; makeLock():void; tmpPath:string }): "launched" | "in-flight"` — a pure-ish helper (in a new small module `packages/agent/src/update-launch.ts`) so the launch logic is testable without a real spawn.

- [ ] **Step 1: Write the failing test**
```ts
import { describe, it, expect } from "vitest";
import { launchUpdater } from "./update-launch.js";

function harness(inFlight = false) {
  const calls: string[] = [];
  const deps = {
    bundleUrl: "u", version: "1.0.0", updaterPath: "/opt/dgx-agent/updater.js",
    nodeIdFile: "/opt/dgx-agent/node-id", tmpPath: "/tmp/dgx-updater-1.js",
    copyFile: (_a: string, _b: string) => calls.push("copy"),
    spawnDetached: (_c: string, _a: string[]) => calls.push("spawn"),
    lockExists: () => inFlight,
    makeLock: () => calls.push("lock"),
  };
  return { deps, calls };
}

describe("launchUpdater", () => {
  it("copies updater, makes lock, spawns detached, returns launched", () => {
    const h = harness(false);
    expect(launchUpdater(h.deps)).toBe("launched");
    expect(h.calls).toEqual(["lock", "copy", "spawn"]);
  });
  it("skips when an update is already in flight", () => {
    const h = harness(true);
    expect(launchUpdater(h.deps)).toBe("in-flight");
    expect(h.calls).toEqual([]);
  });
});
```

- [ ] **Step 2: Run** — FAIL.
Run: `npx vitest run packages/agent/src/update-launch.test.ts`

- [ ] **Step 3: Implement `update-launch.ts`**
```ts
export interface LaunchDeps {
  bundleUrl: string; version: string; updaterPath: string; nodeIdFile: string; tmpPath: string;
  copyFile(src: string, dest: string): void;
  spawnDetached(cmd: string, args: string[]): void;
  lockExists(): boolean;
  makeLock(): void;
}
/** Non-blocking: guard on the in-flight lock, copy the updater to /tmp, spawn it detached. */
export function launchUpdater(d: LaunchDeps): "launched" | "in-flight" {
  if (d.lockExists()) return "in-flight";
  d.makeLock();
  d.copyFile(d.updaterPath, d.tmpPath);
  d.spawnDetached("node", [d.tmpPath, d.bundleUrl, d.version, d.nodeIdFile]);
  return "launched";
}
```

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Wire into `index.ts`.** Replace the body of `case "cmd:update":` (lines 1238-1280) with:
```ts
    case "cmd:update": {
      const { bundleUrl, version } = msg.payload as { bundleUrl: string; version: string };
      const updaterPath = join(__dirname, "updater.js"); // dist/updater.js — same dir as index.js (verified: tsc outDir=dist, ExecStart runs /opt/dgx-agent/dist/index.js)
      const tmpPath = `/tmp/dgx-updater-${Date.now()}.js`;
      const RUN_DIR = "/run/dgx-agent";
      const outcome = launchUpdater({
        bundleUrl, version, updaterPath, nodeIdFile: NODE_ID_FILE, tmpPath,
        lockExists: () => existsSync(`${RUN_DIR}/updating`),
        makeLock: () => { try { execSync(`mkdir -p ${RUN_DIR} && touch ${RUN_DIR}/updating`); } catch { /* */ } },
        copyFile: (src, dest) => execSync(`cp "${src}" "${dest}"`),
        spawnDetached: (cmd, cargs) => { const c = spawn(cmd, cargs, { detached: true, stdio: "ignore" }); c.unref(); },
      });
      console.log(`[update] ${outcome === "launched" ? `launched detached updater for v${version}` : "update already in flight — ignored"}`);
      sendMsg("agent:update-status", { status: outcome === "launched" ? "downloading" : "in-flight", version });
      break;
    }
```
Add `import { launchUpdater } from "./update-launch.js";`. Confirm `join`, `spawn`, `execSync`, `existsSync`, `__dirname`, `NODE_ID_FILE` are already imported/defined in `index.ts` (they are — `__dirname` at line 32). **Path RESOLVED:** `tsc` has `outDir: dist`, the bundle ships `dist/`, and systemd runs `/opt/dgx-agent/dist/index.js`, so `__dirname = /opt/dgx-agent/dist` and `join(__dirname, "updater.js")` = `/opt/dgx-agent/dist/updater.js` — the correct location (do NOT use `AGENT_DIR`, which is one level too high).

- [ ] **Step 6: Verify.** `npx tsc --noEmit -p packages/agent/tsconfig.json` clean; `npm test` green.
- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat(self-update): non-blocking cmd:update via detached updater"`

---

### Task 4: Connect marker + result report + bump

**Files:**
- Modify: `packages/agent/src/index.ts` (the "Connected to manager" path, ~line 132)
- Modify: `packages/agent/package.json` (version bump)

**Interfaces:** none new.

- [ ] **Step 1: Touch the marker + report prior result on connect.** In the `ws.on("open", …)` handler, right after `console.log("Connected to manager");` (line 132), add:
```ts
    try {
      execSync("mkdir -p /run/dgx-agent && touch /run/dgx-agent/connected");
    } catch { /* marker best-effort */ }
    // Report the outcome of a just-completed self-update (written by the detached
    // updater), so a rollback/failure is visible instead of silent. Success needs
    // no report — the new version shows up in metrics.
    try {
      const rp = "/run/dgx-agent/update-result.json";
      if (existsSync(rp)) {
        const r = JSON.parse(readFileSync(rp, "utf-8")) as { version: string; outcome: string; error?: string };
        if (r.outcome !== "success") {
          sendMsg("agent:update-status", { status: "failed", version: r.version, error: `${r.outcome}: ${r.error ?? ""}` });
        }
        execSync(`rm -f ${rp} /run/dgx-agent/updating`); // clear result + release any lock
      }
    } catch { /* result report best-effort */ }
```
(`execSync`, `existsSync`, `readFileSync` are already imported.)

- [ ] **Step 2: Bump + verify.** `./scripts/bump-agent-version.sh`; `npx tsc --noEmit -p packages/agent/tsconfig.json` clean; `npm test` green.
- [ ] **Step 3: Commit** — `git add -A && git commit -m "feat(self-update): connected marker + report update outcome on connect; bump agent"`

---

## Self-review (author checklist — completed)

- **Spec coverage:** detached non-blocking updater (T2 main + T3 handler) ✓; pure verify + health helpers (T1) ✓; atomic swap + health-check + rollback + abort-pre-swap-untouched (T2 `runUpdate`, 4 tested paths) ✓; in-flight lock (T3 launch + T4 clear) ✓; connected marker + outcome report reusing `agent:update-status` (T4, server unchanged) ✓; generous timeouts / Node-builtins-only updater (T2 realDeps) ✓; agent bump (T4) ✓.
- **Placeholder scan:** all steps carry real code; the one genuine unknown — the bundle's emitted `updater.js` path — is called out as the correctness-critical check in T3 Step 5, not hand-waved.
- **Type consistency:** `UpdateDeps`/`runUpdate` (T2) match the test harness; `verifyExtractedBundle`/`healthCheckPasses` (T1) consumed by `runUpdate` (T2); `launchUpdater`/`LaunchDeps` (T3) match the handler wiring. Marker/result/lock paths (`/run/dgx-agent/{connected,update-result.json,updating}`) identical across updater (T2), handler (T3), connect (T4).
- **Rollout caveat (from spec):** this fix's own first roll still uses the OLD handler; robust from the next update on. `.36` = power-cycle + SSH-direct roll for this version. (Operational, not a task.)
