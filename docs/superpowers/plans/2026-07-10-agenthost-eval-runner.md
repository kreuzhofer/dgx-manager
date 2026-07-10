# Agenthost Eval Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all three benchmark kinds on `agenthost` (192.168.44.15) as systemd transient units, so a benchmark run outlives both the manager and the agent.

**Architecture:** A new agent `job.*` capability (`start`/`status`/`logs`/`cancel`) hands work to `systemd-run`, which owns the process. Every capability call is short-lived; the manager *polls* `job.logs(offset)` and `job.status` on a ~3s timer instead of holding a long stream. A `Node.role` of `"eval"` keeps the box out of the vLLM/dgxrun deploy path while still allowing Ollama.

**Tech Stack:** TypeScript (ES modules, strict), Express 5, Prisma/SQLite, vitest + @fast-check/vitest + supertest, systemd 255 on the eval node.

**Spec:** `docs/superpowers/specs/2026-07-10-agenthost-eval-runner-design.md`

## Global Constraints

- Tests: `DATABASE_URL="file:./prisma/dev.db" npx vitest run <file>` for one file; `DATABASE_URL="file:./prisma/dev.db" npm test` for all. **Never run a build concurrently with `npm test`** on the Pi — it causes spurious failures. Re-run alone before believing a failure.
- `vitest` does **not** typecheck. Typecheck with `npm run build --workspace=packages/server` and `--workspace=packages/agent`.
- Prisma: `DATABASE_URL="file:./prisma/dev.db" npm run db:push && DATABASE_URL="file:./prisma/dev.db" npm run db:generate`. Never commit `packages/server/src/generated/prisma` (gitignored).
- **Any edit under `packages/agent/src/` requires an agent version bump.** A PostToolUse hook bumps `packages/agent/package.json` on every edit; before committing, normalise it back to a **single** increment from the committed version.
- **Roll agents BEFORE rebuilding the server.** A server restart drops every agent's WS and triggers reconnect reconciliation.
- Bundle builds exceed the 10-minute background-task reaper. Use `setsid nohup … &` + a sentinel file, or build one arch at a time.
- All new columns must be nullable or defaulted (`db push --accept-data-loss` runs at container start).
- `CapClient.invoke` **never rejects**; on timeout it resolves `{ok: false, error: "cap timeout"}`. That is the `unknown` signal.
- **Critical invariant:** an inconclusive `job.status` must NEVER fail a run. Only "unit gone AND no exit file" is death.

---

## File Structure

**Phase 1 — Node role (ships standalone)**
- Create `packages/server/src/nodes/role.ts` — pure role predicates + admission message
- Create `packages/server/src/nodes/role.test.ts`
- Modify `prisma/schema.prisma` — `Node.role`
- Modify `packages/server/src/routes/deployments.ts` — server-side admission
- Create `packages/server/src/__tests__/integration/deployments.eval-node.test.ts`
- Modify `packages/dashboard/app/deployments/page.tsx` — picker filter

**Phase 2 — Agent job capability**
- Create `packages/agent/src/jobs/sh-quote.ts` (+ test) — shell quoting
- Create `packages/agent/src/jobs/job-spec.ts` (+ test) — unit name, job dir, wrapper script, systemd-run argv
- Create `packages/agent/src/jobs/systemctl-parse.ts` (+ test) — `parseSystemctlShow`
- Create `packages/agent/src/jobs/log-slice.ts` (+ test) — offset arithmetic
- Create `packages/agent/src/jobs/prune.ts` — job-dir retention
- Create `packages/agent/src/caps/job-cap.ts` (+ test) — the four capabilities
- Modify `packages/agent/src/index.ts` — register them

**Phase 3 — Onboard the eval node**
- Modify `packages/server/src/ssh/provisioner.ts` — eval profile
- Modify `packages/server/src/routes/nodes.ts` — accept `role` on create

**Phase 4 — Remote orchestration**
- Modify `prisma/schema.prisma` — `BenchmarkRun.runnerNodeId/jobUnit/logOffset`
- Create `packages/server/src/benchmarks/eval-node.ts` (+ test) — `resolveEvalNode`
- Create `packages/server/src/benchmarks/remote-runner.ts` (+ test) — `runTrackedRemote`, poll loop
- Modify `packages/server/src/benchmarks/orchestrator.ts` — transport selection
- Modify `packages/server/src/routes/benchmarks.ts` — 503, 409, persistence, cancel
- Modify `packages/server/src/index.ts` — boot reconciliation
- Create `packages/server/src/__tests__/integration/benchmarks.remote.test.ts`

**Phase 5 — Cutover**
- Modify `packages/dashboard/app/benchmarks/compare/page.tsx` — provenance warning

---

# Phase 1 — Node role

### Task 1: Pure role helpers + `Node.role` column

**Files:**
- Create: `packages/server/src/nodes/role.ts`
- Create: `packages/server/src/nodes/role.test.ts`
- Modify: `prisma/schema.prisma` (model `Node`, after `arch`)

**Interfaces:**
- Consumes: nothing
- Produces: `NodeRole = "gpu" | "eval"`, `isEvalNode(role: string | null | undefined): boolean`, `RUNTIMES_ALLOWED_ON_EVAL: readonly string[]`, `runtimeAllowedOnNode(role, runtime): boolean`, `evalNodeRejectionMessage(nodeName, runtime): string`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/nodes/role.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import {
  isEvalNode,
  runtimeAllowedOnNode,
  evalNodeRejectionMessage,
  RUNTIMES_ALLOWED_ON_EVAL,
} from "./role.js";

describe("isEvalNode", () => {
  it("recognises the eval role", () => {
    expect(isEvalNode("eval")).toBe(true);
  });

  // A legacy row predating the column, or an unknown value, must never be
  // mistaken for an eval node — that would silently exclude a real GPU node
  // from the deploy picker.
  it.each([["gpu"], [null], [undefined], [""], ["EVAL"], ["evaluator"]])(
    "treats %j as a normal node",
    (role) => {
      expect(isEvalNode(role as string | null | undefined)).toBe(false);
    },
  );
});

describe("runtimeAllowedOnNode", () => {
  it("permits ollama on an eval node", () => {
    expect(runtimeAllowedOnNode("eval", "ollama")).toBe(true);
  });

  it.each([["vllm"], ["dgxrun"], ["sglang"], [""], ["VLLM"]])(
    "refuses %j on an eval node",
    (runtime) => {
      expect(runtimeAllowedOnNode("eval", runtime)).toBe(false);
    },
  );

  // A gpu node hosts anything; the role only ever *restricts*.
  test.prop([fc.string()])("permits every runtime on a gpu node", (runtime) => {
    expect(runtimeAllowedOnNode("gpu", runtime)).toBe(true);
  });

  /** Invariant: the only role that restricts anything is "eval". */
  test.prop([fc.string().filter((r) => r !== "eval"), fc.string()])(
    "no role other than eval restricts any runtime",
    (role, runtime) => {
      expect(runtimeAllowedOnNode(role, runtime)).toBe(true);
    },
  );

  /** Invariant: on an eval node, allowed === membership of the allow-list. */
  test.prop([fc.string()])("eval allows exactly the allow-list", (runtime) => {
    expect(runtimeAllowedOnNode("eval", runtime)).toBe(
      RUNTIMES_ALLOWED_ON_EVAL.includes(runtime),
    );
  });
});

describe("evalNodeRejectionMessage", () => {
  it("names the node and the runtime, and says what is allowed", () => {
    const m = evalNodeRejectionMessage("agenthost", "vllm");
    expect(m).toContain("agenthost");
    expect(m).toContain("vllm");
    expect(m).toContain("ollama");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/nodes/role.test.ts`
Expected: FAIL — `Failed to resolve import "./role.js"`

- [ ] **Step 3: Write minimal implementation**

Create `packages/server/src/nodes/role.ts`:

```ts
/**
 * A node's role decides what may be *deployed* onto it. It never affects
 * metrics, provisioning, or agent management.
 *
 * `eval` exists for agenthost: a benchmark runner with no CUDA VRAM. It may
 * serve small Ollama models (embeddings) but must never host a vLLM/dgxrun
 * deployment. VRAM and arch admission would usually reject those anyway, but
 * relying on that is accidental, not intentional — this makes it explicit.
 */
export type NodeRole = "gpu" | "eval";

/** Runtimes an `eval` node is permitted to host. Everything else is refused. */
export const RUNTIMES_ALLOWED_ON_EVAL: readonly string[] = ["ollama"];

/** Only the exact string "eval" restricts a node. Legacy rows have no role. */
export function isEvalNode(role: string | null | undefined): boolean {
  return role === "eval";
}

/** True when `runtime` may be deployed onto a node with this `role`. */
export function runtimeAllowedOnNode(
  role: string | null | undefined,
  runtime: string,
): boolean {
  if (!isEvalNode(role)) return true;
  return RUNTIMES_ALLOWED_ON_EVAL.includes(runtime);
}

export function evalNodeRejectionMessage(nodeName: string, runtime: string): string {
  return (
    `Node "${nodeName}" has role "eval" and cannot host a "${runtime}" deployment. ` +
    `Allowed runtimes: ${RUNTIMES_ALLOWED_ON_EVAL.join(", ")}. ` +
    `Eval nodes run benchmarks; GPU nodes host models.`
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/nodes/role.test.ts`
Expected: PASS (all tests)

- [ ] **Step 5: Add the schema column**

In `prisma/schema.prisma`, inside `model Node`, immediately after the `arch String?` line, add:

```prisma
  // What may be DEPLOYED here. "gpu" (default) hosts anything; "eval" is a
  // benchmark runner (agenthost) that may serve Ollama but never vLLM/dgxrun.
  // See nodes/role.ts. Enforced server-side in POST /api/deployments.
  role            String           @default("gpu")
```

- [ ] **Step 6: Apply and regenerate**

```bash
DATABASE_URL="file:./prisma/dev.db" npm run db:push
DATABASE_URL="file:./prisma/dev.db" npm run db:generate
npm run build --workspace=packages/server
```
Expected: `db push` reports the database is in sync; `tsc` exits 0.

- [ ] **Step 7: Commit**

```bash
git add prisma/schema.prisma packages/server/src/nodes/role.ts packages/server/src/nodes/role.test.ts
git commit -m "feat(nodes): Node.role — eval nodes may host ollama, never vLLM/dgxrun"
```

---

### Task 2: Enforce the role in `POST /api/deployments`

**Files:**
- Modify: `packages/server/src/routes/deployments.ts` (import block; POST handler, before the VRAM admission check)
- Create: `packages/server/src/__tests__/integration/deployments.eval-node.test.ts`

**Interfaces:**
- Consumes: `runtimeAllowedOnNode`, `evalNodeRejectionMessage` from Task 1
- Produces: `POST /api/deployments` returns **400** when any target node has role `eval` and the runtime is not `ollama`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/deployments.eval-node.test.ts`:

```ts
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-evalnode-test-"));
process.env.DATABASE_URL = `file:${join(TMP_DIR, "test.db")}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "I understand this is a test database and consent to it being reset",
    },
    stdio: "ignore",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

function makeApp() {
  const app = express();
  app.use(express.json());
  app.set("agentHub", {
    getRecipes: () => [],
    getTrainingRecipes: () => [],
    getOllamaModels: () => [{ name: "nomic-embed-text", size: "274MB", description: "" }],
    isAgentOnline: () => true,
    onlineNodeIds: () => [] as string[],
    sendToAgent: () => {},
  });
  app.set("sshExec", async () => ({ code: 0, stdout: "false", stderr: "" }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function wipeAll() {
  await prisma.clusterNode.deleteMany({});
  await prisma.deployment.deleteMany({});
  await prisma.model.deleteMany({});
  await prisma.node.deleteMany({});
}

describe("eval-node deploy admission", () => {
  it("refuses a vllm deployment onto an eval node with 400", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "evalnode", name: "agenthost", ipAddress: "192.168.44.15",
        status: "online", role: "eval", arch: "amd64",
      },
    });

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeIds: ["evalnode"], recipeYaml: "runner: dgxrun\ncommand: vllm serve x\ncontainer: img\n" });

    expect(res.status).toBe(400);
    expect(res.body.error).toContain("agenthost");
    expect(res.body.error).toContain("ollama");
  });

  it("permits an ollama deployment onto an eval node", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "evalnode", name: "agenthost", ipAddress: "192.168.44.15",
        status: "online", role: "eval", arch: "amd64", vramTotal: 0,
      },
    });

    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeIds: ["evalnode"], runtime: "ollama", modelName: "nomic-embed-text" });

    expect(res.status).toBe(201);
  });

  // The role only restricts. A normal node is unaffected.
  it("leaves a gpu node unrestricted", async () => {
    await wipeAll();
    await prisma.node.create({
      data: {
        id: "gpunode", name: "dgx-1", ipAddress: "192.168.44.36",
        status: "online", vramTotal: 122_502, arch: "arm64",
      },
    });
    const res = await request(makeApp())
      .post("/api/deployments")
      .send({ nodeIds: ["gpunode"], recipeYaml: "runner: dgxrun\ncommand: vllm serve x\ncontainer: img\n" });

    expect(res.status).not.toBe(400);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/__tests__/integration/deployments.eval-node.test.ts -t "refuses a vllm"`
Expected: FAIL — status is 201/409, not 400.

- [ ] **Step 3: Write minimal implementation**

In `packages/server/src/routes/deployments.ts`, add to the import block:

```ts
import { runtimeAllowedOnNode, evalNodeRejectionMessage } from "../nodes/role.js";
```

Then, in the `POST "/"` handler, immediately **before** the `// Declarative maxoutmem recipe flag` block (i.e. before any agent dispatch or SSH work), insert:

```ts
  // Role admission. An `eval` node (agenthost) is a benchmark runner: it may
  // serve small Ollama models but must never host a vLLM/dgxrun deployment.
  // Enforced here, server-side, rather than relying on the dashboard picker or
  // on VRAM/arch admission happening to reject it.
  {
    const targetIds = recordNodeIds.length > 0 ? recordNodeIds : [headNodeId];
    const targets = await prisma.node.findMany({
      where: { id: { in: targetIds } },
      select: { id: true, name: true, role: true },
    });
    const effectiveRuntime = isOllama ? "ollama" : (isDgxrun ? "dgxrun" : "vllm");
    const offender = targets.find((n) => !runtimeAllowedOnNode(n.role, effectiveRuntime));
    if (offender) {
      return res.status(400).json({
        error: evalNodeRejectionMessage(offender.name, effectiveRuntime),
      });
    }
  }
```

> If `recordNodeIds` is not yet in scope at that point, use the variable holding the resolved target node ids for this deploy (the same list passed to `prisma.clusterNode.create`). Read the surrounding code and use the correct name — do not introduce a new one.

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/__tests__/integration/deployments.eval-node.test.ts`
Expected: PASS (3 tests)

- [ ] **Step 5: Typecheck and run the neighbouring suites**

```bash
npm run build --workspace=packages/server
DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/__tests__/integration/deployments.dgxrun.test.ts
```
Expected: `tsc` exits 0; dgxrun suite still passes (the new guard must not reject normal nodes).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/deployments.ts packages/server/src/__tests__/integration/deployments.eval-node.test.ts
git commit -m "feat(deployments): refuse vLLM/dgxrun onto an eval-role node"
```

---

### Task 3: Keep eval nodes out of the deploy picker

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (the `Node` interface; the node-selection list)

**Interfaces:**
- Consumes: `role` field, already returned by `GET /api/nodes` (Prisma passthrough)
- Produces: no exports; UI behaviour only

- [ ] **Step 1: Add `role` to the dashboard's `Node` type**

In `packages/dashboard/app/deployments/page.tsx`, find `interface Node {` and add:

```ts
  /** "gpu" (default) or "eval". An eval node may only host ollama. */
  role?: string;
```

- [ ] **Step 2: Filter the picker**

Find where `idleNodes` is rendered as selectable deploy targets. Wrap the source list:

```ts
  // An eval node (agenthost) may only host ollama. Server-side admission is the
  // real boundary (routes/deployments.ts); this just keeps it out of the picker
  // so nobody selects a node that will be refused.
  const selectableNodes = idleNodes.filter(
    (n) => runtimeMode === "ollama" || n.role !== "eval",
  );
```

Then use `selectableNodes` everywhere the picker previously used `idleNodes`.

- [ ] **Step 3: Typecheck**

```bash
cd packages/dashboard && npx tsc --noEmit -p tsconfig.json && cd -
```
Expected: no output (clean).

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "feat(dashboard): hide eval nodes from the vLLM/dgxrun deploy picker"
```

---

# Phase 2 — Agent job capability

### Task 4: Shell quoting

**Files:**
- Create: `packages/agent/src/jobs/sh-quote.ts`
- Create: `packages/agent/src/jobs/sh-quote.test.ts`

**Interfaces:**
- Produces: `shQuote(s: string): string`

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/jobs/sh-quote.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { execFileSync } from "child_process";
import { shQuote } from "./sh-quote.js";

describe("shQuote", () => {
  it("wraps a plain word", () => {
    expect(shQuote("hello")).toBe("'hello'");
  });

  it("escapes an embedded single quote", () => {
    expect(shQuote("it's")).toBe("'it'\\''s'");
  });

  /**
   * Invariant: a quoted string, passed through `sh -c`, comes back byte-for-byte.
   * This is the property that matters — the wrapper script interpolates argv, and
   * an escaping bug there is a shell-injection bug.
   */
  test.prop([fc.string({ minLength: 1 }).filter((s) => !s.includes("\0"))])(
    "round-trips through sh",
    (s) => {
      const out = execFileSync("sh", ["-c", `printf %s ${shQuote(s)}`], { encoding: "utf8" });
      expect(out).toBe(s);
    },
  );

  /** Invariant: shell metacharacters can never escape the quoting. */
  test.prop([fc.constantFrom(";", "&&", "|", "$(id)", "`id`", "\n", ">out")])(
    "neutralises metacharacters",
    (evil) => {
      const out = execFileSync("sh", ["-c", `printf %s ${shQuote(evil)}`], { encoding: "utf8" });
      expect(out).toBe(evil);
    },
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/sh-quote.test.ts`
Expected: FAIL — cannot resolve `./sh-quote.js`

- [ ] **Step 3: Write minimal implementation**

Create `packages/agent/src/jobs/sh-quote.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/sh-quote.test.ts`
Expected: PASS

- [ ] **Step 5: Commit** (do not bump the agent version yet — bump once at the end of Phase 2)

```bash
git add packages/agent/src/jobs/sh-quote.ts packages/agent/src/jobs/sh-quote.test.ts
git commit -m "feat(agent): POSIX shell quoting for job argv interpolation"
```

---

### Task 5: Job spec — unit name, job dir, wrapper script, systemd-run argv

**Files:**
- Create: `packages/agent/src/jobs/job-spec.ts`
- Create: `packages/agent/src/jobs/job-spec.test.ts`

**Interfaces:**
- Consumes: `shQuote` (Task 4)
- Produces:
  - `jobUnitName(runId: string): string` — throws on an unsafe runId
  - `jobDir(home: string, runId: string): string`
  - `buildWrapperScript(o: { argv: string[]; jobDir: string; outputDir: string; resultGlob: string }): string`
  - `buildSystemdRunArgv(o: { unit: string; jobDir: string; user: string; scriptPath: string }): string[]`

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/jobs/job-spec.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { jobUnitName, jobDir, buildWrapperScript, buildSystemdRunArgv } from "./job-spec.js";

describe("jobUnitName", () => {
  it("prefixes the runId", () => {
    expect(jobUnitName("cmremi4os0033")).toBe("dgxbench-cmremi4os0033");
  });

  // The runId reaches a shell command line and a systemd unit name. It comes
  // from the DB, but "it's a cuid" is an assumption, not a guarantee.
  it.each([["a b"], ["a;id"], ["../x"], ["a$(id)"], [""], ["a/b"], ["a\nb"]])(
    "rejects unsafe runId %j",
    (bad) => {
      expect(() => jobUnitName(bad)).toThrow(/unsafe/i);
    },
  );

  /** Invariant: every accepted unit name is alphanumeric + dash only. */
  test.prop([fc.stringMatching(/^[a-z0-9]{1,32}$/)])("accepts cuid-shaped ids", (id) => {
    expect(jobUnitName(id)).toMatch(/^dgxbench-[A-Za-z0-9]+$/);
  });
});

describe("jobDir", () => {
  it("is per-run under the agent state dir", () => {
    expect(jobDir("/home/daniel", "abc123")).toBe("/home/daniel/.dgx-agent/jobs/abc123");
  });
  it("rejects an unsafe runId", () => {
    expect(() => jobDir("/home/daniel", "../etc")).toThrow(/unsafe/i);
  });
});

describe("buildWrapperScript", () => {
  const script = buildWrapperScript({
    argv: ["uvx", "--from", "lm-eval[api]", "lm_eval", "--tasks", "ifeval"],
    jobDir: "/home/daniel/.dgx-agent/jobs/r1",
    outputDir: "/home/daniel/.dgx-agent/jobs/r1/out",
    resultGlob: "results_*.json",
  });

  it("redirects both streams to the log", () => {
    expect(script).toContain("> log 2>&1");
  });

  /**
   * ORDERING INVARIANT: `exit` must be written LAST, after result.json is in
   * place. The manager treats "exit exists" as "the job is finished and its
   * result is final". Writing exit first would let it read a result.json that
   * has not been copied yet.
   */
  it("writes the exit file after copying the result", () => {
    expect(script.indexOf("result.json")).toBeLessThan(script.lastIndexOf("> exit.tmp"));
  });

  it("writes the exit file atomically", () => {
    expect(script).toContain("mv exit.tmp exit");
  });

  it("preserves the command's exit code, not the copy's", () => {
    expect(script).toMatch(/code=\$\?/);
  });

  it("quotes every argv element", () => {
    expect(script).toContain("'lm-eval[api]'");
  });

  it("neutralises an injected metacharacter in argv", () => {
    const s = buildWrapperScript({
      argv: ["echo", "; rm -rf /"],
      jobDir: "/j", outputDir: "/j/out", resultGlob: "result.json",
    });
    expect(s).toContain("'; rm -rf /'");
    expect(s).not.toMatch(/^; rm -rf \//m);
  });
});

describe("buildSystemdRunArgv", () => {
  const argv = buildSystemdRunArgv({
    unit: "dgxbench-r1", jobDir: "/j", user: "daniel", scriptPath: "/j/cmd.sh",
  });

  it("runs under sudo -n because the agent is not root", () => {
    expect(argv.slice(0, 2)).toEqual(["sudo", "-n"]);
  });

  it("names the transient unit and drops privileges back to the agent user", () => {
    expect(argv).toContain("--unit=dgxbench-r1");
    expect(argv).toContain("-p");
    expect(argv).toContain("User=daniel");
  });

  it("does not use --collect, so the unit survives for status inspection", () => {
    expect(argv).not.toContain("--collect");
  });

  it("execs the wrapper via sh", () => {
    expect(argv.slice(-2)).toEqual(["/bin/sh", "/j/cmd.sh"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/job-spec.test.ts`
Expected: FAIL — cannot resolve `./job-spec.js`

- [ ] **Step 3: Write minimal implementation**

Create `packages/agent/src/jobs/job-spec.ts`:

```ts
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/job-spec.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/jobs/job-spec.ts packages/agent/src/jobs/job-spec.test.ts
git commit -m "feat(agent): job spec — unit name, wrapper script, systemd-run argv"
```

---

### Task 6: `parseSystemctlShow`

**Files:**
- Create: `packages/agent/src/jobs/systemctl-parse.ts`
- Create: `packages/agent/src/jobs/systemctl-parse.test.ts`

**Interfaces:**
- Produces: `JobStatus = {kind:"active"} | {kind:"exited"; code:number} | {kind:"missing"} | {kind:"unknown"; reason:string}`, `parseSystemctlShow(status: number | null, stdout: string, stderr: string): JobStatus`

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/jobs/systemctl-parse.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { parseSystemctlShow } from "./systemctl-parse.js";

const show = (o: Record<string, string>) =>
  Object.entries(o).map(([k, v]) => `${k}=${v}`).join("\n") + "\n";

describe("parseSystemctlShow", () => {
  it("reads a live unit as active", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "active", ExecMainStatus: "0" }), "");
    expect(r).toEqual({ kind: "active" });
  });

  it("treats activating as active", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "activating", ExecMainStatus: "0" }), "");
    expect(r.kind).toBe("active");
  });

  it("reads a finished unit's exit code", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "inactive", ExecMainStatus: "0" }), "");
    expect(r).toEqual({ kind: "exited", code: 0 });
  });

  it("reads a failed unit's exit code", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "loaded", ActiveState: "failed", ExecMainStatus: "1" }), "");
    expect(r).toEqual({ kind: "exited", code: 1 });
  });

  it("reports a not-found unit as missing", () => {
    const r = parseSystemctlShow(0, show({ LoadState: "not-found", ActiveState: "inactive", ExecMainStatus: "0" }), "");
    expect(r).toEqual({ kind: "missing" });
  });

  // The whole point. A timeout, a busy dbus, an empty answer: we do not know.
  it.each([
    [null, "", "spawn timeout"],
    [1, "", "Failed to connect to bus"],
    [0, "", ""],
    [0, "garbage without equals signs", ""],
  ])("returns unknown for status=%j stdout=%j", (status, stdout, stderr) => {
    expect(parseSystemctlShow(status as number | null, stdout, stderr).kind).toBe("unknown");
  });

  /**
   * INVARIANT — the one that matters. An inconclusive answer must NEVER be read
   * as a finished job. Reporting `exited(0)` for "we could not tell" would mark a
   * running benchmark complete and parse a result file that does not exist; the
   * mirror-image mistake (`exited(1)`) killed four healthy GLM-5.2 ranks.
   */
  test.prop([
    fc.oneof(fc.constant(null), fc.integer({ min: -1, max: 3 })),
    fc.string(),
    fc.string(),
  ])("never reports exited without an explicit ExecMainStatus", (status, stdout, stderr) => {
    const r = parseSystemctlShow(status, stdout, stderr);
    if (r.kind === "exited") {
      expect(stdout).toMatch(/ExecMainStatus=\d+/);
      expect(stdout).toMatch(/ActiveState=(inactive|failed|deactivating)/);
    }
  });

  /** Invariant: `missing` requires LoadState to positively say not-found. */
  test.prop([fc.string()])("never reports missing without LoadState=not-found", (stdout) => {
    const r = parseSystemctlShow(0, stdout, "");
    if (r.kind === "missing") expect(stdout).toContain("LoadState=not-found");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/systemctl-parse.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Write minimal implementation**

Create `packages/agent/src/jobs/systemctl-parse.ts`:

```ts
/**
 * Outcome of `systemctl show <unit> -p LoadState -p ActiveState -p ExecMainStatus`.
 *
 * The four cases are NOT interchangeable. `missing` means systemd positively said
 * the unit does not exist; `unknown` means we failed to ask (timeout, dead bus,
 * empty output). Collapsing `unknown` into anything else is the bug that tore down
 * four healthy dgxrun ranks on 2026-07-09 — here it would either kill an
 * 80-minute eval or mark it complete with no result.
 */
export type JobStatus =
  | { kind: "active" }
  | { kind: "exited"; code: number }
  | { kind: "missing" }
  | { kind: "unknown"; reason: string };

const ACTIVE_STATES = new Set(["active", "activating", "reloading"]);
const FINISHED_STATES = new Set(["inactive", "failed", "deactivating"]);

function parseProps(stdout: string): Map<string, string> {
  const m = new Map<string, string>();
  for (const line of stdout.split("\n")) {
    const eq = line.indexOf("=");
    if (eq > 0) m.set(line.slice(0, eq).trim(), line.slice(eq + 1).trim());
  }
  return m;
}

export function parseSystemctlShow(
  status: number | null,
  stdout: string,
  stderr: string,
): JobStatus {
  if (status !== 0) {
    return { kind: "unknown", reason: `systemctl exited ${status}: ${stderr.trim().slice(0, 200)}` };
  }
  const props = parseProps(stdout);
  const load = props.get("LoadState");
  const active = props.get("ActiveState");
  if (!load && !active) {
    return { kind: "unknown", reason: "systemctl produced no recognisable properties" };
  }
  if (load === "not-found") return { kind: "missing" };
  if (active && ACTIVE_STATES.has(active)) return { kind: "active" };
  if (active && FINISHED_STATES.has(active)) {
    const raw = props.get("ExecMainStatus");
    const code = raw === undefined ? NaN : Number(raw);
    if (!Number.isInteger(code)) {
      return { kind: "unknown", reason: `ActiveState=${active} but ExecMainStatus=${raw}` };
    }
    return { kind: "exited", code };
  }
  return { kind: "unknown", reason: `unrecognised ActiveState=${active ?? "<absent>"}` };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/systemctl-parse.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/jobs/systemctl-parse.ts packages/agent/src/jobs/systemctl-parse.test.ts
git commit -m "feat(agent): parseSystemctlShow — unknown is never exited"
```

---

### Task 7: Log offset arithmetic

**Files:**
- Create: `packages/agent/src/jobs/log-slice.ts`
- Create: `packages/agent/src/jobs/log-slice.test.ts`

**Interfaces:**
- Produces: `planRead(prevOffset: number, size: number): { from: number; to: number; truncated: boolean }`

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/jobs/log-slice.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { planRead } from "./log-slice.js";

describe("planRead", () => {
  it("reads the new tail", () => {
    expect(planRead(10, 25)).toEqual({ from: 10, to: 25, truncated: false });
  });

  it("reads nothing when the file has not grown", () => {
    expect(planRead(25, 25)).toEqual({ from: 25, to: 25, truncated: false });
  });

  // Log rotated or the job dir was recreated: start over rather than read garbage.
  it("restarts from zero when the file shrank", () => {
    expect(planRead(100, 20)).toEqual({ from: 0, to: 20, truncated: true });
  });

  it("clamps a negative stored offset", () => {
    expect(planRead(-5, 10)).toEqual({ from: 0, to: 10, truncated: true });
  });

  /** Invariant: the read window is always valid — 0 <= from <= to <= size. */
  test.prop([fc.integer({ min: -50, max: 500 }), fc.nat({ max: 500 })])(
    "always yields a valid window",
    (prev, size) => {
      const r = planRead(prev, size);
      expect(r.from).toBeGreaterThanOrEqual(0);
      expect(r.to).toBe(size);
      expect(r.from).toBeLessThanOrEqual(r.to);
    },
  );

  /**
   * Invariant: successive reads of a growing file reproduce it exactly — no lost
   * bytes, no duplicates. This is what makes reattach-after-manager-restart safe.
   */
  test.prop([fc.array(fc.string({ minLength: 1, maxLength: 8 }), { minLength: 1, maxLength: 20 })])(
    "concatenating successive reads reproduces the log",
    (chunks) => {
      let file = "";
      let offset = 0;
      let seen = "";
      for (const c of chunks) {
        file += c;
        const { from, to } = planRead(offset, file.length);
        seen += file.slice(from, to);
        offset = to;
      }
      expect(seen).toBe(file);
    },
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/log-slice.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Write minimal implementation**

Create `packages/agent/src/jobs/log-slice.ts`:

```ts
/**
 * Decide which byte range of the job log to return, given what the caller has
 * already consumed.
 *
 * The manager persists its offset, so this is also the reattach path after a
 * manager restart: it asks from the byte it last stored. If the file is smaller
 * than the stored offset the log was rotated or the job dir recreated — restart
 * from zero rather than slice past the end and emit garbage.
 */
export function planRead(
  prevOffset: number,
  size: number,
): { from: number; to: number; truncated: boolean } {
  const truncated = !Number.isFinite(prevOffset) || prevOffset < 0 || prevOffset > size;
  const from = truncated ? 0 : prevOffset;
  return { from, to: size, truncated };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs/log-slice.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/agent/src/jobs/log-slice.ts packages/agent/src/jobs/log-slice.test.ts
git commit -m "feat(agent): log offset arithmetic for reattachable job logs"
```

---

### Task 8: The `job.*` capabilities

**Files:**
- Create: `packages/agent/src/caps/job-cap.ts`
- Create: `packages/agent/src/caps/job-cap.test.ts`

**Interfaces:**
- Consumes: `jobUnitName`, `jobDir`, `buildWrapperScript`, `buildSystemdRunArgv` (Task 5); `parseSystemctlShow`, `JobStatus` (Task 6); `planRead` (Task 7)
- Produces: `makeJobCaps(deps?: JobCapDeps): Capability[]` registering `job.start`, `job.status`, `job.logs`, `job.cancel`
  - `job.start` input `{ runId, argv, resultGlob?, env? }` → `{ unit, jobDir, outputDir }`
  - `job.status` input `{ runId }` → `JobStatus` (with `exited` also derived from the `exit` file)
  - `job.logs` input `{ runId, offset }` → `{ chunk, nextOffset, truncated }`
  - `job.cancel` input `{ runId }` → `{ stopped: true }`

- [ ] **Step 1: Write the failing test**

Create `packages/agent/src/caps/job-cap.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { EventEmitter } from "events";
import { makeJobCaps, type JobCapDeps } from "./job-cap.js";
import type { Capability } from "./registry.js";

const noopCtx = { emitChunk: () => {} };

function capsByName(caps: Capability[]): Record<string, Capability> {
  return Object.fromEntries(caps.map((c) => [c.name, c]));
}

/** A spawn stub that records argv and reports the given exit code + stdout. */
function fakeSpawn(result: { code: number; stdout?: string; stderr?: string }, calls: string[][]) {
  return ((cmd: string, args: string[]) => {
    calls.push([cmd, ...args]);
    const child: any = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = () => {};
    setTimeout(() => {
      if (result.stdout) child.stdout.emit("data", Buffer.from(result.stdout));
      if (result.stderr) child.stderr.emit("data", Buffer.from(result.stderr));
      child.emit("close", result.code);
    }, 0);
    return child;
  }) as unknown as JobCapDeps["spawnFn"];
}

function deps(over: Partial<JobCapDeps> = {}): JobCapDeps {
  return {
    home: "/home/daniel",
    user: "daniel",
    spawnFn: fakeSpawn({ code: 0 }, []),
    writeFile: vi.fn(),
    mkdir: vi.fn(),
    readFileSlice: vi.fn(() => ({ chunk: "", size: 0 })),
    readTextFile: vi.fn(() => null),
    ...over,
  };
}

describe("job.start", () => {
  it("writes the wrapper script then launches a transient unit", async () => {
    const calls: string[][] = [];
    const writeFile = vi.fn();
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0 }, calls), writeFile })));

    const out = (await caps["job.start"].handle(
      { runId: "r1", argv: ["uvx", "lm_eval"], resultGlob: "results_*.json" },
      noopCtx,
    )) as { unit: string; jobDir: string };

    expect(out.unit).toBe("dgxbench-r1");
    expect(out.jobDir).toBe("/home/daniel/.dgx-agent/jobs/r1");
    // Script written before spawn.
    expect(writeFile).toHaveBeenCalled();
    const [scriptPath, script] = writeFile.mock.calls[0];
    expect(scriptPath).toBe("/home/daniel/.dgx-agent/jobs/r1/cmd.sh");
    expect(script).toContain("'lm_eval'");
    // Launched via sudo -n systemd-run.
    expect(calls[0].slice(0, 3)).toEqual(["sudo", "-n", "systemd-run"]);
    expect(calls[0]).toContain("--unit=dgxbench-r1");
  });

  it("rejects an unsafe runId before touching the shell", async () => {
    const calls: string[][] = [];
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0 }, calls) })));
    await expect(caps["job.start"].handle({ runId: "a;id", argv: ["x"] }, noopCtx)).rejects.toThrow(/unsafe/i);
    expect(calls).toHaveLength(0);
  });

  it("fails when systemd-run cannot start the unit", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 1, stderr: "sudo: a password is required" }, []) })),
    );
    await expect(caps["job.start"].handle({ runId: "r1", argv: ["x"] }, noopCtx)).rejects.toThrow(/password is required/);
  });
});

describe("job.status", () => {
  const showActive = "LoadState=loaded\nActiveState=active\nExecMainStatus=0\n";
  const showGone = "LoadState=not-found\nActiveState=inactive\nExecMainStatus=0\n";

  it("reports a live unit as active", async () => {
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0, stdout: showActive }, []) })));
    expect(await caps["job.status"].handle({ runId: "r1" }, noopCtx)).toEqual({ kind: "active" });
  });

  // A unit garbage-collected by systemd, but the wrapper left an exit file: the
  // job DID finish, and its code is authoritative.
  it("prefers the exit file when the unit is gone", async () => {
    const caps = capsByName(
      makeJobCaps(deps({
        spawnFn: fakeSpawn({ code: 0, stdout: showGone }, []),
        readTextFile: (p: string) => (p.endsWith("/exit") ? "0" : null),
      })),
    );
    expect(await caps["job.status"].handle({ runId: "r1" }, noopCtx)).toEqual({ kind: "exited", code: 0 });
  });

  it("reports missing only when the unit is gone AND there is no exit file", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0, stdout: showGone }, []), readTextFile: () => null })),
    );
    expect(await caps["job.status"].handle({ runId: "r1" }, noopCtx)).toEqual({ kind: "missing" });
  });

  /**
   * THE invariant. systemctl could not answer. The job may well be running. We
   * must say "unknown" — never "missing", never "exited". The manager skips the
   * tick; anything else would kill an 80-minute eval on one slow poll.
   */
  it("returns unknown when systemctl fails, even if an exit file is absent", async () => {
    const caps = capsByName(
      makeJobCaps(deps({
        spawnFn: fakeSpawn({ code: 1, stderr: "Failed to connect to bus" }, []),
        readTextFile: () => null,
      })),
    );
    const r = (await caps["job.status"].handle({ runId: "r1" }, noopCtx)) as { kind: string };
    expect(r.kind).toBe("unknown");
  });
});

describe("job.logs", () => {
  it("returns the tail from the caller's offset", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ readFileSlice: (_p, from, to) => ({ chunk: "world".slice(0, to - from), size: 10 }) })),
    );
    const r = (await caps["job.logs"].handle({ runId: "r1", offset: 5 }, noopCtx)) as {
      chunk: string; nextOffset: number; truncated: boolean;
    };
    expect(r.nextOffset).toBe(10);
    expect(r.truncated).toBe(false);
  });

  it("restarts at zero when the log shrank", async () => {
    const caps = capsByName(makeJobCaps(deps({ readFileSlice: () => ({ chunk: "abc", size: 3 }) })));
    const r = (await caps["job.logs"].handle({ runId: "r1", offset: 99 }, noopCtx)) as { truncated: boolean; nextOffset: number };
    expect(r.truncated).toBe(true);
    expect(r.nextOffset).toBe(3);
  });
});

describe("job.cancel", () => {
  it("stops the unit", async () => {
    const calls: string[][] = [];
    const caps = capsByName(makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 0 }, calls) })));
    await caps["job.cancel"].handle({ runId: "r1" }, noopCtx);
    expect(calls[0]).toEqual(["sudo", "-n", "systemctl", "stop", "dgxbench-r1"]);
  });

  // Cancelling an already-finished job is a no-op, not an error.
  it("is idempotent when the unit is already gone", async () => {
    const caps = capsByName(
      makeJobCaps(deps({ spawnFn: fakeSpawn({ code: 5, stderr: "not loaded" }, []) })),
    );
    await expect(caps["job.cancel"].handle({ runId: "r1" }, noopCtx)).resolves.toEqual({ stopped: true });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/caps/job-cap.test.ts`
Expected: FAIL — cannot resolve `./job-cap.js`

- [ ] **Step 3: Write minimal implementation**

Create `packages/agent/src/caps/job-cap.ts`:

```ts
import { spawn as realSpawn } from "child_process";
import { mkdirSync, writeFileSync, readFileSync, statSync, openSync, readSync, closeSync } from "fs";
import type { Capability } from "./registry.js";
import { jobUnitName, jobDir, buildWrapperScript, buildSystemdRunArgv } from "../jobs/job-spec.js";
import { parseSystemctlShow, type JobStatus } from "../jobs/systemctl-parse.js";
import { planRead } from "../jobs/log-slice.js";

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

function defaultDeps(): JobCapDeps {
  return {
    home: process.env.HOME ?? "/home/daniel",
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
  const d: JobCapDeps = { ...defaultDeps(), ...depsIn };

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
```

> Note: `job.logs` calls `readFileSlice(p.log, 0, 0)` purely to learn the size; the default impl returns `size` without reading bytes.

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/caps/job-cap.test.ts`
Expected: PASS

- [ ] **Step 5: Prune stale job directories**

The spec requires `job.start` to prune job dirs older than 14 days; `.15` has 870 GB but a
job dir holds a full lm-eval output tree, and nothing else ever deletes them.

Add to `packages/agent/src/caps/job-cap.test.ts`:

```ts
describe("job.start pruning", () => {
  it("removes job dirs older than the retention window and keeps recent ones", async () => {
    const removed: string[] = [];
    const now = 1_000_000_000_000;
    const day = 86_400_000;
    const caps = capsByName(makeJobCaps(deps({
      now: () => now,
      listJobDirs: () => [
        { path: "/home/daniel/.dgx-agent/jobs/old", mtimeMs: now - 15 * day },
        { path: "/home/daniel/.dgx-agent/jobs/fresh", mtimeMs: now - 1 * day },
      ],
      removeDir: (p: string) => removed.push(p),
    })));
    await caps["job.start"].handle({ runId: "r1", argv: ["x"] }, noopCtx);
    expect(removed).toEqual(["/home/daniel/.dgx-agent/jobs/old"]);
  });

  it("never lets a pruning failure abort the launch", async () => {
    const caps = capsByName(makeJobCaps(deps({
      listJobDirs: () => { throw new Error("EACCES"); },
    })));
    await expect(caps["job.start"].handle({ runId: "r1", argv: ["x"] }, noopCtx)).resolves.toBeTruthy();
  });
});
```

Add a pure helper `packages/agent/src/jobs/prune.ts`:

```ts
/** Job dirs whose mtime is older than the retention window. Pure. */
export const JOB_RETENTION_MS = 14 * 86_400_000;

export function stalePaths(
  entries: { path: string; mtimeMs: number }[],
  nowMs: number,
  retentionMs: number = JOB_RETENTION_MS,
): string[] {
  return entries.filter((e) => nowMs - e.mtimeMs > retentionMs).map((e) => e.path);
}
```

Extend `JobCapDeps` with `now(): number`, `listJobDirs(): {path:string; mtimeMs:number}[]`, `removeDir(path: string): void`
(defaults: `Date.now`, `readdirSync`+`statSync` over `${home}/.dgx-agent/jobs`, `rmSync(p,{recursive:true,force:true})`),
and at the top of `job.start`'s handler:

```ts
      // Best-effort housekeeping. A pruning failure must never stop a benchmark.
      try {
        for (const p of stalePaths(d.listJobDirs(), d.now())) d.removeDir(p);
      } catch (e) {
        // eslint-disable-next-line no-console
        console.warn(`[job.start] prune skipped: ${(e as Error).message}`);
      }
```

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/caps/job-cap.test.ts`
Expected: PASS

- [ ] **Step 6: Register the capabilities**

In `packages/agent/src/index.ts`, add the import:

```ts
import { makeJobCaps } from "./caps/job-cap.js";
```

and next to the existing `caps.register(makeExecCap(...))` line:

```ts
// Long-running benchmark jobs, owned by systemd so they outlive agent rolls.
for (const c of makeJobCaps()) caps.register(c);
```

- [ ] **Step 7: Typecheck, normalise the agent version, run the agent suites**

```bash
npm run build --workspace=packages/agent
# The PostToolUse hook bumped the version on every edit. Normalise to ONE bump:
COMMITTED=$(git show HEAD:packages/agent/package.json | python3 -c "import sys,json;print(json.load(sys.stdin)['version'])")
echo "committed was $COMMITTED — set package.json to exactly one patch above it"
DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/agent/src/jobs packages/agent/src/caps
```
Expected: `tsc` exits 0; all job + cap tests pass.

- [ ] **Step 8: Full suite, alone**

Run: `DATABASE_URL="file:./prisma/dev.db" npm test`
Expected: all green. **Do not run any build at the same time.**

- [ ] **Step 9: Commit**

```bash
git add packages/agent/src/caps/job-cap.ts packages/agent/src/caps/job-cap.test.ts \
        packages/agent/src/jobs/prune.ts packages/agent/src/index.ts packages/agent/package.json
git commit -m "feat(agent): job.* capability — systemd-owned jobs that outlive the agent"
```

- [ ] **Step 10: Build bundles and roll agents (BEFORE any server rebuild)**

```bash
./scripts/build-agent-bundles.sh   # exceeds the 10-min reaper: use setsid nohup + sentinel
# Verify the bundle carries the new code and version BEFORE rolling:
tar -xzOf packages/server/agent-bundles/agent-bundle-arm64.tar.gz ./package.json | grep version
tar -tzf packages/server/agent-bundles/agent-bundle-arm64.tar.gz | grep -c 'dist/caps/job-cap.js'
```
Then roll workers first, head last, via `POST /api/nodes/:id/update-agent`, verifying the model endpoint answers between each.

---

# Phase 3 — Onboard the eval node

### Task 9: Eval provisioning profile

**Files:**
- Modify: `packages/server/src/ssh/provisioner.ts`
- Modify: `packages/server/src/routes/nodes.ts` (accept `role` on `POST /api/nodes`)

**Interfaces:**
- Consumes: `NodeRole` (Task 1)
- Produces: `provisionNode(..., { profile: "gpu" | "eval" })`; an `eval` profile installs Node.js, the agent, and `uv` — nothing else

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/ssh/provisioner.eval-profile.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { evalProfileSteps, gpuProfileSteps } from "./provisioner.js";

describe("eval provisioning profile", () => {
  const steps = evalProfileSteps().map((s) => s.name);

  it("installs uv, which the benchmark runner needs", () => {
    expect(steps).toContain("install-uv");
  });

  it("asserts passwordless sudo up front rather than on the first benchmark", () => {
    expect(steps).toContain("assert-nopasswd-sudo");
  });

  // agenthost runs a hand-installed ollama serving embeddings. Provisioning must
  // not reinstall, restart, or reconfigure it.
  it("never touches ollama, sparkrun, or the nvidia container toolkit", () => {
    const joined = JSON.stringify(evalProfileSteps());
    for (const forbidden of ["ollama", "sparkrun", "nvidia-container-toolkit"]) {
      expect(joined.toLowerCase()).not.toContain(forbidden);
    }
  });

  it("still installs the agent", () => {
    expect(steps).toContain("install-agent");
  });

  it("is a strict subset of the gpu profile plus install-uv", () => {
    const gpu = new Set(gpuProfileSteps().map((s) => s.name));
    const extra = steps.filter((s) => !gpu.has(s));
    expect(extra).toEqual(["install-uv"]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/ssh/provisioner.eval-profile.test.ts`
Expected: FAIL — `evalProfileSteps` is not exported.

- [ ] **Step 3: Write minimal implementation**

Read `packages/server/src/ssh/provisioner.ts` and identify the existing ordered list of provisioning steps. Extract it into an exported `gpuProfileSteps()` returning `{ name: string; cmd: string }[]` (no behaviour change). Then add:

```ts
/**
 * Provisioning for an `eval` node (agenthost): a benchmark runner, not a model
 * host. It gets the agent and `uv`, and nothing else.
 *
 * Deliberately absent: sparkrun (it never launches a deployment), the nvidia
 * container toolkit (it has no CUDA GPU), and Ollama — agenthost runs a
 * hand-installed ollama serving embeddings, and re-provisioning it would stomp a
 * working service.
 *
 * `assert-nopasswd-sudo` fails loudly at onboarding: the job capability runs
 * `sudo -n systemd-run`, and discovering a missing sudoers rule on the first
 * 80-minute benchmark would be a poor time to find out.
 */
export function evalProfileSteps(): { name: string; cmd: string }[] {
  const shared = gpuProfileSteps().filter((s) =>
    ["install-node", "install-agent"].includes(s.name),
  );
  return [
    { name: "assert-nopasswd-sudo", cmd: "sudo -n true" },
    ...shared,
    {
      name: "install-uv",
      cmd:
        "command -v uv >/dev/null 2>&1 || " +
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    },
  ];
}
```

> Match `gpuProfileSteps()`'s real step names when filtering. If the existing steps are not named, name them as part of this task — the eval profile is defined by exclusion, so the names must exist.

Then have `provisionNode` select the step list from `node.role`.

In `packages/server/src/routes/nodes.ts`, accept an optional `role` on `POST /api/nodes`, validating it against `["gpu", "eval"]` and defaulting to `"gpu"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/ssh/provisioner.eval-profile.test.ts`
Expected: PASS

- [ ] **Step 5: Typecheck + commit**

```bash
npm run build --workspace=packages/server
git add packages/server/src/ssh/provisioner.ts packages/server/src/ssh/provisioner.eval-profile.test.ts packages/server/src/routes/nodes.ts
git commit -m "feat(provisioner): eval-node profile — agent + uv only, never ollama/sparkrun"
```

---

### Task 10: Onboard agenthost (manual, verified)

**Files:** none (operational)

- [ ] **Step 1: Rebuild and restart the server** (agents were already rolled in Task 8)

```bash
setsid nohup docker compose build server > /tmp/build.log 2>&1 &
# wait for completion, then:
MANAGER_ADVERTISE_HOST=192.168.44.14 SSH_USER=daniel docker compose up -d --no-deps server
```

- [ ] **Step 2: Add the node with role=eval**

```bash
curl -s -X POST http://localhost:4000/api/nodes \
  -H 'Content-Type: application/json' \
  -d '{"name":"agenthost","ipAddress":"192.168.44.15","role":"eval"}' | python3 -m json.tool
```
Expected: a node row with `"role": "eval"`.

- [ ] **Step 3: Provision it, then verify the agent registered**

Trigger provisioning through the existing endpoint, then:

```bash
curl -s http://localhost:4000/api/nodes | python3 -c "
import sys,json
for n in json.load(sys.stdin):
    if n['name']=='agenthost':
        print('status', n['status'], '| role', n['role'], '| agent', n.get('agentVersion'), '| vram', n.get('vramTotal'))"
```
Expected: `status online`, `role eval`, an agent version, `vram None` (no CUDA GPU — this is correct, not a fault).

- [ ] **Step 4: Verify the guardrails hold on the real box**

```bash
# uv installed
ssh 192.168.44.15 'command -v uv && uv --version'
# the hand-installed ollama is untouched and still serving
ssh 192.168.44.15 'systemctl is-active ollama && ollama ps | tail -1'
# sudo -n systemd-run works for the agent user
ssh 192.168.44.15 'sudo -n systemd-run --unit=dgxbench-probe0 /bin/sh -c "echo ok" && sleep 1 && systemctl show dgxbench-probe0 -p ActiveState -p ExecMainStatus; sudo -n systemctl reset-failed dgxbench-probe0 2>/dev/null; true'
```
Expected: `uv` present; ollama `active` and still `100% CPU`; the probe unit reports `ExecMainStatus=0`.

- [ ] **Step 5: Verify the deploy guard end to end**

```bash
NODE=$(curl -s http://localhost:4000/api/nodes | python3 -c "import sys,json;print([n['id'] for n in json.load(sys.stdin) if n['name']=='agenthost'][0])")
curl -s -X POST http://localhost:4000/api/deployments -H 'Content-Type: application/json' \
  -d "{\"nodeIds\":[\"$NODE\"],\"recipeFile\":\"@dgxrun/glm-5.2-quanttrio-unpruned-dcp2\"}" | head -c 200
```
Expected: **HTTP 400** naming `agenthost` and `ollama`. The eval node must refuse a dgxrun deploy.

---

# Phase 4 — Remote orchestration

### Task 11: `BenchmarkRun` columns + `resolveEvalNode` + 503/409 guards

**Files:**
- Modify: `prisma/schema.prisma` (model `BenchmarkRun`)
- Create: `packages/server/src/benchmarks/eval-node.ts`
- Create: `packages/server/src/benchmarks/eval-node.test.ts`
- Modify: `packages/server/src/routes/benchmarks.ts`
- Create: `packages/server/src/__tests__/integration/benchmarks.remote.test.ts`

**Interfaces:**
- Consumes: `isEvalNode` (Task 1)
- Produces:
  - `resolveEvalNode(nodes: {id,name,role,status}[], explicitId?: string): {ok:true; nodeId:string} | {ok:false; reason:"none"|"ambiguous"; detail:string}`
  - `BenchmarkRun.runnerNodeId: String?`, `.jobUnit: String?`, `.logOffset: Int @default(0)`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/eval-node.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { resolveEvalNode } from "./eval-node.js";

const n = (id: string, role: string, status = "online") => ({ id, name: id, role, status });

describe("resolveEvalNode", () => {
  it("picks the single online eval node", () => {
    expect(resolveEvalNode([n("a", "gpu"), n("b", "eval")])).toEqual({ ok: true, nodeId: "b" });
  });

  it("fails when no eval node is online", () => {
    const r = resolveEvalNode([n("a", "gpu"), n("b", "eval", "offline")]);
    expect(r).toMatchObject({ ok: false, reason: "none" });
  });

  it("fails when there are none at all", () => {
    expect(resolveEvalNode([n("a", "gpu")])).toMatchObject({ ok: false, reason: "none" });
  });

  // Silently picking the first would make a run's provenance depend on row order —
  // exactly what runnerNodeId exists to prevent.
  it("refuses to guess between two eval nodes", () => {
    const r = resolveEvalNode([n("a", "eval"), n("b", "eval")]);
    expect(r).toMatchObject({ ok: false, reason: "ambiguous" });
    expect((r as { detail: string }).detail).toContain("EVAL_NODE_ID");
  });

  it("honours an explicit EVAL_NODE_ID", () => {
    expect(resolveEvalNode([n("a", "eval"), n("b", "eval")], "b")).toEqual({ ok: true, nodeId: "b" });
  });

  it("rejects an explicit id that is not an online eval node", () => {
    expect(resolveEvalNode([n("a", "eval"), n("b", "gpu")], "b")).toMatchObject({ ok: false });
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks/eval-node.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Write minimal implementation**

Create `packages/server/src/benchmarks/eval-node.ts`:

```ts
import { isEvalNode } from "../nodes/role.js";

export interface EvalNodeCandidate { id: string; name: string; role: string | null; status: string }

export type EvalNodeResolution =
  | { ok: true; nodeId: string }
  | { ok: false; reason: "none" | "ambiguous"; detail: string };

/**
 * Pick the node that runs benchmarks. Exactly one online `eval` node is expected.
 *
 * Ambiguity is an error, not a coin flip: silently taking the first would make a
 * run's provenance depend on row ordering, and `BenchmarkRun.runnerNodeId` exists
 * precisely so that throughput numbers can be trusted to a host.
 */
export function resolveEvalNode(
  nodes: EvalNodeCandidate[],
  explicitId?: string,
): EvalNodeResolution {
  const online = nodes.filter((n) => isEvalNode(n.role) && n.status === "online");
  if (explicitId) {
    const hit = online.find((n) => n.id === explicitId);
    return hit
      ? { ok: true, nodeId: hit.id }
      : { ok: false, reason: "none", detail: `EVAL_NODE_ID=${explicitId} is not an online eval node` };
  }
  if (online.length === 0) {
    return { ok: false, reason: "none", detail: "no online node with role \"eval\"" };
  }
  if (online.length > 1) {
    return {
      ok: false,
      reason: "ambiguous",
      detail: `multiple online eval nodes (${online.map((n) => n.name).join(", ")}); set EVAL_NODE_ID`,
    };
  }
  return { ok: true, nodeId: online[0].id };
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks/eval-node.test.ts`
Expected: PASS

- [ ] **Step 5: Add the schema columns**

In `prisma/schema.prisma`, inside `model BenchmarkRun`, after `kind`:

```prisma
  // WHERE this run executed. null = a legacy run on the manager (the Pi).
  // Provenance, not bookkeeping: throughput measures tok/s and TTFR from the
  // client, so runs from different hosts are not comparable.
  runnerNodeId    String?
  jobUnit         String?   // systemd transient unit on the runner
  logOffset       Int      @default(0)  // bytes of the remote log already persisted
```

Apply:
```bash
DATABASE_URL="file:./prisma/dev.db" npm run db:push
DATABASE_URL="file:./prisma/dev.db" npm run db:generate
```

- [ ] **Step 6: Add the 503 + 409 guards**

Create `packages/server/src/__tests__/integration/benchmarks.remote.test.ts` covering:

```ts
// 1. POST /api/benchmarks with no online eval node -> 503, and NO BenchmarkRun row created.
// 2. POST twice for the same deploymentId while the first is `running` -> 409 naming the first runId.
// 3. POST happy path -> 201, row has runnerNodeId set and jobUnit non-null.
```

Use the same harness idiom as `deployments.dgxrun.test.ts`: per-suite SQLite via `mkdtempSync`, `DATABASE_URL` set before importing prisma, `npx prisma db push --force-reset` with `PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION`, an Express app mounting only `benchmarksRouter`, and `app.set("agentHub", stubHub)` where the stub exposes a `capClient` whose `invoke` returns canned `{ok:true,data:{unit:"dgxbench-x",jobDir:"/j"}}`.

In `packages/server/src/routes/benchmarks.ts`, in the `POST "/"` handler, **before** `prisma.benchmarkRun.create`:

```ts
  // One benchmark per deployment at a time. Two runs share the model's batch
  // slots, so their throughput and latency numbers describe neither run.
  const inFlight = await prisma.benchmarkRun.findFirst({
    where: { deploymentId, status: { in: ["pending", "running"] } },
    select: { id: true },
  });
  if (inFlight) {
    return res.status(409).json({
      error: `benchmark ${inFlight.id} is already running against this deployment`,
      runId: inFlight.id,
    });
  }

  // Resolve the eval runner. Fail fast: a throughput number whose runner you
  // cannot identify is worse than no number, so never fall back to the manager.
  const runnerNodes = await prisma.node.findMany({ select: { id: true, name: true, role: true, status: true } });
  const resolved = resolveEvalNode(runnerNodes, process.env.EVAL_NODE_ID);
  if (!resolved.ok) {
    return res.status(503).json({ error: `eval runner unavailable: ${resolved.detail}` });
  }
  const runnerNodeId = resolved.nodeId;
```

Persist `runnerNodeId` in the `benchmarkRun.create` data block.

- [ ] **Step 7: Run the tests, typecheck, commit**

```bash
DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/__tests__/integration/benchmarks.remote.test.ts
npm run build --workspace=packages/server
git add prisma/schema.prisma packages/server/src/benchmarks/eval-node.ts packages/server/src/benchmarks/eval-node.test.ts \
        packages/server/src/routes/benchmarks.ts packages/server/src/__tests__/integration/benchmarks.remote.test.ts
git commit -m "feat(benchmarks): resolve the eval runner; 503 when absent, 409 when busy"
```

---

### Task 12: `runTrackedRemote` — the poll loop

**Files:**
- Create: `packages/server/src/benchmarks/remote-runner.ts`
- Create: `packages/server/src/benchmarks/remote-runner.test.ts`

**Interfaces:**
- Consumes: `JobStatus` shape from the agent (`{kind:"active"|"exited"|"missing"|"unknown"}`); `CapClient.invoke`
- Produces:
  - `type CapInvoker = (nodeId: string, name: string, input: unknown) => Promise<{ok:boolean; data?:unknown; error?:string}>`
  - `nextPollAction(status: JobStatus, hasExitFile: boolean): "continue" | "finish" | "fail"`
  - `runTrackedRemote(o: RemoteRunOpts): Promise<{exitCode: number|null; rawOutput: string|null}>`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/remote-runner.test.ts`:

```ts
import { describe, expect, it, vi } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { nextPollAction, runTrackedRemote } from "./remote-runner.js";

describe("nextPollAction", () => {
  it("keeps polling a live job", () => {
    expect(nextPollAction({ kind: "active" }, false)).toBe("continue");
  });

  it("finishes on a clean exit", () => {
    expect(nextPollAction({ kind: "exited", code: 0 }, true)).toBe("finish");
  });

  it("finishes on a non-zero exit — the run failed, the poll did not", () => {
    expect(nextPollAction({ kind: "exited", code: 1 }, true)).toBe("finish");
  });

  it("fails only when the unit is gone and no exit file exists", () => {
    expect(nextPollAction({ kind: "missing" }, false)).toBe("fail");
  });

  /**
   * THE invariant. A cap timeout, a busy box, a dropped WS: we could not ask. The
   * job is almost certainly still running. Skip the tick. Failing here would kill
   * an 80-minute eval on one slow poll — the same absent-vs-unknown conflation
   * that tore down four healthy GLM-5.2 ranks on 2026-07-09.
   */
  test.prop([fc.boolean(), fc.string()])(
    "an unknown status never fails or finishes a run",
    (hasExit, reason) => {
      expect(nextPollAction({ kind: "unknown", reason }, hasExit)).toBe("continue");
    },
  );

  /** Invariant: only an explicit `exited` ever finishes a run. */
  test.prop([
    fc.oneof(
      fc.constant({ kind: "active" as const }),
      fc.constant({ kind: "missing" as const }),
      fc.string().map((reason) => ({ kind: "unknown" as const, reason })),
    ),
    fc.boolean(),
  ])("never finishes without an exited status", (status, hasExit) => {
    expect(nextPollAction(status, hasExit)).not.toBe("finish");
  });
});

describe("runTrackedRemote", () => {
  const baseOpts = {
    runId: "r1", nodeId: "n1", argv: ["uvx", "lm_eval"], resultGlob: "results_*.json",
    pollMs: 1, onLog: () => {}, onOffset: () => {},
  };

  it("starts the job, drains logs, and returns the result", async () => {
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "dgxbench-r1", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "hello\n", nextOffset: 6, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
      if (name === "job.result") return { ok: true, data: { raw: '{"results":{}}' } };
      throw new Error("unexpected " + name);
    });
    const lines: string[] = [];
    const r = await runTrackedRemote({ ...baseOpts, invoke, onLog: (l) => lines.push(l) });
    expect(r.exitCode).toBe(0);
    expect(r.rawOutput).toBe('{"results":{}}');
    expect(lines).toContain("hello");
  });

  // A cap timeout mid-run must not end the run.
  it("survives an inconclusive status and keeps polling", async () => {
    let statusCalls = 0;
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") {
        statusCalls += 1;
        if (statusCalls < 3) return { ok: false, error: "cap timeout" };
        return { ok: true, data: { kind: "exited", code: 0 } };
      }
      if (name === "job.result") return { ok: true, data: { raw: "{}" } };
      throw new Error("unexpected " + name);
    });
    const r = await runTrackedRemote({ ...baseOpts, invoke });
    expect(statusCalls).toBe(3);
    expect(r.exitCode).toBe(0);
  });

  it("reports a non-zero exit without a result", async () => {
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
      if (name === "job.status") return { ok: true, data: { kind: "exited", code: 2 } };
      if (name === "job.result") return { ok: true, data: { raw: null } };
      throw new Error("unexpected " + name);
    });
    const r = await runTrackedRemote({ ...baseOpts, invoke });
    expect(r.exitCode).toBe(2);
    expect(r.rawOutput).toBeNull();
  });

  it("throws when the job cannot be started", async () => {
    const invoke = vi.fn(async () => ({ ok: false, error: "sudo: a password is required" }));
    await expect(runTrackedRemote({ ...baseOpts, invoke })).rejects.toThrow(/password is required/);
  });

  it("persists the log offset as it advances, so a restart can reattach", async () => {
    const offsets: number[] = [];
    let done = false;
    const invoke = vi.fn(async (_n: string, name: string) => {
      if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
      if (name === "job.logs") return { ok: true, data: { chunk: "abc", nextOffset: 3, truncated: false } };
      if (name === "job.status") { const r = done ? { kind: "exited", code: 0 } : { kind: "active" }; done = true; return { ok: true, data: r }; }
      if (name === "job.result") return { ok: true, data: { raw: "{}" } };
      throw new Error("unexpected " + name);
    });
    await runTrackedRemote({ ...baseOpts, invoke, onOffset: (o) => offsets.push(o) });
    expect(offsets.at(-1)).toBe(3);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks/remote-runner.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Write minimal implementation**

Create `packages/server/src/benchmarks/remote-runner.ts`:

```ts
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

  const started = await o.invoke(o.nodeId, "job.start", {
    runId: o.runId, argv: o.argv, resultGlob: o.resultGlob ?? "result.json",
  });
  if (!started.ok) throw new Error(`job.start failed: ${started.error}`);

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
    // A null status means the capability call itself failed → unknown → keep going.
    const action = nextPollAction(status ?? { kind: "unknown", reason: "cap call failed" }, false);
    if (action === "fail") {
      throw new Error(`job ${o.runId} vanished on ${o.nodeId} (unit gone, no exit file)`);
    }
    if (action === "finish") {
      await drain(); // final tail
      const code = (status as { kind: "exited"; code: number }).code;
      const result = await call<{ raw: string | null }>(o.invoke, o.nodeId, "job.result", { runId: o.runId });
      return { exitCode: code, rawOutput: code === 0 ? (result?.raw ?? null) : null };
    }
    await sleep(pollMs);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks/remote-runner.test.ts`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/benchmarks/remote-runner.ts packages/server/src/benchmarks/remote-runner.test.ts
git commit -m "feat(benchmarks): runTrackedRemote — poll a systemd job; unknown never kills a run"
```

---

### Task 13: Wire the transport into the orchestrator

**Files:**
- Modify: `packages/server/src/benchmarks/orchestrator.ts`

**Interfaces:**
- Consumes: `runTrackedRemote` (Task 12)
- Produces: `RunBenchmarkOpts`, `RunToolEvalOpts`, `RunAccuracyOpts` each gain `runnerNodeId?: string`, `invoke?: CapInvoker`; `cancelBenchmark(runId)` gains a remote path

- [ ] **Step 1: Add the transport switch**

In `orchestrator.ts`, replace the body of `spawnTracked`'s call sites with a `dispatch` that chooses transport:

```ts
/**
 * Local execution stays available for laptop dev, behind an explicit env flag.
 * It is never an implicit fallback: a benchmark whose runner you cannot identify
 * is worse than no benchmark (see BenchmarkRun.runnerNodeId).
 */
const RUNNER = process.env.BENCH_RUNNER ?? "remote";

async function dispatch(opts: {
  runId: string; command: string; args: string[]; outputDir: string;
  onLog: (l: string) => void; resultFile?: (dir: string) => string | null;
  resultGlob?: string; runnerNodeId?: string; invoke?: CapInvoker; startOffset?: number;
  onOffset?: (o: number) => void;
}): Promise<SpawnTrackedResult> {
  if (RUNNER === "local") return spawnTracked(opts);
  if (!opts.runnerNodeId || !opts.invoke) {
    throw new Error("remote runner requires runnerNodeId + invoke (set BENCH_RUNNER=local for dev)");
  }
  // Register BEFORE awaiting: a cancel arriving mid-run must find this entry.
  REMOTE_ACTIVE.set(opts.runId, { nodeId: opts.runnerNodeId, invoke: opts.invoke });
  try {
    return await runTrackedRemote({
      runId: opts.runId, nodeId: opts.runnerNodeId, invoke: opts.invoke,
      argv: [opts.command, ...opts.args], resultGlob: opts.resultGlob,
      onLog: opts.onLog, onOffset: opts.onOffset, startOffset: opts.startOffset,
    });
  } finally {
    REMOTE_ACTIVE.delete(opts.runId);
  }
}
```

Track remote runs so cancel works:

```ts
const REMOTE_ACTIVE = new Map<string, { nodeId: string; invoke: CapInvoker }>();
```

The entry is registered **before** the await (a cancel arriving during the run must find it) and removed in a `finally`.

Extend `cancelBenchmark`:

```ts
export function cancelBenchmark(runId: string): boolean {
  const remote = REMOTE_ACTIVE.get(runId);
  if (remote) {
    // Idempotent on the agent side; stopping an already-finished unit is success.
    void remote.invoke(remote.nodeId, "job.cancel", { runId });
    return true;
  }
  const child = ACTIVE.get(runId);
  // … existing local path unchanged …
}
```

Pass `resultGlob: "results_*.json"` from `runAccuracy`, and `"result.json"` from `runBenchmark` / `runToolEval`.

- [ ] **Step 2: Update the orchestrator test for the accuracy resultGlob**

In `packages/server/src/benchmarks/orchestrator.test.ts`, add:

```ts
it("asks the remote wrapper to resolve lm-eval's nested result file", async () => {
  // runAccuracy must pass results_*.json, not result.json — lm-eval writes a
  // nested results_<timestamp>.json that the manager cannot stat remotely.
  const invoke = vi.fn(async (_n: string, name: string) => {
    if (name === "job.start") return { ok: true, data: { unit: "u", jobDir: "/j" } };
    if (name === "job.logs") return { ok: true, data: { chunk: "", nextOffset: 0, truncated: false } };
    if (name === "job.status") return { ok: true, data: { kind: "exited", code: 0 } };
    if (name === "job.result") return { ok: true, data: { raw: "{}" } };
    throw new Error("unexpected " + name);
  });
  await runAccuracy({
    runId: "r1", runnerNodeId: "n1", invoke,
    config: { tasks: ["ifeval"], primaryTask: "ifeval", primaryMetric: "prompt_level_strict_acc" } as never,
    endpointV1Url: "http://x/v1", servedModel: "m", outputDir: "/tmp/o", onLog: () => {},
  });
  const startCall = invoke.mock.calls.find((c) => c[1] === "job.start")!;
  expect((startCall[2] as { resultGlob: string }).resultGlob).toBe("results_*.json");
});
```

- [ ] **Step 3: Run tests + typecheck**

```bash
DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks
npm run build --workspace=packages/server
```
Expected: PASS; `tsc` exits 0.

- [ ] **Step 4: Commit**

```bash
git add packages/server/src/benchmarks/orchestrator.ts packages/server/src/benchmarks/orchestrator.test.ts
git commit -m "feat(benchmarks): dispatch runs to the eval node; BENCH_RUNNER=local for dev"
```

---

### Task 14: Boot reconciliation + cancel wiring

**Files:**
- Modify: `packages/server/src/index.ts` (the `main()` reconciliation block, currently ~line 103)
- Create: `packages/server/src/benchmarks/reconcile.ts`
- Create: `packages/server/src/benchmarks/reconcile.test.ts`

**Interfaces:**
- Consumes: `JobStatus` (Task 12)
- Produces: `reconcileAction(run: {runnerNodeId: string|null}, status: JobStatus|null): "resume" | "finalize" | "fail-orphan" | "fail-legacy"`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/benchmarks/reconcile.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { test, fc } from "@fast-check/vitest";
import { reconcileAction } from "./reconcile.js";

const remote = { runnerNodeId: "n1" };
const legacy = { runnerNodeId: null };

describe("reconcileAction", () => {
  // Legacy runs were children of the old server container. They died with it.
  it("fails a legacy local run, preserving today's message", () => {
    expect(reconcileAction(legacy, null)).toBe("fail-legacy");
  });

  it("resumes a remote job that is still running", () => {
    expect(reconcileAction(remote, { kind: "active" })).toBe("resume");
  });

  it("finalizes a remote job that finished while we were down", () => {
    expect(reconcileAction(remote, { kind: "exited", code: 0 })).toBe("finalize");
  });

  it("fails a remote job whose unit is genuinely gone", () => {
    expect(reconcileAction(remote, { kind: "missing" })).toBe("fail-orphan");
  });

  /**
   * Invariant: if we could not reach the agent at boot, we must NOT declare the
   * run dead. Resume and let the poll loop find out — the job is a systemd unit
   * and does not care that the manager was restarted.
   */
  test.prop([fc.string()])("an unknown status resumes rather than fails", (reason) => {
    expect(reconcileAction(remote, { kind: "unknown", reason })).toBe("resume");
  });

  it("resumes when the agent is offline at boot (null status)", () => {
    expect(reconcileAction(remote, null)).toBe("resume");
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks/reconcile.test.ts`
Expected: FAIL — cannot resolve module

- [ ] **Step 3: Write minimal implementation**

Create `packages/server/src/benchmarks/reconcile.ts`:

```ts
import type { JobStatus } from "./remote-runner.js";

/**
 * What to do at boot with a BenchmarkRun still marked pending/running.
 *
 * A run WITHOUT a runnerNodeId executed as a child of the old server container
 * and died with it — that is the pre-existing contract and we keep it.
 *
 * A run WITH one is a systemd unit on the eval node. It very probably survived.
 * If the agent is unreachable right now (`null`) or systemd could not answer
 * (`unknown`), resume and let the poll loop discover the truth. Declaring it dead
 * because we could not ask is the mistake this whole design exists to avoid.
 */
export function reconcileAction(
  run: { runnerNodeId: string | null },
  status: JobStatus | null,
): "resume" | "finalize" | "fail-orphan" | "fail-legacy" {
  if (!run.runnerNodeId) return "fail-legacy";
  if (status === null) return "resume";
  switch (status.kind) {
    case "active": return "resume";
    case "unknown": return "resume";
    case "exited": return "finalize";
    case "missing": return "fail-orphan";
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `DATABASE_URL="file:./prisma/dev.db" npx vitest run packages/server/src/benchmarks/reconcile.test.ts`
Expected: PASS

- [ ] **Step 5: Replace the boot block**

In `packages/server/src/index.ts`, replace the blanket `updateMany` (which currently fails **every** pending/running row) with a per-row decision. Keep the legacy message byte-identical:

```ts
  // Legacy (local) runs died with the old container. Remote runs are systemd
  // units on the eval node and usually survived — ask before killing them.
  const stale = await prisma.benchmarkRun.findMany({
    where: { status: { in: ["pending", "running"] } },
    select: { id: true, runnerNodeId: true, logOffset: true },
  });
  for (const run of stale) {
    if (!run.runnerNodeId) {
      await prisma.benchmarkRun.update({
        where: { id: run.id },
        data: { status: "failed", error: "server restarted before run completed", completedAt: new Date() },
      });
      continue;
    }
    // Agent may not be connected yet at boot; `resume` is the safe default.
    console.log(`[benchmarks] run ${run.id} has a remote job on ${run.runnerNodeId}; will reattach`);
  }
```

> Reattaching the poll loop requires the agentHub's capClient, which is constructed after this block. Move the reconciliation call to **after** `agentHub.start()`, and for each remote row call `job.status`, apply `reconcileAction`, and either resume `runTrackedRemote` (with `startOffset: run.logOffset`) or finalize/fail.

- [ ] **Step 6: Wire cancel**

`POST /api/benchmarks/:id/cancel` already calls `cancelBenchmark(run.id)`. No route change is needed — Task 13 gave `cancelBenchmark` its remote branch.

- [ ] **Step 7: Full suite alone, then commit**

```bash
DATABASE_URL="file:./prisma/dev.db" npm test
npm run build --workspace=packages/server
git add packages/server/src/index.ts packages/server/src/benchmarks/reconcile.ts packages/server/src/benchmarks/reconcile.test.ts
git commit -m "feat(benchmarks): reattach remote runs across a manager restart"
```

---

# Phase 5 — Cutover

### Task 15: Flip the kinds, mark provenance, re-baseline

**Files:**
- Modify: `packages/dashboard/app/benchmarks/compare/page.tsx`

- [ ] **Step 1: Warn when comparing across runners**

Throughput measures decode tok/s and TTFR **from the client**. Runs from different hosts are not comparable. In the compare view, when the selected runs have more than one distinct `runnerNodeId` (treating `null` as "manager"), render a warning above the bars:

```tsx
{distinctRunners.size > 1 && kind === "throughput" && (
  <p className="text-xs text-amber-300 bg-amber-950/30 border border-amber-900/50 rounded px-3 py-2 mb-3">
    These runs were measured from different hosts. Throughput and TTFR depend on the
    client's network path, so the numbers are not directly comparable.
  </p>
)}
```

- [ ] **Step 2: Deploy — agents were rolled in Task 8, so rebuild the server now**

```bash
setsid nohup docker compose build server > /tmp/build.log 2>&1 &
# on completion:
MANAGER_ADVERTISE_HOST=192.168.44.14 SSH_USER=daniel docker compose up -d --no-deps server
docker compose logs server --since 2m | grep -iE "in sync|benchmarks"
```

- [ ] **Step 3: Smoke-test the shortest real benchmark**

```bash
DEP=$(curl -s http://localhost:4000/api/deployments | python3 -c "import sys,json;print([d['id'] for d in json.load(sys.stdin) if d['status']=='running'][0])")
RUN=$(curl -s -X POST http://localhost:4000/api/benchmarks -H 'Content-Type: application/json' \
  -d "{\"deploymentId\":\"$DEP\",\"presetId\":\"quick-smoke\"}" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
echo "run $RUN"
ssh 192.168.44.15 "systemctl show dgxbench-$RUN -p ActiveState -p ExecMainStatus"
```
Expected: the unit exists on agenthost and is `active`; logs stream to the dashboard.

- [ ] **Step 4: Prove a run outlives a manager restart** (the whole point)

While that run is active:

```bash
docker compose restart server
sleep 20
curl -s "http://localhost:4000/api/benchmarks/$RUN" | python3 -c "import sys,json;r=json.load(sys.stdin);print(r['status'], r.get('runnerNodeId'))"
```
Expected: still `running`, with `runnerNodeId` set. Logs resume from the persisted offset. **This is the acceptance test for the entire plan.**

- [ ] **Step 5: Verify the 409 and 503 guards on the live system**

```bash
# 409 — second run against a busy deployment
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://localhost:4000/api/benchmarks \
  -H 'Content-Type: application/json' -d "{\"deploymentId\":\"$DEP\",\"presetId\":\"quick-smoke\"}"
```
Expected: `409`.

- [ ] **Step 6: Re-baseline throughput and record the break**

Run `quick-smoke` and `throughput` once each against the GLM-5.2 deployment from agenthost. Record the new tok/s and TTFR in `docs/ROADMAP.md` under Phase 5, noting that runs before this date were measured from the Pi and are not comparable.

- [ ] **Step 7: Commit**

```bash
git add packages/dashboard/app/benchmarks/compare/page.tsx docs/ROADMAP.md
git commit -m "feat(benchmarks): warn on cross-runner throughput comparison; re-baseline"
```

---

## Acceptance criteria

1. A benchmark started on agenthost **survives `docker compose restart server`** and reattaches with its log intact.
2. A benchmark survives an **agent roll** on the eval node.
3. `POST /api/deployments` targeting agenthost with a vLLM/dgxrun recipe returns **400**; with `ollama`, **201**.
4. `POST /api/benchmarks` returns **503** when no eval node is online, and **409** when one is already running against that deployment.
5. An inconclusive `job.status` never fails a run — covered by property tests in `remote-runner.test.ts`, `systemctl-parse.test.ts`, and `reconcile.test.ts`.
6. `npm test` is green, run alone.
7. agenthost's hand-installed ollama is still `active` and still serving embeddings on CPU.
