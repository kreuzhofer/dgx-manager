# Copy Claude Code launch command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-row "Claude" button on running deployments that opens a modal with a ready-to-paste shell snippet (bash/zsh + PowerShell) pointing the `claude` CLI at that deployment's vLLM endpoint.

**Architecture:** A pure server helper renders the env-var snippet for each shell. A new `GET /api/deployments/:id/claude-launch` route resolves the deployment's live served model name (reusing existing benchmark helpers) and returns both snippets. A dashboard modal fetches and displays them with a shell toggle and copy-to-clipboard.

**Tech Stack:** Express 5 + Prisma (server), Next.js 15 / React 19 / Tailwind (dashboard), Vitest + fast-check + supertest (tests).

## Global Constraints

- **`ANTHROPIC_BASE_URL` MUST NOT include a `/v1` suffix** — it is the vLLM server root (`http://<ip>:<port>`). The Anthropic Messages API lives at the root; the OpenAI surface (with `/v1`) is a different path.
- **Endpoint host/port** come from `deployment.node.ipAddress` and `deployment.port` — never hardcode `8000`.
- **Auth token value is the constant `dgx-local`** (throwaway; vLLM ignores it; its presence suppresses Claude Code's OAuth prompt).
- All three model-tier vars (`ANTHROPIC_DEFAULT_OPUS_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`) map to the **one** served model name.
- Reuse existing helpers `deploymentEndpointUrl` and `resolveServedModelName` from `packages/server/src/benchmarks/endpoint.ts`. Do NOT duplicate URL-building or model-resolution logic.
- **No agent code touched** → **do not** run the agent version bump.
- No Prisma schema change. No `/lb` proxy work.
- `npm test` must be green before any task is claimed done.

---

### Task 1: Pure snippet builder (`buildClaudeLaunchSnippet`)

**Files:**
- Create: `packages/server/src/deployments/claude-launch.ts`
- Test: `packages/server/src/deployments/claude-launch.test.ts`

**Interfaces:**
- Consumes: nothing (pure).
- Produces:
  - `type LaunchShell = "bash" | "powershell"`
  - `const CLAUDE_AUTH_TOKEN = "dgx-local"`
  - `function buildClaudeLaunchSnippet(input: { baseUrl: string; model: string; authToken: string; shell: LaunchShell }): string`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/deployments/claude-launch.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import { buildClaudeLaunchSnippet } from "./claude-launch.js";

const VAR_NAMES = [
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
] as const;

// Reverse the POSIX single-quote escaping the builder applies, so the property
// verifies a genuine round-trip rather than re-deriving the encoder.
function bashUnquote(rhs: string): string {
  // rhs looks like: '...'  where inner ' were replaced by the 4 chars '\''
  return rhs.slice(1, -1).split(`'\\''`).join("'");
}
function pwshUnquote(rhs: string): string {
  // rhs looks like: '...'  where inner ' were doubled to ''
  return rhs.slice(1, -1).split("''").join("'");
}
// Pull the quoted RHS for a given var out of a rendered snippet.
function rhsOf(snippet: string, varName: string, shell: "bash" | "powershell"): string {
  const prefix = shell === "bash" ? `export ${varName}=` : `$env:${varName} = `;
  const line = snippet.split("\n").find((l) => l.startsWith(prefix));
  if (!line) throw new Error(`no line for ${varName}`);
  return line.slice(prefix.length);
}

describe("buildClaudeLaunchSnippet", () => {
  it("renders the bash export block with a trailing run hint", () => {
    const out = buildClaudeLaunchSnippet({
      baseUrl: "http://10.0.0.5:8000",
      model: "glm-5.2",
      authToken: "dgx-local",
      shell: "bash",
    });
    expect(out).toContain("export ANTHROPIC_BASE_URL='http://10.0.0.5:8000'");
    expect(out).toContain("export ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.2'");
    expect(out.trimEnd().endsWith("# then run: claude")).toBe(true);
    // Base URL must not carry a /v1 suffix.
    expect(out).not.toContain("/v1");
  });

  it("renders the PowerShell block with $env: assignments", () => {
    const out = buildClaudeLaunchSnippet({
      baseUrl: "http://10.0.0.5:8000",
      model: "glm-5.2",
      authToken: "dgx-local",
      shell: "powershell",
    });
    expect(out).toContain("$env:ANTHROPIC_BASE_URL = 'http://10.0.0.5:8000'");
    expect(out).toContain("$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.2'");
  });

  /**
   * Invariant: for any baseUrl / model / token, every ANTHROPIC_* var is present
   * and each value survives the round trip through the shell quoting unchanged.
   */
  fcIt.prop([fc.string(), fc.string(), fc.string()])(
    "round-trips all values through bash and powershell quoting",
    (baseUrl, model, authToken) => {
      for (const shell of ["bash", "powershell"] as const) {
        const out = buildClaudeLaunchSnippet({ baseUrl, model, authToken, shell });
        const unquote = shell === "bash" ? bashUnquote : pwshUnquote;
        for (const name of VAR_NAMES) {
          const recovered = unquote(rhsOf(out, name, shell));
          const expected =
            name === "ANTHROPIC_BASE_URL" ? baseUrl :
            name === "ANTHROPIC_AUTH_TOKEN" ? authToken : model;
          expect(recovered).toBe(expected);
        }
      }
    },
  );
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/deployments/claude-launch.test.ts`
Expected: FAIL — `Failed to resolve import "./claude-launch.js"` / `buildClaudeLaunchSnippet is not a function`.

- [ ] **Step 3: Write minimal implementation**

Create `packages/server/src/deployments/claude-launch.ts`:

```ts
export type LaunchShell = "bash" | "powershell";

/** Throwaway bearer token — vLLM ignores its value; its presence stops
 *  Claude Code from starting the interactive OAuth login flow. */
export const CLAUDE_AUTH_TOKEN = "dgx-local";

/** Ordered so the rendered snippet is deterministic. All three model-tier
 *  vars map to the one served model, so whichever tier Claude Code picks
 *  hits this deployment. */
function envPairs(baseUrl: string, model: string, authToken: string): [string, string][] {
  return [
    ["ANTHROPIC_BASE_URL", baseUrl],
    ["ANTHROPIC_AUTH_TOKEN", authToken],
    ["ANTHROPIC_DEFAULT_OPUS_MODEL", model],
    ["ANTHROPIC_DEFAULT_SONNET_MODEL", model],
    ["ANTHROPIC_DEFAULT_HAIKU_MODEL", model],
  ];
}

/** POSIX single-quote: wrap in '...', closing/escaping/reopening for any '. */
function bashQuote(v: string): string {
  return `'${v.split("'").join(`'\\''`)}'`;
}

/** PowerShell single-quote literal: double any embedded single quote. */
function pwshQuote(v: string): string {
  return `'${v.split("'").join("''")}'`;
}

/** Render the export block that sets up a shell to drive Claude Code against a
 *  vLLM deployment. Pure + deterministic. NOTE: baseUrl must be the server root
 *  with NO /v1 suffix — the Anthropic Messages API lives at the root. */
export function buildClaudeLaunchSnippet(input: {
  baseUrl: string;
  model: string;
  authToken: string;
  shell: LaunchShell;
}): string {
  const { baseUrl, model, authToken, shell } = input;
  const pairs = envPairs(baseUrl, model, authToken);
  const lines =
    shell === "bash"
      ? pairs.map(([k, v]) => `export ${k}=${bashQuote(v)}`)
      : pairs.map(([k, v]) => `$env:${k} = ${pwshQuote(v)}`);
  lines.push("# then run: claude");
  return lines.join("\n");
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `npx vitest run packages/server/src/deployments/claude-launch.test.ts`
Expected: PASS (3 tests: two named + one property).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/deployments/claude-launch.ts packages/server/src/deployments/claude-launch.test.ts
git commit -m "feat(deployments): pure Claude Code launch-snippet builder"
```

---

### Task 2: Server route `GET /api/deployments/:id/claude-launch`

**Files:**
- Modify: `packages/server/src/routes/deployments.ts` (add imports near lines 1-21; add the route handler right after the existing `GET /:id/logs` handler)
- Test: `packages/server/src/__tests__/integration/deployments.claude-launch.test.ts`

**Interfaces:**
- Consumes: `buildClaudeLaunchSnippet`, `CLAUDE_AUTH_TOKEN` (Task 1); `deploymentEndpointUrl`, `resolveServedModelName` (existing `benchmarks/endpoint.ts`).
- Produces: `GET /api/deployments/:id/claude-launch` → `200 { baseUrl: string; model: string; shells: { bash: string; powershell: string } }`; `404` unknown id; `409` not running / no port / no node IP. Reads an optional injected `req.app.get("fetchImpl")` for the served-name lookup (falls back to global `fetch`).

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/__tests__/integration/deployments.claude-launch.test.ts`:

```ts
/**
 * Integration test for GET /api/deployments/:id/claude-launch.
 * Mirrors deployments.vram-admission.test.ts: per-suite SQLite set before
 * prisma import, only deploymentsRouter mounted, supertest, no port bind.
 * A stub fetchImpl is injected via app.set("fetchImpl", …) so served-name
 * resolution is deterministic and offline.
 */
import { afterAll, beforeAll, afterEach, describe, expect, it } from "vitest";
import { execSync } from "child_process";
import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import express from "express";
import request from "supertest";

const TMP_DIR = mkdtempSync(join(tmpdir(), "dgx-test-"));
const DB_PATH = join(TMP_DIR, "test.db");
process.env.DATABASE_URL = `file:${DB_PATH}`;

let prisma: typeof import("../../prisma.js").prisma;
let deploymentsRouter: typeof import("../../routes/deployments.js").deploymentsRouter;

beforeAll(async () => {
  execSync("npx prisma db push --force-reset", {
    cwd: process.cwd().replace(/\/packages\/server.*$/, ""),
    env: {
      ...process.env,
      DATABASE_URL: `file:${DB_PATH}`,
      PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION:
        "User consented to db push --force-reset against per-suite SQLite test databases in /tmp on 2026-05-03 (option #1)",
    },
    stdio: "pipe",
  });
  ({ prisma } = await import("../../prisma.js"));
  ({ deploymentsRouter } = await import("../../routes/deployments.js"));
});

afterAll(async () => {
  await prisma?.$disconnect();
  rmSync(TMP_DIR, { recursive: true, force: true });
});

afterEach(async () => {
  await prisma.deployment.deleteMany();
  await prisma.model.deleteMany();
  await prisma.node.deleteMany();
});

// Stub that resolves /v1/models to a fixed served id.
function makeApp(servedId = "served-xyz") {
  const app = express();
  app.use(express.json());
  app.set("fetchImpl", async () => ({
    ok: true,
    text: async () => JSON.stringify({ data: [{ id: servedId }] }),
  }));
  app.use("/api/deployments", deploymentsRouter);
  return app;
}

async function seedDeployment(overrides: { status?: string; port?: number | null; displayName?: string | null } = {}) {
  const node = await prisma.node.create({ data: { name: `n-${Math.random().toString(36).slice(2)}`, ipAddress: "10.0.0.5" } });
  const model = await prisma.model.create({ data: { name: `m-${Math.random().toString(36).slice(2)}`, runtime: "vllm" } });
  return prisma.deployment.create({
    data: {
      nodeId: node.id,
      modelId: model.id,
      status: overrides.status ?? "running",
      port: overrides.port === undefined ? 8000 : overrides.port,
      displayName: overrides.displayName ?? null,
    },
  });
}

describe("GET /api/deployments/:id/claude-launch", () => {
  it("returns both shell snippets with the live served model name", async () => {
    const d = await seedDeployment();
    const res = await request(makeApp("live-served-name")).get(`/api/deployments/${d.id}/claude-launch`);
    expect(res.status).toBe(200);
    expect(res.body.baseUrl).toBe("http://10.0.0.5:8000");
    expect(res.body.model).toBe("live-served-name");
    expect(res.body.shells.bash).toContain("export ANTHROPIC_BASE_URL='http://10.0.0.5:8000'");
    expect(res.body.shells.bash).toContain("export ANTHROPIC_DEFAULT_OPUS_MODEL='live-served-name'");
    expect(res.body.shells.bash).not.toContain("/v1");
    expect(res.body.shells.powershell).toContain("$env:ANTHROPIC_BASE_URL = 'http://10.0.0.5:8000'");
  });

  it("404s for an unknown deployment id", async () => {
    const res = await request(makeApp()).get("/api/deployments/does-not-exist/claude-launch");
    expect(res.status).toBe(404);
  });

  it("409s when the deployment is not running", async () => {
    const d = await seedDeployment({ status: "stopped" });
    const res = await request(makeApp()).get(`/api/deployments/${d.id}/claude-launch`);
    expect(res.status).toBe(409);
  });

  it("409s when the deployment has no port", async () => {
    const d = await seedDeployment({ port: null });
    const res = await request(makeApp()).get(`/api/deployments/${d.id}/claude-launch`);
    expect(res.status).toBe(409);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.claude-launch.test.ts`
Expected: FAIL — the first test 404s (route not defined yet), so `expect(res.status).toBe(200)` fails.

- [ ] **Step 3: Add imports to `deployments.ts`**

Add these to the import block at the top of `packages/server/src/routes/deployments.ts` (after the existing imports, before `export const deploymentsRouter`):

```ts
import { deploymentEndpointUrl, resolveServedModelName } from "../benchmarks/endpoint.js";
import { buildClaudeLaunchSnippet, CLAUDE_AUTH_TOKEN } from "../deployments/claude-launch.js";
```

- [ ] **Step 4: Add the route handler**

Insert this handler in `packages/server/src/routes/deployments.ts` immediately after the existing `deploymentsRouter.get("/:id/logs", …)` handler (any location on `deploymentsRouter` before the file's other `/:id` routes is fine; placing it by `/logs` keeps the read-only GETs together):

```ts
/**
 * @openapi
 * /api/deployments/{id}/claude-launch:
 *   get:
 *     tags: [Deployments]
 *     summary: Shell snippet to launch Claude Code against this deployment
 *     description: >
 *       Returns bash/zsh and PowerShell export blocks that point the `claude`
 *       CLI at this deployment's vLLM endpoint via the Anthropic Messages API
 *       (ANTHROPIC_BASE_URL = http://<node-ip>:<port>, no /v1 suffix). The
 *       served model name is resolved from the live `/v1/models` endpoint,
 *       falling back to the deployment's display/catalog name if unreachable.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200': { description: baseUrl, resolved model, and per-shell snippets }
 *       '404': { description: deployment not found }
 *       '409': { description: deployment is not serving an endpoint }
 */
deploymentsRouter.get("/:id/claude-launch", async (req, res) => {
  const deployment = await prisma.deployment.findUnique({
    where: { id: req.params.id },
    include: { node: true, model: true },
  });
  if (!deployment) {
    return res.status(404).json({ error: "deployment not found" });
  }
  if (deployment.status !== "running" || !deployment.port || !deployment.node?.ipAddress) {
    return res.status(409).json({
      error: "deployment is not serving an endpoint (needs status=running, a port, and a node IP)",
    });
  }
  const baseUrl = deploymentEndpointUrl(deployment);
  const fetchImpl = (req.app.get("fetchImpl") as typeof fetch) ?? fetch;
  const model = await resolveServedModelName(
    `${baseUrl}/v1`,
    deployment.displayName ?? deployment.model.name,
    fetchImpl,
  );
  res.json({
    baseUrl,
    model,
    shells: {
      bash: buildClaudeLaunchSnippet({ baseUrl, model, authToken: CLAUDE_AUTH_TOKEN, shell: "bash" }),
      powershell: buildClaudeLaunchSnippet({ baseUrl, model, authToken: CLAUDE_AUTH_TOKEN, shell: "powershell" }),
    },
  });
});
```

- [ ] **Step 5: Run test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/deployments.claude-launch.test.ts`
Expected: PASS (4 tests).

- [ ] **Step 6: Run the full server suite to confirm no regressions**

Run: `npm test`
Expected: PASS (all suites green).

- [ ] **Step 7: Commit**

```bash
git add packages/server/src/routes/deployments.ts packages/server/src/__tests__/integration/deployments.claude-launch.test.ts
git commit -m "feat(deployments): GET /:id/claude-launch endpoint returning shell snippets"
```

---

### Task 3: Dashboard API helper + Claude launch modal

**Files:**
- Create: `packages/dashboard/lib/claude-launch.ts`
- Create: `packages/dashboard/components/claude-launch-modal.tsx`

**Interfaces:**
- Consumes: the Task 2 endpoint shape `{ baseUrl, model, shells: { bash, powershell } }`; existing `apiFetch` from `@/lib/api`.
- Produces:
  - `type ClaudeLaunch = { baseUrl: string; model: string; shells: { bash: string; powershell: string } }`
  - `function fetchClaudeLaunch(deploymentId: string): Promise<ClaudeLaunch>`
  - React component `ClaudeLaunchModal({ deploymentId, deploymentLabel, onClose }: { deploymentId: string; deploymentLabel: string; onClose: () => void })`

> This task is UI. The repo has no React component-test harness (its Vitest suites cover pure helpers and HTTP routes), so per CLAUDE.md's "say so explicitly" rule these components are verified by a typecheck build (Step 3) and the manual click-through in Task 4. The `fetchClaudeLaunch` helper is a one-line `apiFetch` wrapper with no logic to unit-test.

- [ ] **Step 1: Create the API helper**

Create `packages/dashboard/lib/claude-launch.ts`:

```ts
import { apiFetch } from "./api";

export type ClaudeLaunch = {
  baseUrl: string;
  model: string;
  shells: { bash: string; powershell: string };
};

export function fetchClaudeLaunch(deploymentId: string): Promise<ClaudeLaunch> {
  return apiFetch<ClaudeLaunch>(`/api/deployments/${deploymentId}/claude-launch`);
}
```

- [ ] **Step 2: Create the modal component**

Create `packages/dashboard/components/claude-launch-modal.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { fetchClaudeLaunch, type ClaudeLaunch } from "@/lib/claude-launch";

type Shell = "bash" | "powershell";

type Props = {
  deploymentId: string;
  deploymentLabel: string;
  onClose: () => void;
};

export function ClaudeLaunchModal({ deploymentId, deploymentLabel, onClose }: Props) {
  const [data, setData] = useState<ClaudeLaunch | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [shell, setShell] = useState<Shell>("bash");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchClaudeLaunch(deploymentId)
      .then((d) => { if (!cancelled) setData(d); })
      .catch((e) => { if (!cancelled) setError((e as Error).message); });
    return () => { cancelled = true; };
  }, [deploymentId]);

  const snippet = data ? data.shells[shell] : "";

  async function handleCopy() {
    try {
      if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(snippet);
      } else {
        // Fallback for non-secure contexts (dashboard is often http://<ip>:3000)
        const ta = document.createElement("textarea");
        ta.value = snippet;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.top = "0";
        ta.style.left = "0";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.focus();
        ta.select();
        const ok = document.execCommand("copy");
        document.body.removeChild(ta);
        if (!ok) throw new Error("execCommand('copy') returned false");
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error("copy failed", err);
      alert("Copy failed — please select and copy the snippet manually.");
    }
  }

  return (
    <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50" onClick={onClose}>
      <div
        className="bg-gray-900 rounded-lg p-6 w-[640px] max-h-[90vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-xl font-semibold mb-1">Launch Claude Code</h2>
        <p className="text-sm text-gray-400 mb-4">Target: {deploymentLabel}</p>

        {error && (
          <div className="text-sm text-red-400 bg-red-950/40 border border-red-900 rounded p-3 mb-4">
            {error}
          </div>
        )}

        {!data && !error && <p className="text-sm text-gray-400">Resolving served model…</p>}

        {data && (
          <>
            <div className="flex gap-2 mb-3">
              {(["bash", "powershell"] as Shell[]).map((s) => (
                <button
                  key={s}
                  onClick={() => setShell(s)}
                  className={`text-xs px-3 py-1.5 rounded transition-colors ${
                    shell === s ? "bg-green-600 text-white" : "bg-gray-800 hover:bg-gray-700 text-gray-300"
                  }`}
                >
                  {s === "bash" ? "bash / zsh" : "PowerShell"}
                </button>
              ))}
            </div>

            <pre className="bg-gray-950 rounded p-3 font-mono text-xs whitespace-pre-wrap break-all border border-gray-800 mb-3">
              {snippet}
            </pre>

            <div className="flex gap-2 items-center mb-4">
              <button
                onClick={handleCopy}
                className={`px-3 py-1.5 rounded text-xs font-medium transition-colors inline-flex items-center gap-1.5 ${
                  copied ? "bg-emerald-500 text-white" : "bg-green-600 hover:bg-green-500 text-white"
                }`}
              >
                {copied ? (<><span aria-hidden>&#10003;</span> Copied</>) : "Copy snippet"}
              </button>
              <button
                onClick={onClose}
                className="px-3 py-1.5 bg-gray-700 hover:bg-gray-600 rounded text-xs transition-colors"
              >
                Close
              </button>
            </div>

            <p className="text-xs text-gray-500">
              Paste into a new shell, then run <code className="text-gray-400">claude</code>. Requires this
              model served with tool-calling enabled on a vLLM build that exposes{" "}
              <code className="text-gray-400">/v1/messages</code>.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Typecheck the dashboard build**

Run: `npm run build --workspace packages/dashboard`
Expected: build succeeds (no TypeScript errors). The new component is not imported yet, so this only verifies it compiles in isolation.

- [ ] **Step 4: Commit**

```bash
git add packages/dashboard/lib/claude-launch.ts packages/dashboard/components/claude-launch-modal.tsx
git commit -m "feat(dashboard): Claude Code launch modal + API helper"
```

---

### Task 4: Wire the "Claude" button into the deployments page

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (import ~line 9; new state ~line 201; button ~line 1435; modal render ~line 1697)

**Interfaces:**
- Consumes: `ClaudeLaunchModal` (Task 3). Uses existing row variable `d` (has `id`, `status`, `port`, `displayName`, `model?.name`, `modelId`, `node?.name`).
- Produces: nothing consumed by later tasks (terminal task).

- [ ] **Step 1: Add the import**

In `packages/dashboard/app/deployments/page.tsx`, next to the existing component imports (near line 9, `import { BenchmarkFormModal } from "@/components/benchmark-form-modal";`), add:

```tsx
import { ClaudeLaunchModal } from "@/components/claude-launch-modal";
```

- [ ] **Step 2: Add the modal-target state**

Immediately after the `benchmarkTarget` state declaration (around line 199-201):

```tsx
  const [benchmarkTarget, setBenchmarkTarget] = useState<
    { id: string; label: string } | null
  >(null);
```

add:

```tsx
  const [claudeLaunchTarget, setClaudeLaunchTarget] = useState<
    { id: string; label: string } | null
  >(null);
```

- [ ] **Step 3: Add the "Claude" button**

In the per-row button cluster, immediately after the closing `)}` of the **API** link block (around line 1435, right before the `{d.status === "running" && d.port && (` block that renders the **Benchmark** button), insert:

```tsx
                        {d.status === "running" && d.port && (
                          <button
                            type="button"
                            className="text-xs px-2 py-1 rounded bg-gray-800 hover:bg-gray-700 text-orange-300 transition-colors"
                            onClick={() => setClaudeLaunchTarget({
                              id: d.id,
                              label: `${d.displayName ?? d.model?.name ?? d.modelId} @ ${d.node?.name ?? "?"}`,
                            })}
                          >
                            Claude
                          </button>
                        )}
```

- [ ] **Step 4: Render the modal**

Immediately after the `benchmarkTarget` modal block (around lines 1690-1697):

```tsx
      {benchmarkTarget && (
        <BenchmarkFormModal
          deploymentId={benchmarkTarget.id}
          deploymentLabel={benchmarkTarget.label}
          onClose={() => setBenchmarkTarget(null)}
          onStarted={() => {/* SSE will populate latestBenchmarkStatus */}}
        />
      )}
```

add:

```tsx
      {claudeLaunchTarget && (
        <ClaudeLaunchModal
          deploymentId={claudeLaunchTarget.id}
          deploymentLabel={claudeLaunchTarget.label}
          onClose={() => setClaudeLaunchTarget(null)}
        />
      )}
```

- [ ] **Step 5: Typecheck the dashboard build**

Run: `npm run build --workspace packages/dashboard`
Expected: build succeeds with no TypeScript errors.

- [ ] **Step 6: Manual end-to-end verification**

Bring up the stack (per CLAUDE.md) and exercise the button against a running deployment:

```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build
```

Then, with at least one deployment in `running` state:
1. Open the dashboard `/deployments` page.
2. Confirm a **Claude** button appears next to the **API** link on the running row (and is absent on non-running / worker rows).
3. Click it → modal opens, shows "Resolving served model…", then the bash block.
4. Confirm `ANTHROPIC_BASE_URL='http://<node-ip>:<port>'` has **no `/v1`**, and the model name matches what the row's **API** link (`/v1/models`) reports.
5. Toggle **PowerShell** → `$env:` syntax renders.
6. Click **Copy snippet** → button flips to "Copied ✓".
7. (Optional real proof) Paste the bash block into a fresh terminal, run `claude`, confirm it connects to the deployment (works only if the model was served with tool-calling enabled — the caveat in the modal).

Record the result. If step 4's model name is wrong or the endpoint errors, stop and debug before claiming done.

- [ ] **Step 7: Commit**

```bash
git add packages/dashboard/app/deployments/page.tsx
git commit -m "feat(dashboard): add Claude launch button to deployment rows"
```

---

## Self-Review Notes

- **Spec coverage:** snippet format + no-`/v1` (Task 1 + constraints), export-block-only with run hint (Task 1), bash + PowerShell (Task 1/3), server endpoint with 404/409 + reused helpers (Task 2), server-side served-name resolution avoiding CORS (Task 2), modal + shell toggle + copy-with-fallback + caveat (Task 3), button next to API link under the same guard (Task 4), property + integration tests (Task 1/2), no schema change / no agent bump (constraints). All spec sections map to a task.
- **Types:** `LaunchShell`/`Shell` values `"bash" | "powershell"`, `ClaudeLaunch.shells` keys `bash`/`powershell`, and the endpoint JSON shape are consistent across Tasks 1→2→3→4.
- **No placeholders:** every code and command step is complete.
