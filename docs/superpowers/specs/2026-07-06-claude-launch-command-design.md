# Copy Claude Code launch command for deployments — design

**Date:** 2026-07-06
**Status:** Approved, ready for implementation plan

## Problem

To point the `claude` CLI at a self-hosted model, a user must hand-assemble
environment variables (endpoint URL from the node IP + port, a throwaway auth
token, and the served model name for each Claude model tier), open a fresh
shell, export them, and launch. This is fiddly and error-prone. We want a
one-click **Copy** affordance next to each running deployment that yields a
ready-to-paste shell snippet setting up a new terminal to drive Claude Code
against that specific deployment.

## Feasibility (settled by research)

Modern **vLLM natively implements the Anthropic Messages API** (`/v1/messages`),
the same protocol Claude Code uses to talk to Anthropic. Claude Code therefore
talks **directly** to the vLLM instance — **no proxy** (LiteLLM,
claude-code-router, or the repo's unmounted `/lb` proxy) is required.

Source: <https://docs.vllm.ai/en/stable/serving/integrations/claude_code/>

Two conditions must hold on the *deployment* for the launched session to be
usable. They are recipe/version concerns, **out of scope** for this feature, and
surfaced to the user as a caveat rather than fixed here:

1. The vLLM build is recent enough to expose `/v1/messages` (older builds only
   had `/v1/chat/completions`).
2. The model was launched with tool-calling enabled
   (`--enable-auto-tool-choice --tool-call-parser <x>`). Claude Code is unusable
   without tools. For GLM-5.2 (the workhorse, already used for agentic coding)
   both hold because the sparkrun recipe enables tool choice.

## The snippet

The user chose an **export block only** (sets up the shell; the user runs
`claude` themselves) targeting **bash/zsh and PowerShell** (toggle in the UI).

bash/zsh:
```bash
export ANTHROPIC_BASE_URL='http://192.168.44.36:8000'
export ANTHROPIC_AUTH_TOKEN='dgx-local'
export ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.2'
export ANTHROPIC_DEFAULT_SONNET_MODEL='glm-5.2'
export ANTHROPIC_DEFAULT_HAIKU_MODEL='glm-5.2'
# then run: claude
```

PowerShell:
```powershell
$env:ANTHROPIC_BASE_URL = 'http://192.168.44.36:8000'
$env:ANTHROPIC_AUTH_TOKEN = 'dgx-local'
$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'glm-5.2'
$env:ANTHROPIC_DEFAULT_SONNET_MODEL = 'glm-5.2'
$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.2'
# then run: claude
```

Rules baked into the format:

- **`ANTHROPIC_BASE_URL` has no `/v1` suffix** — the Anthropic Messages API is at
  the server root. (This is the one gotcha vs. the existing "API" link, which
  appends `/v1` for the OpenAI-compatible surface.)
- All three tier vars (`OPUS`/`SONNET`/`HAIKU`) map to the **one** served model,
  so whichever tier Claude Code selects hits this deployment.
- `ANTHROPIC_AUTH_TOKEN` is a required throwaway (`dgx-local`); vLLM ignores its
  value and its presence suppresses Claude Code's OAuth login prompt.
- Values are **single-quoted** (bash) / single-quoted with embedded `'` doubled
  (PowerShell) so arbitrary model ids / URLs are shell-safe.

## Architecture

### Server endpoint

`GET /api/deployments/:id/claude-launch`, added to `deploymentsRouter`
(`packages/server/src/routes/deployments.ts`; mounted at `/api/deployments`).

Behavior:

1. Load the deployment with `node` + `model`.
   - **404** if the deployment id is unknown.
   - **409** if it is not `running`, or has no `port`. (Fail fast — the reused
     `deploymentEndpointUrl` throws on a missing node/ip/port; the route maps
     that to a 409 with a clear message rather than a 500.)
2. `baseUrl = deploymentEndpointUrl(d)` → `http://<node.ipAddress>:<port>`
   (reused from `packages/server/src/benchmarks/endpoint.ts`).
3. `model = await resolveServedModelName(baseUrl + "/v1", d.displayName ?? d.model.name)`
   (reused). This queries the live `{url}/v1/models` and returns vLLM's actual
   served id. **Explicit, documented fallback:** if the node is briefly
   unreachable it returns `displayName ?? model.name` (best known name) rather
   than erroring — consistent with "no *silent* fallback": the behavior is
   documented and the returned name is still shown to the user, who can verify
   against the API link.
4. Respond `200` with:
   ```json
   {
     "baseUrl": "http://192.168.44.36:8000",
     "model": "glm-5.2",
     "shells": { "bash": "export ...", "powershell": "$env:..." }
   }
   ```

Resolving the served name **on the manager** (not in the browser) sidesteps
CORS and mixed-content entirely: the dashboard keeps talking only to the
manager, exactly like every other API call.

### Pure formatter

New module `packages/server/src/deployments/claude-launch.ts` exports a pure,
deterministic:

```ts
type Shell = "bash" | "powershell";
function buildClaudeLaunchSnippet(input: {
  baseUrl: string;
  model: string;
  authToken: string;   // "dgx-local"
  shell: Shell;
}): string;
```

All shell syntax and value escaping live here; the route does IO and calls this
once per shell. This mirrors the repo's mandated split of pure logic out of
route files (`computeVramShortfall` vs `checkVllmVramAdmission`).

### Dashboard

- A **`Claude`** button on each deployment row in
  `packages/dashboard/app/deployments/page.tsx`, placed next to the existing
  **API** link (~line 1426) under the same guard `d.status === "running" && d.port`.
- Clicking opens a **modal** (mirroring the existing Benchmark modal pattern in
  that page):
  - fetches `GET /api/deployments/:id/claude-launch` (spinner while resolving),
  - shows endpoint + resolved model id as context,
  - **shell toggle** `bash/zsh` ⇄ `PowerShell` (just selects a field from the
    response — no refetch),
  - renders the snippet in a monospace block with a **Copy** button reusing the
    `packages/dashboard/components/onboarding-command.tsx` clipboard logic
    **including its non-secure-context `document.execCommand("copy")` fallback**
    (required — the dashboard is commonly served over plain `http://<ip>:3000`,
    where `navigator.clipboard` is unavailable),
  - shows a one-line caveat: *"Requires the model served with tool-calling
    enabled on a vLLM build that exposes /v1/messages."*
  - surfaces a fetch error inline (e.g. node unreachable / not running) rather
    than failing silently.

## Data (all already available; no schema change)

- `deployment.node.ipAddress` — SSH-reachable management IP (already in the
  `/api/deployments` payload via `include: { node: true }`). **Not**
  `fastIpAddress` (fabric IP).
- `deployment.port` — authoritative stored column, reported by the agent after
  launch (default 8000, auto-bumped on conflict). Read the column; never
  hardcode 8000.
- `deployment.displayName ?? deployment.model.name` — fallback served name when
  the live `/v1/models` lookup can't run.

No Prisma schema change. No new columns. No `/lb` proxy work.

## Testing

Per repo conventions (Vitest + fast-check + supertest):

- **Property test** on `buildClaudeLaunchSnippet` (`claude-launch.test.ts`,
  next to source): for arbitrary `baseUrl` + `model` + `authToken` and each
  shell, the rendered snippet (a) contains all five `ANTHROPIC_*` vars, (b)
  quotes/escapes each value so it round-trips back to the exact input, and (c)
  never contains `/v1` in the base URL line. Invariant documented in a
  doc-comment above the property.
- **Integration test** (supertest against an Express app mounting only
  `deploymentsRouter`, with a per-test SQLite): happy path — running deployment
  with a port, an injected `fetchImpl` stub returning a `/v1/models` body →
  `200`, `shells.bash` contains the stubbed served id and the correct
  `host:port`. Error paths — unknown id → `404`; status not `running` → `409`;
  `port` null → `409`.

`npm test` must be green before the change is claimed done.

## Scope / non-goals

- **Claude Code only.** The route is named `claude-launch`; other CLIs (Codex,
  etc.) are a future sibling route, **not built now** (YAGNI). OpenCode is
  explicitly excluded (it configures multiple models natively).
- **No agent changes**, so **no agent version bump** required.
- We do **not** modify sparkrun recipes or enable tool-calling — that is a
  deployment concern; the caveat line surfaces the requirement.
- No changes to the `/lb` inference proxy (it remains unmounted).

## Risk tier

Medium — a new read-only endpoint + a new config-string generator, no data
mutation, no auth/money/data-loss surface. Covered by one property test (pure
helper) + happy/error-path integration tests per the medium-risk expectation.
