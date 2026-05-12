# Fine-Tune Launch Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring the deploy-a-fine-tuned-model UX on par with the deploy-a-recipe UX: display the user's custom `displayName` first (base model as a smaller note), and expose the same launch knobs (TP, PP, gpu-mem, max-model-len, port + cluster picker + VRAM admission) the normal deploy form already has.

**Architecture:** The vLLM launch infrastructure (`launchRecipe` in `packages/agent/src/runtime/vllm.ts`) already accepts cluster + TP/PP/gpuMem/maxModelLen options — both `cmd:deploy` and `cmd:finetune:deploy` end up calling it. The current `cmd:finetune:deploy` simply doesn't pass any of those options through, and `generateLocalModelRecipe` hardcodes `solo_only: true`. We extend the data flow (dashboard → server → agent) so the same knobs that are wired for normal recipes also reach `launchRecipe` for fine-tune deploys, and adjust the generated YAML so its metadata reflects the requested topology. The cross-repo split stays as-is (training recipes in `dgx-manager-fine-tune-recipes`, vLLM deploy infra in `spark-vllm-docker`) — the per-deploy generated YAML continues to live in the deploy repo's `recipes/` dir.

**Tech Stack:** Express 5 + Prisma (SQLite), Node WS agent + bash launch scripts (`run-recipe.sh`, `launch-cluster.sh`), Next.js 15 App Router (React 19), Tailwind CSS 4.

---

## Discussion: how fine-tune deploys currently work

**Recipe handling (cross-repo).** Fine-tune training recipes live in `dgx-manager-fine-tune-recipes` and optionally expose a `deploy:` block that names the vLLM container image and gives default `gpu_memory_utilization` + `max_model_len`. Deploy infrastructure (`run-recipe.sh`, `launch-cluster.sh`, the `recipes/` dir) lives in `spark-vllm-docker` (`VLLM_REPO_PATH`). The bridge between the two is **dynamic recipe generation**: on every fine-tune deploy, the agent writes a new YAML to `${VLLM_REPO_PATH}/recipes/finetune-<jobId-prefix>.yaml` via `generateLocalModelRecipe`, then launches it with the same `launchRecipe` function normal deploys use.

**This dynamic-per-deploy pattern is the right design and we keep it.** Reusing a static recipe from the deploy repo doesn't work because the model path is per-job (`/mnt/tank/outputs/<jobId>/merged`). The training-recipe `deploy:` block stays the source of defaults; user overrides at launch time get merged on top; the result lands in the generated YAML and `launchRecipe` flags.

**Why we don't need a third repo or a manager-side recipe registry.** Each side already owns the right thing: the training-recipe repo owns "what the model is and what defaults make sense", the vLLM repo owns "how to launch it". The manager mediates by generating per-deploy YAMLs that capture the merged config. The only thing currently missing is that the manager + agent code path doesn't actually plumb through all the knobs the launch infra already supports.

**Current limitations (what this plan fixes):**
1. Dashboard's finetune-mode form shows only a Node picker — no TP / PP / port / gpu-mem / max-model-len overrides.
2. Server's `POST /api/finetune/:id/deploy` accepts only `{ nodeId, config }` — no cluster `nodeIds[]`, no VRAM admission check.
3. Agent's `cmd:finetune:deploy` payload omits `clusterNodes`, `tensorParallel`, `pipelineParallel`. The generated YAML hardcodes `solo_only: true`, and `launchRecipe` is called without cluster info.
4. The finetune-mode UI block shows the base model name as the primary label and the user's custom `displayName` is invisible (it doesn't even flow through the redirect URL).

---

## File Structure

**Modified:**
- `packages/dashboard/app/finetune/page.tsx` — `deployJob` propagates `displayName` via URL params.
- `packages/dashboard/app/deployments/page.tsx` — finetune-mode form gets full knob set + cluster picker; primary label is now displayName (base model becomes secondary line).
- `packages/server/src/routes/finetune.ts` — `POST /:id/deploy` accepts `nodeIds[]`, runs VRAM admission, persists cluster nodes, forwards full config to the agent.
- `packages/agent/src/index.ts` — `cmd:finetune:deploy` handler accepts cluster + TP/PP params, passes them to `launchRecipe`.
- `packages/agent/src/runtime/vllm.ts` — `generateLocalModelRecipe` accepts cluster flag (drops `solo_only` for cluster), accepts TP/PP for the comment block (the actual flags are CLI-injected by `launchRecipe`).
- `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts` — add cluster-deploy + VRAM-admission integration tests (same file, since these exercise the same deploy route).

**Not modified:**
- `packages/server/src/admission/vram.ts` — `checkVllmVramAdmission` is already used by normal deploys and is reused as-is.
- `packages/agent/src/runtime/vllm.ts:launchRecipe` — already accepts `tensorParallel`, `pipelineParallel`, `clusterNodes`, `clusterNodeFastIps`. No change needed.
- `dgx-manager-fine-tune-recipes` repo — training recipes already have an optional `deploy:` block; no schema change required.
- `spark-vllm-docker` repo — launch infra (`run-recipe.sh`, `launch-cluster.sh`) already supports `--tp`, `-pp`, `-n` flags.

---

## Task 1: Dashboard — `deployJob` propagates `displayName` to URL

**Files:**
- Modify: `packages/dashboard/app/finetune/page.tsx` (the `deployJob` function, currently around line 405)

- [ ] **Step 1: Locate the existing `deployJob`**

Run from `/home/daniel/src/github/dgx-manager`:
```
grep -nB1 -A 10 "const deployJob = " packages/dashboard/app/finetune/page.tsx
```

You should see a function that builds a `URLSearchParams` with `finetuneModel`, `finetuneJobId`, `baseModel` and redirects via `window.location.href`.

- [ ] **Step 2: Add `displayName` to the URL params**

Replace the body of `deployJob` with:

```typescript
  const deployJob = (job: FineTuneJob) => {
    const modelPath = job.mergedPath || `${job.outputDir}/merged`;
    const params = new URLSearchParams({
      finetuneModel: modelPath,
      finetuneJobId: job.id,
      baseModel: job.baseModel,
    });
    if (job.displayName) params.set("displayName", job.displayName);
    window.location.href = `/deployments?${params.toString()}`;
  };
```

We send `displayName` only when it's set so the receiving page can fall back cleanly (the URL stays shorter for unnamed jobs).

- [ ] **Step 3: TypeScript-check**

Run:
```
cd /home/daniel/src/github/dgx-manager/packages/dashboard
npx tsc --noEmit
```

Expected: no new errors.

- [ ] **Step 4: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/dashboard/app/finetune/page.tsx
git commit -m "dashboard: propagate finetune displayName to deploy redirect"
```

---

## Task 2: Dashboard — deployments page reads displayName + reorders the finetune-mode label

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx` (URL-param reader around line 173, and the finetune-mode JSX block around line 540)

- [ ] **Step 1: Add `displayName` state + URL read**

Find the existing `useState` for `finetuneModel`, `finetuneJobId`, `finetuneBaseModel`. Add a new state for displayName right after them:

```typescript
  const [finetuneDisplayName, setFinetuneDisplayName] = useState<string | null>(null);
```

Then in the URL-param reader block (search for `params.get("finetuneModel")`), after the existing reads add:

```typescript
    const fn = params.get("displayName");
    if (fn) setFinetuneDisplayName(fn);
```

- [ ] **Step 2: Reorder the finetune-mode label**

Find the existing block:

```tsx
            {runtimeMode === "finetune" && finetuneModel ? (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Fine-tuned Model</label>
                <div className="bg-gray-800 border border-purple-700 rounded px-3 py-2 text-sm text-purple-300">
                  {finetuneBaseModel || "Fine-tuned model"}
                  <span className="ml-2 text-[10px] text-gray-500">{finetuneModel}</span>
                </div>
              </div>
            ) : ...
```

Replace with:

```tsx
            {runtimeMode === "finetune" && finetuneModel ? (
              <div>
                <label className="block text-xs text-gray-400 mb-1">Fine-tuned Model</label>
                <div className="bg-gray-800 border border-purple-700 rounded px-3 py-2 text-sm text-purple-300">
                  <div className="font-medium">
                    {finetuneDisplayName || finetuneBaseModel || "Fine-tuned model"}
                  </div>
                  {finetuneDisplayName && finetuneBaseModel && (
                    <div className="text-[10px] text-gray-500 mt-0.5">
                      base: {finetuneBaseModel}
                    </div>
                  )}
                  <div className="text-[10px] text-gray-500 mt-0.5 truncate">{finetuneModel}</div>
                </div>
              </div>
            ) : ...
```

Behavior: primary line is `displayName` if set (e.g. the user's `build123d-v1`); falls back to base model name when no displayName. Secondary lines: base model (only when displayName is shown, to disambiguate) and the model path. Long paths get `truncate`.

- [ ] **Step 3: Reset `finetuneDisplayName` after submit**

In the submit handler's finetune branch, near the existing `setFinetuneModel(null); setFinetuneJobId(null);`, add:

```typescript
        setFinetuneDisplayName(null);
```

- [ ] **Step 4: TypeScript-check + commit**

```bash
cd /home/daniel/src/github/dgx-manager/packages/dashboard
npx tsc --noEmit
cd /home/daniel/src/github/dgx-manager
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: deploy form shows finetune displayName first, base model as note"
```

---

## Task 3: Server — `POST /api/finetune/:id/deploy` accepts `nodeIds[]` + persists cluster

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (POST `/:id/deploy`, currently around line 521)

The new route signature mirrors `POST /api/deployments`: accept either `nodeId` (single) or `nodeIds[]` (cluster); when multi-node, create `ClusterNode` rows.

- [ ] **Step 1: Read the current handler**

Run:
```
sed -n '520,580p' /home/daniel/src/github/dgx-manager/packages/server/src/routes/finetune.ts
```

This shows the current single-node-only handler. Hold it in mind for the diff in the next step.

- [ ] **Step 2: Replace the request-body parse + node resolution**

Find the line:

```typescript
  const { nodeId, config } = req.body;
  const targetNodeId = nodeId || job.nodeId;
```

Replace with:

```typescript
  const { nodeId, nodeIds, config } = req.body;
  const isCluster = Array.isArray(nodeIds) && nodeIds.length > 1;
  const headNodeId = isCluster ? nodeIds[0] : (nodeId || job.nodeId);
  if (!headNodeId) {
    return res.status(400).json({ error: "nodeId or nodeIds required" });
  }
```

- [ ] **Step 3: Add ClusterNode persistence after the deployment is created**

Find the existing block:

```typescript
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: targetNodeId,
      modelId: model.id,
      status: "pending",
      config: JSON.stringify({ ...config, localModelPath: modelPath }),
    },
  });
```

Replace it with:

```typescript
  const deployment = await prisma.deployment.create({
    data: {
      nodeId: headNodeId,
      modelId: model.id,
      status: "pending",
      clusterMode: isCluster,
      config: JSON.stringify({ ...config, localModelPath: modelPath }),
    },
  });

  // For multi-node deploys, persist cluster membership the same way normal
  // deploys do — one ClusterNode row per participant, head first. The agent
  // gets the node IP list separately (see clusterNodeIps/Fast below).
  let clusterNodeIps: string[] | undefined;
  let clusterNodeFastIps: (string | null)[] | undefined;
  if (isCluster) {
    const nodes = await prisma.node.findMany({ where: { id: { in: nodeIds } } });
    const nodeMap = new Map(nodes.map((n) => [n.id, n]));
    for (let i = 0; i < nodeIds.length; i++) {
      await prisma.clusterNode.create({
        data: {
          deploymentId: deployment.id,
          nodeId: nodeIds[i],
          role: i === 0 ? "head" : "worker",
          status: "pending",
        },
      });
    }
    clusterNodeIps = nodeIds.map((id: string) => nodeMap.get(id)?.ipAddress).filter(Boolean);
    clusterNodeFastIps = nodeIds.map((id: string) => nodeMap.get(id)?.fastIpAddress ?? null);
  }
```

- [ ] **Step 4: Forward `targetNodeId` rename + payload changes** (next task — Task 5 covers the agent payload). For now leave the rest of the handler untouched. We come back to the `sendToAgent` call in Task 5.

- [ ] **Step 5: Replace `targetNodeId` references with `headNodeId`**

In the rest of the same handler, find any remaining `targetNodeId` reference (e.g., `agentHub.sendToAgent(targetNodeId, ...)`) and rename it to `headNodeId`. There is one such reference in the existing code, at the `sendToAgent` call.

- [ ] **Step 6: Commit**

This commit lands schema/route plumbing only — full agent payload + admission come in Tasks 4 + 5:

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/server/src/routes/finetune.ts
git commit -m "finetune: deploy route accepts nodeIds[] + persists ClusterNode rows"
```

(No tests in this commit — the wiring is exercised by Task 10's integration tests once the full chain is in.)

---

## Task 4: Server — VRAM admission for finetune deploys

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (the same `POST /:id/deploy` handler)

The check is identical in semantics to the one in `POST /api/deployments`: build a list of nodes to check (head + workers, or just head for solo), call `checkVllmVramAdmission(nodeIds, gpuMemUtil)`, and 409 if there are shortfalls.

- [ ] **Step 1: Confirm the helper is already imported, or add the import**

Run:
```
grep -n "checkVllmVramAdmission\|vramShortfallMessage" /home/daniel/src/github/dgx-manager/packages/server/src/routes/finetune.ts
```

If no matches, add this import at the top of the file (right after the other `import ... from "../..."` lines):

```typescript
import { checkVllmVramAdmission, vramShortfallMessage } from "../admission/vram.js";
```

The helper lives at `packages/server/src/admission/vram.ts` (signature: `checkVllmVramAdmission(nodeIds: string[], gpuMemUtil: number)` returns `Promise<VramShortfall[]>`).

- [ ] **Step 2: Insert the admission check**

In the deploy handler, AFTER the cluster-node persistence block from Task 3 and BEFORE the `agentHub.sendToAgent(...)` call, insert:

```typescript
  // Pre-flight VRAM admission, same as normal vLLM deploys. The gpuMem
  // ceiling either comes from the user's launch override or the training
  // recipe's deploy.gpu_memory_utilization default (set when the recipe
  // was authored). 0.85 fallback matches the normal-deploy default.
  const gpuMemForAdmission =
    (config?.gpuMem as number) ||
    (deployConfig?.gpu_memory_utilization as number) ||
    0.85;
  const admissionNodeIds = isCluster ? (nodeIds as string[]) : [headNodeId];
  const shortfalls = await checkVllmVramAdmission(admissionNodeIds, gpuMemForAdmission);
  if (shortfalls.length > 0) {
    return res.status(409).json({
      error: `Not enough VRAM on ${shortfalls.length} of ${admissionNodeIds.length} node(s): ${vramShortfallMessage(shortfalls)}`,
      shortfalls,
      gpuMemoryUtilization: gpuMemForAdmission,
    });
  }
```

Note: the existing handler already reads `deployConfig` from the training recipe earlier in the flow — your insertion must come AFTER that read. If `deployConfig` is read AFTER the current `sendToAgent` call in the existing code, MOVE its read up to just before the admission insertion. (Verify in the actual file: search for `trainingRecipe.deploy`.)

- [ ] **Step 3: Run existing tests to confirm nothing broke**

```
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
```

Expected: 17/17 still pass. (The admission check will run during the existing tests — make sure the test stub provides VRAM ceiling data via `prisma.node.create({ ..., vramTotal: 122_502 })` if needed. Some existing tests already do this; if a new test fails because the admission check now blocks, it means the seed node didn't have `vramTotal` set. Add `vramTotal: 122_502` to `seedNode()` in the test file.)

- [ ] **Step 4: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/server/src/routes/finetune.ts packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: pre-flight VRAM admission on /:id/deploy"
```

---

## Task 5: Server — forward full config + cluster info to the agent

**Files:**
- Modify: `packages/server/src/routes/finetune.ts` (the `agentHub.sendToAgent` call in `POST /:id/deploy`)

- [ ] **Step 1: Find the existing `cmd:finetune:deploy` payload**

The current payload is:

```typescript
  agentHub.sendToAgent(headNodeId, {
    type: "cmd:finetune:deploy",
    payload: {
      jobId: job.id,
      deploymentId: deployment.id,
      modelPath,
      baseModel: job.baseModel,
      deployContainer: deployConfig?.container || "vllm-node",
      config: {
        gpuMem: deployConfig?.gpu_memory_utilization,
        maxModelLen: deployConfig?.max_model_len,
        ...config,
      },
    },
  });
```

- [ ] **Step 2: Extend the payload with cluster + TP/PP**

Replace it with:

```typescript
  agentHub.sendToAgent(headNodeId, {
    type: "cmd:finetune:deploy",
    payload: {
      jobId: job.id,
      deploymentId: deployment.id,
      modelPath,
      baseModel: job.baseModel,
      deployContainer: deployConfig?.container || "vllm-node",
      // clusterNodeIps / clusterNodeFastIps are undefined for solo, set for
      // multi-node — see Task 3 for where they're computed.
      clusterNodes: clusterNodeIps,
      clusterNodeFastIps,
      config: {
        // Recipe defaults first, user overrides win via the spread.
        gpuMem: deployConfig?.gpu_memory_utilization,
        maxModelLen: deployConfig?.max_model_len,
        ...config,
      },
    },
  });
```

The `config` object now contains everything the agent needs: `port`, `gpuMem`, `maxModelLen`, `tensorParallel`, `pipelineParallel` (the dashboard's submit handler already builds this shape — Task 8 adds the inputs that fill those fields).

- [ ] **Step 3: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/server/src/routes/finetune.ts
git commit -m "finetune: deploy payload includes clusterNodes + full config"
```

---

## Task 6: Agent — `cmd:finetune:deploy` accepts cluster + TP/PP, passes to `launchRecipe`

**Files:**
- Modify: `packages/agent/src/index.ts` (the `cmd:finetune:deploy` case, currently around line 778)

- [ ] **Step 1: Locate the existing handler**

Run:
```
grep -nA 50 "case \"cmd:finetune:deploy\"" /home/daniel/src/github/dgx-manager/packages/agent/src/index.ts | head -60
```

Read the current handler. Key locals: `jobId`, `deploymentId`, `modelPath`, `deployContainer`, `config`. The current call to `generateLocalModelRecipe` passes `{ jobId, modelPath, container, port, gpuMemoryUtilization, maxModelLen }` and the current `launchRecipe` call passes `{ port, skipSetup: true }`.

- [ ] **Step 2: Destructure the new payload fields + thread them through**

Replace the entire `case "cmd:finetune:deploy"` block with:

```typescript
    case "cmd:finetune:deploy": {
      const {
        jobId, deploymentId, modelPath, deployContainer, config,
        clusterNodes, clusterNodeFastIps,
      } = msg.payload as {
        jobId: string;
        deploymentId: string;
        modelPath: string;
        deployContainer?: string;
        config?: Record<string, unknown>;
        clusterNodes?: string[];
        clusterNodeFastIps?: (string | null)[];
      };

      const isCluster = Array.isArray(clusterNodes) && clusterNodes.length > 1;
      const port = (config?.port as number) ?? 8000;
      const gpuMem = (config?.gpuMem as number) ?? 0.85;
      const maxModelLen = (config?.maxModelLen as number) ?? 4096;
      const tensorParallel = config?.tensorParallel as number | undefined;
      const pipelineParallel = config?.pipelineParallel as number | undefined;

      console.log(`[finetune] Deploying merged model from ${modelPath} (container: ${deployContainer || "vllm-node"}, cluster: ${isCluster ? clusterNodes!.length + " nodes" : "solo"})`);

      try {
        const recipeFile = generateLocalModelRecipe({
          jobId,
          modelPath,
          container: deployContainer || "vllm-node",
          port,
          gpuMemoryUtilization: gpuMem,
          maxModelLen,
          // Mark the generated YAML's solo_only flag based on the topology
          // we were given. Doesn't affect actual launch (CLI flags do that),
          // just keeps the recipe metadata honest for any future re-launch
          // via the recipe selector.
          isCluster,
        });

        sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
        let lastPhase = "starting";
        launchRecipe(
          deploymentId,
          recipeFile,
          {
            port,
            gpuMem,
            maxModelLen,
            tensorParallel,
            pipelineParallel,
            clusterNodes: isCluster ? clusterNodes : undefined,
            clusterNodeFastIps: isCluster ? clusterNodeFastIps : undefined,
            skipSetup: true,
          },
          (line) => {
            sendMsg("agent:deployment:log", { deploymentId, log: line });
            const phase = detectPhase(line);
            if (phase && phase !== lastPhase) {
              lastPhase = phase;
              sendMsg("agent:deployment:status", { deploymentId, status: phase });
            }
          },
          (code) => {
            if (code === 0 && isVllmContainerRunning()) {
              console.log(`[finetune] Deploy run-recipe.sh exited 0, container running`);
            } else if (!isVllmContainerRunning()) {
              sendMsg("agent:deployment:status", {
                deploymentId, status: "failed",
                error: code === 0 ? "Container not running after launch" : `Launch failed with exit code ${code}`,
              });
            }
          }
        );
      } catch (err) {
        sendMsg("agent:deployment:status", {
          deploymentId, status: "failed", error: String(err),
        });
      }
      break;
    }
```

- [ ] **Step 3: TypeScript check on the agent package**

Run:
```
cd /home/daniel/src/github/dgx-manager/packages/agent
npx tsc --noEmit
```

Expected: a type error referencing `isCluster` not existing on `generateLocalModelRecipe`'s param type — that gets resolved in Task 7. Note the error and proceed (this task does not compile in isolation; we batch-fix with Task 7).

- [ ] **Step 4: Commit (without running the agent)**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/agent/src/index.ts
git commit -m "agent: cmd:finetune:deploy threads cluster + TP/PP to launchRecipe"
```

---

## Task 7: Agent — `generateLocalModelRecipe` accepts `isCluster`

**Files:**
- Modify: `packages/agent/src/runtime/vllm.ts` (the `generateLocalModelRecipe` function, currently around line 49)

- [ ] **Step 1: Update the param type + body**

Find the function. Replace the existing signature + body with:

```typescript
export function generateLocalModelRecipe(params: {
  jobId: string;
  modelPath: string;
  container?: string;
  port?: number;
  gpuMemoryUtilization?: number;
  maxModelLen?: number;
  // When true, omit the `solo_only: true` marker so the dashboard's
  // recipe selector lists this as a cluster-capable recipe. The actual
  // launch topology is decided by the CLI flags launchRecipe builds,
  // not by this marker — this just keeps recipe metadata honest.
  isCluster?: boolean;
}): string {
  const recipeName = `finetune-${params.jobId.slice(0, 12)}`;
  const recipeFile = `recipes/${recipeName}.yaml`;
  const fullPath = join(VLLM_REPO_PATH, recipeFile);

  const containerModelPath = params.modelPath.replace(`${SHARED_STORAGE}/`, `${WORKSPACE}/`);

  const port = params.port ?? 8000;
  const gpuMem = params.gpuMemoryUtilization ?? 0.85;
  const maxLen = params.maxModelLen ?? 4096;
  const container = params.container || "vllm-node";
  const soloLine = params.isCluster ? "" : "solo_only: true\n";

  const yaml = `# Auto-generated recipe for fine-tuned model
recipe_version: "1"
name: ${recipeName}
description: Fine-tuned model from job ${params.jobId}
model: ${containerModelPath}
container: ${container}
${soloLine}
defaults:
  port: ${port}
  host: 0.0.0.0
  gpu_memory_utilization: ${gpuMem}
  max_model_len: ${maxLen}

command: |
  vllm serve ${containerModelPath} \\
    --host {host} \\
    --port {port} \\
    --max-model-len {max_model_len} \\
    --gpu-memory-utilization {gpu_memory_utilization} \\
    --dtype auto
`;

  mkdirSync(join(VLLM_REPO_PATH, "recipes"), { recursive: true });
  writeFileSync(fullPath, yaml, "utf-8");
  console.log(`Generated vLLM recipe: ${fullPath} (cluster=${!!params.isCluster})`);
  return recipeFile;
}
```

Note: `--tensor-parallel-size` / `--pipeline-parallel-size` are NOT added to the YAML's `command` block. `launchRecipe` passes them as CLI flags to `run-recipe.sh` (`--tp N`, then `-- -pp M`), which then forwards them to `vllm serve` at launch time. The YAML stays minimal.

- [ ] **Step 2: TypeScript check on agent package**

```
cd /home/daniel/src/github/dgx-manager/packages/agent
npx tsc --noEmit
```

Expected: clean. (Task 6's `isCluster` reference now resolves.)

- [ ] **Step 3: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/agent/src/runtime/vllm.ts
git commit -m "agent: generateLocalModelRecipe accepts isCluster (drops solo_only marker)"
```

- [ ] **Step 4: Bump the agent version**

`CLAUDE.md` mandates an agent version bump for any change under `packages/agent/src/`. Run:

```
cd /home/daniel/src/github/dgx-manager
./scripts/bump-agent-version.sh
```

This bumps the patch version in `packages/agent/package.json`. Stage + amend the previous commit:

```bash
git add packages/agent/package.json
git commit --amend --no-edit
```

(Note: this folds the version bump into the same commit since we're still amending pre-push. If the commit was already pushed, do a separate bump commit instead.)

---

## Task 8: Dashboard — finetune-mode form gets the full knob set

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx`

The form's existing state already has `port`, `maxModelLen`, `tensorParallel`, `pipelineParallel`, `gpuMem` — they're shared with the vLLM-mode path. We need to (a) render those inputs in finetune mode too, and (b) wire them into the finetune submit payload.

- [ ] **Step 1: Find the existing knob-input JSX (used by vLLM mode)**

Run:
```
grep -nE "setTensorParallel|setPipelineParallel|setGpuMem|setMaxModelLen" /home/daniel/src/github/dgx-manager/packages/dashboard/app/deployments/page.tsx | head
```

The inputs are rendered inside a div around line 660-720. Identify the wrapping conditional. If the current conditional is `runtimeMode === "vllm"`, the simplest fix is to change it to `runtimeMode === "vllm" || runtimeMode === "finetune"`.

If the inputs are deeply nested inside a vLLM-only block, hoist them into a helper or share the parent block. Pick whichever is least invasive given the file's current structure.

- [ ] **Step 2: Loosen the conditional**

Find the JSX guard around the TP/PP/port/gpuMem/maxModelLen input rows. Change any `runtimeMode === "vllm"` to:

```typescript
(runtimeMode === "vllm" || runtimeMode === "finetune")
```

- [ ] **Step 3: Send full config in the finetune submit branch**

Find the current finetune submit branch (around line 280):

```typescript
      if (runtimeMode === "finetune" && finetuneJobId) {
        if (!selectedNode) return;
        const config: Record<string, unknown> = { port: parseInt(port) || 8000 };
        if (gpuMem) config.gpuMem = parseFloat(gpuMem);
        if (maxModelLen) config.maxModelLen = parseInt(maxModelLen);
        const result = await apiFetch<Deployment>(`/api/finetune/${finetuneJobId}/deploy`, {
          method: "POST",
          body: JSON.stringify({ nodeId: selectedNode, config }),
        });
```

Replace it with:

```typescript
      if (runtimeMode === "finetune" && finetuneJobId) {
        const config: Record<string, unknown> = { port: parseInt(port) || 8000 };
        if (gpuMem) config.gpuMem = parseFloat(gpuMem);
        if (maxModelLen) config.maxModelLen = parseInt(maxModelLen);
        if (tensorParallel) config.tensorParallel = parseInt(tensorParallel);
        if (pipelineParallel) config.pipelineParallel = parseInt(pipelineParallel);

        // Cluster vs solo: when TP * PP > 1, send the explicit node list
        // selectedClusterNodes already holds (the cluster picker is reused).
        const tp = parseInt(tensorParallel) || 1;
        const pp = parseInt(pipelineParallel) || 1;
        const needsCluster = tp * pp > 1;
        if (needsCluster && selectedClusterNodes.size !== tp * pp) return;
        if (!needsCluster && !selectedNode) return;

        const body: Record<string, unknown> = { config };
        if (needsCluster) body.nodeIds = Array.from(selectedClusterNodes);
        else body.nodeId = selectedNode;

        const result = await apiFetch<Deployment>(`/api/finetune/${finetuneJobId}/deploy`, {
          method: "POST",
          body: JSON.stringify(body),
        });
```

- [ ] **Step 4: Update the disabled-state check on the submit button**

Find:

```tsx
disabled={deploying || (runtimeMode === "finetune" ? !selectedNode : runtimeMode === "vllm" ? (!selectedRecipe || !canDeploy) : (!selectedOllamaModel || !selectedNode))}
```

Replace the finetune branch so it accepts either single-node or full cluster picks:

```tsx
disabled={deploying || (runtimeMode === "finetune"
  ? (() => {
      const tp = parseInt(tensorParallel) || 1;
      const pp = parseInt(pipelineParallel) || 1;
      const needsCluster = tp * pp > 1;
      return needsCluster ? selectedClusterNodes.size !== tp * pp : !selectedNode;
    })()
  : runtimeMode === "vllm" ? (!selectedRecipe || !canDeploy) : (!selectedOllamaModel || !selectedNode))}
```

- [ ] **Step 5: TypeScript-check + commit**

```bash
cd /home/daniel/src/github/dgx-manager/packages/dashboard
npx tsc --noEmit
cd /home/daniel/src/github/dgx-manager
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: finetune-mode exposes TP/PP/gpuMem/maxModelLen + cluster picker"
```

---

## Task 9: Dashboard — cluster picker reuse for finetune mode

The existing cluster picker (used by vLLM mode when a recipe has `cluster_only: true` or the user sets TP*PP>1) renders a list of nodes with checkboxes. It's wired to `selectedClusterNodes` state. The picker's conditional render is based on `needsCluster` which is computed from `effectiveTP * effectivePP > 1`.

In finetune mode, the same logic should apply: when the user sets TP=4 (e.g.), the picker should appear and let them choose 4 nodes.

**Files:**
- Modify: `packages/dashboard/app/deployments/page.tsx`

- [ ] **Step 1: Find the existing cluster picker render**

Run:
```
grep -nE "needsCluster|selectedClusterNodes|effectiveTP" /home/daniel/src/github/dgx-manager/packages/dashboard/app/deployments/page.tsx | head -20
```

Find the JSX that renders the picker (search for `clusterCandidates.map`). Note the surrounding conditional.

- [ ] **Step 2: Confirm the picker's gating works in finetune mode too**

The `needsCluster` variable is computed from `effectiveTP * effectivePP > 1`. `effectiveTP` and `effectivePP` are computed from `parseInt(tensorParallel) || (selectedRecipeData?.defaults?.tensor_parallel as number) || 1`. In finetune mode there's no `selectedRecipeData`, so `effectiveTP` falls back to 1 unless the user fills the input.

That's the correct behavior: user must explicitly set TP/PP > 1 in finetune mode to trigger cluster mode. No code change needed if the picker's parent conditional doesn't gate on `runtimeMode === "vllm"`.

Verify by inspection: find the JSX that wraps `clusterCandidates.map(...)`. If it's gated only by `needsCluster`, this task is a no-op verification. If it's also gated on `runtimeMode === "vllm"`, change that guard the same way as Task 8 step 2: `(runtimeMode === "vllm" || runtimeMode === "finetune")`.

- [ ] **Step 3: If you changed anything, commit; otherwise no commit**

```bash
cd /home/daniel/src/github/dgx-manager
git status --short
# If anything is modified:
git add packages/dashboard/app/deployments/page.tsx
git commit -m "dashboard: cluster picker enabled in finetune mode"
```

---

## Task 10: Integration tests for the new finetune deploy contract

**Files:**
- Modify: `packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts`

We add tests for the new behaviors: `nodeIds[]` accepted, cluster nodes persisted, VRAM admission fires when over budget.

- [ ] **Step 1: Update `seedNode` to include `vramTotal`**

Find the existing `seedNode` helper. Add `vramTotal: 122_502` to the data block:

```typescript
async function seedNode() {
  return prisma.node.create({
    data: {
      id: "node-1", name: "dgx-spark-01", ipAddress: "192.168.44.36",
      status: "online", vramTotal: 122_502,
    },
  });
}
```

Add a multi-node seed helper near it:

```typescript
async function seedFourNodes() {
  return Promise.all(
    [1, 2, 3, 4].map((i) =>
      prisma.node.create({
        data: {
          id: `node-${i}`, name: `dgx-spark-0${i}`,
          ipAddress: `192.168.44.${35 + i}`, status: "online",
          vramTotal: 122_502,
        },
      }),
    ),
  );
}
```

- [ ] **Step 2: Append the cluster-deploy test**

```typescript
  it("POST /:id/deploy accepts nodeIds[] and creates ClusterNode rows", async () => {
    await wipeAll();
    await seedFourNodes();
    const { hub, sentMessages } = makeStubHub();
    const app = makeApp(hub);

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });

    const dep = await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({
        nodeIds: ["node-1", "node-2", "node-3", "node-4"],
        config: { tensorParallel: 4 },
      });
    expect(dep.status).toBe(201);
    expect(dep.body.clusterMode).toBe(true);

    const cluster = await prisma.clusterNode.findMany({
      where: { deploymentId: dep.body.id },
      orderBy: { role: "asc" },
    });
    expect(cluster).toHaveLength(4);
    expect(cluster.find((c) => c.role === "head")?.nodeId).toBe("node-1");
    expect(cluster.filter((c) => c.role === "worker").map((c) => c.nodeId).sort())
      .toEqual(["node-2", "node-3", "node-4"]);

    // Agent message went to head and included cluster info.
    const startMsg = sentMessages.find((m) =>
      (m.message as { type?: string }).type === "cmd:finetune:deploy"
    );
    expect(startMsg?.nodeId).toBe("node-1");
    const payload = (startMsg!.message as { payload: { clusterNodes?: string[]; config: Record<string, unknown> } }).payload;
    expect(payload.clusterNodes).toEqual([
      "192.168.44.36", "192.168.44.37", "192.168.44.38", "192.168.44.39",
    ]);
    expect(payload.config.tensorParallel).toBe(4);
  });
```

- [ ] **Step 3: Append the VRAM-admission test**

Add a model seeded with 100 GB of VRAM consumption on node-2, then try to deploy a cluster across all four:

```typescript
  it("POST /:id/deploy returns 409 when a cluster node lacks VRAM", async () => {
    await wipeAll();
    await seedFourNodes();
    const { hub } = makeStubHub();
    const app = makeApp(hub);

    // Saturate node-2 with an existing deployment (102 GB in use leaves
    // < 0.85 of 122 GB free, which the admission helper will reject).
    const existingModel = await prisma.model.create({
      data: { name: "preexisting-large", runtime: "vllm" },
    });
    await prisma.deployment.create({
      data: {
        nodeId: "node-2", modelId: existingModel.id, status: "running",
        vramActual: 102_400,
      },
    });

    const create = await request(app)
      .post("/api/finetune")
      .send({ nodeId: "node-1", recipeFile: RECIPE.file, dataset: "/tmp/fake.jsonl" });
    await prisma.fineTuneJob.update({
      where: { id: create.body.id },
      data: { mergeStatus: "completed", mergedPath: "/tmp/fake-merged" },
    });

    const dep = await request(app)
      .post(`/api/finetune/${create.body.id}/deploy`)
      .send({
        nodeIds: ["node-1", "node-2", "node-3", "node-4"],
        config: { tensorParallel: 4, gpuMem: 0.85 },
      });
    expect(dep.status).toBe(409);
    expect(dep.body.error).toMatch(/Not enough VRAM/i);
    expect(dep.body.shortfalls).toBeDefined();
    expect(dep.body.shortfalls.some((s: { nodeName: string }) => s.nodeName === "dgx-spark-02"))
      .toBe(true);
  });
```

- [ ] **Step 4: Run all tests**

```
cd /home/daniel/src/github/dgx-manager
npx vitest run packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
```

Expected: 19 passed (17 prior + 2 new).

- [ ] **Step 5: Commit**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/server/src/__tests__/integration/finetune.naming-and-cleanup.test.ts
git commit -m "finetune: integration tests for cluster deploy + VRAM admission"
```

---

## Task 11: Final pass — full tests + TypeScript + rebuild

**Files:** none.

- [ ] **Step 1: Run the full test suite**

```
cd /home/daniel/src/github/dgx-manager
npm test 2>&1 | tail -10
```

Expected: 19 passing in `finetune.naming-and-cleanup` + everything else green except the pre-existing `deployments.vram-admission` + `finetune.cluster-persistence` isolation flake (these pass in isolation; the flake is unrelated to this branch).

- [ ] **Step 2: TypeScript-check both packages**

```
cd /home/daniel/src/github/dgx-manager/packages/server && npx tsc --noEmit
cd /home/daniel/src/github/dgx-manager/packages/dashboard && npx tsc --noEmit
cd /home/daniel/src/github/dgx-manager/packages/agent && npx tsc --noEmit
```

Expected: no new errors anywhere.

- [ ] **Step 3: Build per-arch agent bundles** (required because agent code changed)

```
cd /home/daniel/src/github/dgx-manager
./scripts/build-agent-bundles.sh
```

Expected: builds amd64 + arm64 tarballs without errors. The output is consumed by the manager's agent-bundle route so agents can self-update.

- [ ] **Step 4: Rebuild server + dashboard**

```
MANAGER_ADVERTISE_HOST=192.168.44.36 SSH_USER=daniel docker compose up -d --build
```

Expected: server + dashboard images rebuild; containers recreate. Brief WS disconnect for agents (they reconnect automatically). No active training runs are killed (per the standing pattern).

- [ ] **Step 5: Trigger agent self-update**

In the dashboard or via API, hit `POST /api/nodes/update-all-agents` (or whatever the existing per-node update endpoint is). Verify agent versions advance.

Run:
```
curl -s http://192.168.44.36:4000/api/nodes | jq '.[] | {name, agentVersion}'
```

Expected: agent version matches the bumped patch from Task 7.

- [ ] **Step 6: Manual smoke test**

1. Open `http://192.168.44.36:3000/finetune` and confirm your renamed fine-tune shows its displayName in the row + still has a "Deploy" button.
2. Click Deploy on a fine-tune. Confirm the deploy form's primary line shows the displayName (not the base model).
3. Set TP=2. Confirm the cluster picker appears and lists nodes.
4. Pick two nodes, leave gpu-mem default, hit Deploy. Watch the deployment row appear with `clusterMode: true` and 2 cluster nodes.
5. Stop + delete the test deployment.
6. Try over-allocating VRAM (set TP=4 with all nodes selected + a separate large deployment already running). Confirm you get the 409 with shortfall details.

---

## Task 12: Push

**Files:** none.

- [ ] **Step 1: Verify branch state**

```
cd /home/daniel/src/github/dgx-manager
git status -sb
git log --oneline @{u}..HEAD
```

You should see ~10–12 commits on top of origin.

- [ ] **Step 2: Push**

```
git push origin claude/setup-dev-environment-gSVCI
```

- [ ] **Step 3: Done.**
