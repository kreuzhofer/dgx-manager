# Nemotron 3 Ultra NVFP4 TP=4 Deployment Recipe — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> ## ⚠️ EXECUTION UPDATE (2026-06-04) — read this first
>
> Execution surfaced a vLLM-version gate and an engine-isolation decision that change the
> task sequence below. See the spec's "Execution-time revisions" for full rationale. The
> **actually-executed** sequence is:
>
> 1. ✅ **dgx-manager git hygiene** — merged `feat/inference-variant-selector` → `main`, fixed `dev.db` gitignore, pushed.
> 2. ✅ **Resolve model** — `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (LatentMoE + MTP). NVIDIA recommends TP=4.
> 3. ✅ **vLLM gate** — Ultra needs ≥0.22; `vllm-node` ships 0.18.1. eugr's `prebuilt-vllm-current` = **0.22.1rc1** (2026-06-03).
> 4. ✅ **Sync fork** — merged `eugr/upstream/main` (28 commits) into `spark-vllm-docker` fork; pushed.
> 5. ✅ **Decide: v1 recipe in fork** (not sparkrun/v2 — not integrated into dgx-manager); **isolate engine** on `container: vllm-node-eugr022` so shared `vllm-node` (0.18.1) is untouched.
> 6. ✅ **Author recipe** — `recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml` (Super Spark-adapted flags + Ultra deltas; YAML validated).
> 7. ✅ **Build isolated image** — `vllm-node-eugr022` (0.22.1rc1) built on `dgx-spark-01` (0.18.1 wheel preserved in `wheels/.preserve-0.18.1/`); one transient PyTorch-CDN 503 → retried.
> 8. ✅ **Verify flag surface** — `vllm serve --help=all` + registry probes; fixed `fastsafetensors`→`safetensors`, confirmed cutlass/nemotron_v3/qwen3_coder/TRITON_ATTN/fp8/mtp.
> 9. ✅ **Deploy via API** — `POST /api/deployments nodeIds:"auto"`; image synced to 3 workers, 352 GB weights downloaded to NFS (cached for retries), TP=4 launched. **Five deploy-loop fixes** (JSON-in-command, drop expert-parallel, instanttensor→safetensors, **MTP disabled** (unquantized-MoE vs cutlass), fastsafetensors removed) — see spec "Outcome".
> 10. ✅ **Verify inference** — `Application startup complete`; coherent `/v1/chat/completions` reply on `192.168.44.36:8000` (served `nemotron-3-ultra`).
> 11. ⏳ **Commit recipe** to fork; **land spec + this plan** on dgx-manager `main` (in progress).
>
> **Follow-ups:** re-enable MTP; raise `max_model_len`/`gpu_memory_utilization`; revisit `instanttensor` (needs higher container `nofile`); reconcile `/mnt/tank` ↔ origin; agent flips `status:running` before vLLM serves (verify via HTTP, not status).
>
> Tasks 1, 3, 4, 5, 7 below (model-id research, recipe authoring, discovery, preflight,
> deploy, verify) still apply as written; the recipe content and `container:` value are
> superseded by the authored file. The original task bodies are kept below for the
> command-level detail (API calls, log milestones, verification).

**Goal:** Add a `recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml` recipe to the `spark-vllm-docker` repo and stand up one verified NVFP4 TP=4 deployment of Nemotron 3 Ultra across 4 Spark nodes, driven entirely through the dgx-manager REST API.

**Architecture:** The recipe is a diff of the proven `nemotron-3-super-nvfp4.yaml` (same Nemotron hybrid-Mamba/MoE vLLM flags) bumped to `tensor_parallel: 4` and placed in the 4-node cluster dir. Multi-node Ray + IB/RDMA plumbing is already proven by existing `4x-spark-cluster/` recipes, so this is recipe authoring + a live bring-up, not new code. Deployment is exclusively via `POST /api/deployments`; logs/status are observed via REST and SSE; inference is verified straight against the deployment's node IP:port (the `/lb` proxy is unmounted, out of scope).

**Tech Stack:** vLLM (NVFP4 / ModelOpt, CUTLASS MoE kernels, Ray distributed executor), DGX Manager REST API (Express 5, port 4000), `spark-vllm-docker` recipe YAML, SQLite/Prisma.

**Key facts (verified in code):**
- List recipes: `GET /api/recipes` → array of `{file,name,description?,model?,container,cluster_only?,solo_only?,defaults}` (`packages/server/src/routes/recipes.ts:6`).
- Force agent rescan without restart: `POST /api/recipes/refresh` (`packages/server/src/routes/recipes.ts:16`; agent handler `packages/agent/src/index.ts:924` `cmd:rescan-recipes`).
- Idle nodes: `GET /api/nodes/idle` → online nodes (`packages/server/src/routes/nodes.ts:57`).
- Create deployment: `POST /api/deployments` with `{nodeIds:"auto", recipeFile, runtime:"vllm", config, displayName}`; with `nodeIds:"auto"` the server computes `needed = tp*pp` and 409s if fewer online (`packages/server/src/routes/deployments.ts:91-125`). Returns the created Deployment incl. `id`, `status`, `port`, `node`, `clusterNodes`.
- Deploy logs: `GET /api/deployments/:id/logs?tail=N` (file-backed, `deployments.ts:31`); live via SSE `GET /api/events` event types `deployment:status` / `deployment:log` (`agent-hub.ts:406-457`).
- Status flow: `pending → starting → building → downloading → launching → loading → running | failed` (`agent-hub.ts:406`).
- `/lb` inference proxy is NOT mounted (`proxy/inference-proxy.ts` exists but no `app.use` in `index.ts`); verify inference direct against `http://<node.ipAddress>:<port>/v1/...`.

**Conventions used below:**
- `API=http://localhost:4000` (run curls on the manager host). If running remotely, use `MANAGER_ADVERTISE_HOST:4000`.
- `VLLM_REPO` = the local checkout of `spark-vllm-docker` whose `recipes/` the agents scan over NFS. On this machine: `/home/daniel/src/github/spark-vllm-docker` (also mirrored at `/mnt/tank/src/github/spark-vllm-docker`). Confirm which path the agents actually read (Task 3 verifies via the API rather than guessing).
- This plan touches TWO repos: the **recipe** is committed in `spark-vllm-docker`; the **spec + this plan** are committed in `dgx-manager` on `main`.

---

## Task 1: Confirm Ultra NVFP4 model identity and validate carried-over flags

No code. Research step that resolves the `<size>` placeholder and confirms the Super-derived flags fit Ultra's architecture. **Do not author the YAML until this is done** — a wrong parser/backend name is the most likely flag bug.

**Files:** none (produces facts used in Task 2).

- [ ] **Step 1: Resolve the exact HF model id**

Find the canonical NVFP4 checkpoint NVIDIA publishes for Nemotron 3 Ultra. Pattern from the family: Super is `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`, Nano is `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4`. Confirm Ultra's exact string (it encodes total/active params, e.g. `...-Ultra-<XXX>B-A<YY>B-NVFP4`).

Run (whichever is available on the host):
```bash
huggingface-cli search nvidia/NVIDIA-Nemotron-3-Ultra 2>/dev/null || \
curl -s "https://huggingface.co/api/models?search=NVIDIA-Nemotron-3-Ultra&author=nvidia" | python3 -m json.tool
```
Expected: one or more model ids; pick the `*-NVFP4` variant. Record the exact id as `MODEL_ID`.

- [ ] **Step 2: Read the model card / config to validate flags**

Fetch the model card and `config.json`:
```bash
curl -s "https://huggingface.co/api/models/$MODEL_ID" | python3 -m json.tool | head -60
curl -s "https://huggingface.co/$MODEL_ID/raw/main/config.json" | python3 -m json.tool
```
Check and record answers to each (these map 1:1 to recipe flags carried from Super):
  - Architecture: is it the same hybrid Mamba/MoE family as Super? (`config.json` `architectures`, presence of mamba/ssm + MoE fields). → confirms `--moe-backend cutlass` and `--mamba_ssm_cache_dtype float32` still apply.
  - Reasoning parser: does NVIDIA's card recommend `nemotron_v3`? (Super uses `nemotron_v3`; Nano uses `nano_v3` + a plugin file). → confirm `--reasoning-parser nemotron_v3` or adjust.
  - Tool-call parser: Super uses `qwen3_coder`. Confirm the card doesn't specify a different one.
  - Any required `mods/` (Nano/Super reference `mods/nemotron-*`). Check `VLLM_REPO/recipes/mods/` and whether Ultra needs one.
  - Max context length (`max_position_embeddings`) → informs the eventual `max_model_len` ceiling.
  - Quant config: confirm `hf_quant_config.json` / `quantization_config` indicates ModelOpt NVFP4 (so vLLM auto-detects; no explicit `--quantization` flag needed — Super doesn't pass one).

- [ ] **Step 3: Write down the resolved values**

Produce a short note (paste into the deploy task / PR description) with: `MODEL_ID`, confirmed flag set (and any deltas vs Super), whether a `mods/` entry is needed, and the model's max context. Expected outcome: either "flags identical to Super" or an explicit, small list of changes.

---

## Task 2: Author the recipe YAML in spark-vllm-docker

**Files:**
- Create: `<VLLM_REPO>/recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml`

- [ ] **Step 1: Write the recipe file**

Use the resolved `MODEL_ID` and any flag deltas from Task 1. Starting content (replace `MODEL_ID` literally; adjust parser/backend lines only if Task 1 found deltas):

```yaml
# Recipe: Nemotron-3-Ultra-NVFP4 (4-node TP=4)
# Diff of nemotron-3-super-nvfp4.yaml: same Nemotron hybrid-Mamba/MoE flags,
# tensor_parallel 2 -> 4, placed in 4x-spark-cluster, cluster_only, SPREAD placement.
recipe_version: "1"
name: Nemotron-3-Ultra-NVFP4
description: vLLM serving Nemotron 3 Ultra (NVFP4) across a 4-node TP=4 Spark cluster

model: MODEL_ID                      # <- from Task 1, e.g. nvidia/NVIDIA-Nemotron-3-Ultra-...-NVFP4
container: vllm-node
cluster_only: true                   # too large for a single 128GB node

env:
  VLLM_FLASHINFER_ALLREDUCE_BACKEND: trtllm
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: 1
  VLLM_DISTRIBUTED_EXECUTOR_CONFIG: '{"placement_group_options":{"strategy":"SPREAD"}}'

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 4
  gpu_memory_utilization: 0.8        # conservative first launch; raise once stable
  max_model_len: 32768               # conservative first launch; raise toward model max
  max_num_seqs: 10

command: |
  vllm serve MODEL_ID \
    --trust-remote-code \
    --moe-backend cutlass \
    --mamba_ssm_cache_dtype float32 \
    --kv-cache-dtype fp8 \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser nemotron_v3 \
    --attention-backend TRITON_ATTN \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-model-len {max_model_len} \
    --max-num-seqs {max_num_seqs} \
    --host {host} \
    --port {port} \
    --tensor-parallel-size {tensor_parallel} \
    --distributed-executor-backend ray
```

- [ ] **Step 2: Verify the YAML parses and templating vars resolve**

```bash
python3 -c "import yaml,sys; d=yaml.safe_load(open('$VLLM_REPO/recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml')); print('OK', d['name'], 'tp=', d['defaults']['tensor_parallel'], 'cluster_only=', d.get('cluster_only'))"
```
Expected: `OK Nemotron-3-Ultra-NVFP4 tp= 4 cluster_only= True`. Also eyeball that every `{var}` in `command` (`gpu_memory_utilization`, `max_model_len`, `max_num_seqs`, `host`, `port`, `tensor_parallel`) has a matching key in `defaults`.

- [ ] **Step 3: Commit the recipe in the spark-vllm-docker repo**

```bash
cd $VLLM_REPO
git add recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml
git commit -m "recipes: add Nemotron 3 Ultra NVFP4 4-node TP=4 recipe"
```
(Do NOT push yet if the team reviews recipes separately — confirm their workflow. Pushing/merging in this repo is the user's call.)

---

## Task 3: Make the recipe visible to agents and verify discovery via the API

**Files:** none (operational verification through the API).

- [ ] **Step 1: Ensure the recipe is on the NFS path the agents scan**

The agents scan their local `spark-vllm-docker/recipes` (NFS-mounted). If `$VLLM_REPO` is that same NFS checkout, the file is already visible. If agents read a different clone, sync it (e.g. `git pull` on the NFS checkout, or ensure `$VLLM_REPO` IS the NFS path `/mnt/tank/src/github/spark-vllm-docker`). Confirm the file exists where agents look:
```bash
ls -l /mnt/tank/src/github/spark-vllm-docker/recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml
```
Expected: file present. (If `$VLLM_REPO` was the NFS path, this is automatic.)

- [ ] **Step 2: Trigger an agent rescan (no restart needed)**

```bash
curl -s -X POST $API/api/recipes/refresh ; echo
```
Expected: `{"refreshed":N}` where N = number of connected agents (≥1).

- [ ] **Step 3: Confirm the recipe is discovered with the right shape**

```bash
curl -s $API/api/recipes | python3 -c "import sys,json; rs=json.load(sys.stdin); r=[x for x in rs if x['file'].endswith('4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml')]; print(json.dumps(r, indent=2))"
```
Expected: exactly one recipe object with `cluster_only: true` and `defaults.tensor_parallel: 4`. Record its `file` value (the relative path) — this is the `recipeFile` for the deploy. If empty, recheck Step 1 (wrong clone / file not on NFS) before proceeding.

---

## Task 4: Preflight — confirm ≥4 idle nodes online

**Files:** none.

- [ ] **Step 1: Count online nodes**

```bash
curl -s $API/api/nodes/idle | python3 -c "import sys,json; n=json.load(sys.stdin); print('idle online nodes:', len(n)); [print(' -', x['name'], x['ipAddress'], x.get('arch')) for x in n]"
```
Expected: `idle online nodes: 4` (or more). If fewer than 4, stop — the deploy will 409. (The server enforces `needed = tp*pp = 4`.) Note the node archs are consistent (TP group should be homogeneous).

---

## Task 5: Deploy via the API and drive the bring-up to `running`

**Files:** none (live deploy + observation). Follows the standing rule: deployment and any teardown go through the API only — never `ssh`/`docker` on a node.

- [ ] **Step 1: Create the deployment**

```bash
DEPLOY_JSON=$(curl -s -X POST $API/api/deployments \
  -H 'Content-Type: application/json' \
  -d '{
    "nodeIds": "auto",
    "recipeFile": "recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml",
    "runtime": "vllm",
    "displayName": "nemotron-3-ultra"
  }')
echo "$DEPLOY_JSON" | python3 -m json.tool
DEPLOY_ID=$(echo "$DEPLOY_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")
echo "DEPLOY_ID=$DEPLOY_ID"
```
Expected: HTTP 201, JSON with `id`, `status:"pending"`, `clusterMode:true`, and `clusterNodes` listing 4 nodes (one `role:"head"`). If you get 409 "requires 4 nodes", revisit Task 4. Save `DEPLOY_ID`.

- [ ] **Step 2: Watch the bring-up through milestones**

Poll status + tail logs (repeat until terminal state). Status endpoint is the single-deployment include; logs are file-backed:
```bash
# status
curl -s $API/api/deployments | python3 -c "import sys,json,os; d=[x for x in json.load(sys.stdin) if x['id']==os.environ['DEPLOY_ID']][0]; print('status:', d['status'], 'port:', d.get('port'))" 
# logs (last 60 lines)
curl -s "$API/api/deployments/$DEPLOY_ID/logs?tail=60"
```
(For live streaming instead of polling: `curl -sN $API/api/events` and grep for `deployment:status`/`deployment:log` with this `DEPLOY_ID`.)

Expected milestone progression in logs/status: `building`/`downloading` (image sync + NVFP4 weights pull to NFS) → `launching` (Ray head + 3 workers join; look for a Ray cluster with 4 nodes) → `loading` (weights load, CUTLASS NVFP4 kernels init) → `running` with `Application startup complete` and a 4-rank tensor-parallel group. Record the assigned `port`.

- [ ] **Step 3: If it fails — diagnose by bucket, fix, clean up, retry**

Read the tail of the log to classify:
  - **Capacity** (`CUDA out of memory`, KV-cache / `No available memory for the cache blocks`): lower `max_model_len` (e.g. 16384) and/or `gpu_memory_utilization` (e.g. 0.75) in the recipe, re-commit, `POST /api/recipes/refresh`, redeploy.
  - **Flag carry-over** (`unknown/unsupported argument`, `parser ... not found`, `moe-backend`): correct the offending flag per Task 1 findings, re-commit, refresh, redeploy.
  - **Plumbing** (Ray won't form, IB iface missing): unexpected (proven elsewhere) — capture the log and treat as a possible agent bug → Task 7.

Before each retry, delete the failed deployment record so we don't leave a trail (standing rule — clean up failed deployments):
```bash
curl -s -X DELETE $API/api/deployments/$DEPLOY_ID ; echo
```
Then repeat Step 1. Expected: eventually `status: running`.

---

## Task 6: Verify inference directly against the deployment

**Files:** none. The `/lb` proxy is unmounted, so hit the node:port directly.

- [ ] **Step 1: Resolve the head node IP and port**

```bash
curl -s $API/api/deployments | python3 -c "import sys,json,os; d=[x for x in json.load(sys.stdin) if x['id']==os.environ['DEPLOY_ID']][0]; print('IP', d['node']['ipAddress'], 'PORT', d['port'])"
```
Record `NODE_IP` and `PORT` (PORT is typically 8000 unless auto-bumped).

- [ ] **Step 2: Discover the served model name**

```bash
curl -s http://$NODE_IP:$PORT/v1/models | python3 -m json.tool
```
Expected: a `data[0].id` — the served model name. Record it as `SERVED`.

- [ ] **Step 3: Send a chat completion**

```bash
curl -s http://$NODE_IP:$PORT/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SERVED\",\"messages\":[{\"role\":\"user\",\"content\":\"In one sentence, what is tensor parallelism?\"}],\"max_tokens\":128}" \
  | python3 -m json.tool
```
Expected: HTTP 200 with a coherent `choices[0].message.content`. This is the definition-of-done inference check. Note tokens/sec from the log if you want a throughput baseline.

---

## Task 7 (CONTINGENCY — only if Task 5 exposes a genuine agent launch bug)

Skip entirely if the deploy reached `running` without an agent code change. If a real bug in the agent launch path surfaced (e.g. `buildLaunchArgs` emits a wrong flag for this recipe shape, or cluster wiring mishandles a field), fix it with TDD. Example shape for a `buildLaunchArgs` fix:

**Files:**
- Modify: `packages/agent/src/runtime/vllm.ts` (the function with the bug)
- Test: `packages/agent/src/runtime/vllm.test.ts`

- [ ] **Step 1: Write the failing test capturing the exact bug**

```typescript
// in packages/agent/src/runtime/vllm.test.ts
import { describe, it, expect } from "vitest";
import { buildLaunchArgs } from "./vllm.js";

it("cluster TP=4 recipe passes --tp 4 and node IPs to run-recipe.sh", () => {
  const args = buildLaunchArgs({
    recipeName: "nemotron-3-ultra-nvfp4",
    clusterNodes: ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
    tp: 4,
    port: 8000,
    servedModelName: "nemotron-3-ultra",
  });
  // Replace the assertion with the precise correct expectation for the bug found:
  expect(args).toContain("--tp");
  expect(args.join(" ")).toContain("10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4");
});
```
(Adjust the call signature to the real `buildLaunchArgs` — see `packages/agent/src/runtime/vllm.ts:191-225` — and assert exactly what the bug got wrong.)

- [ ] **Step 2: Run it, confirm it fails for the right reason**

```bash
npx vitest run packages/agent/src/runtime/vllm.test.ts -t "cluster TP=4"
```
Expected: FAIL showing the buggy output.

- [ ] **Step 3: Make the minimal fix in `vllm.ts`**

Edit only what's needed to satisfy the assertion. Keep IO out of the pure helper.

- [ ] **Step 4: Run the focused test, then the full suite**

```bash
npx vitest run packages/agent/src/runtime/vllm.test.ts -t "cluster TP=4"
npm test
```
Expected: target test PASS; `npm test` → all green.

- [ ] **Step 5: Bump the agent version (MANDATORY for any packages/agent/src edit)**

```bash
./scripts/bump-agent-version.sh
```

- [ ] **Step 6: Rebuild + redeploy the agent bundle, then re-run Task 5**

```bash
./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=<host> SSH_USER=<user> docker compose up -d --build
```
(Agents must pick up the new bundle so the launch fix takes effect.) Then redeploy via Task 5.

- [ ] **Step 7: Commit the fix (dgx-manager)**

```bash
cd /home/daniel/src/github/dgx-manager
git add packages/agent/src/runtime/vllm.ts packages/agent/src/runtime/vllm.test.ts packages/agent/package.json
git commit -m "agent: fix vLLM launch arg for cluster TP=4 recipe + bump version"
```

---

## Task 8: Land the spec + this plan on main

The recipe is committed in `spark-vllm-docker` (Task 2). The design docs land in `dgx-manager` on `main`.

**Files:**
- `docs/superpowers/specs/2026-06-04-nemotron-3-ultra-nvfp4-tp4-design.md`
- `docs/superpowers/plans/2026-06-04-nemotron-3-ultra-nvfp4-tp4.md`

- [ ] **Step 1: Confirm on main and clean tree (besides these docs)**

```bash
cd /home/daniel/src/github/dgx-manager
git branch --show-current   # expect: main
git status --short          # expect: only the two doc files (untracked)
```

- [ ] **Step 2: Commit the docs**

```bash
git add docs/superpowers/specs/2026-06-04-nemotron-3-ultra-nvfp4-tp4-design.md \
        docs/superpowers/plans/2026-06-04-nemotron-3-ultra-nvfp4-tp4.md
git commit -m "docs: Nemotron 3 Ultra NVFP4 TP=4 recipe spec + plan"
```

- [ ] **Step 3: Push (per earlier decision to keep main in sync)**

```bash
git push origin main
```
Expected: push succeeds; `git status -sb` shows `## main...origin/main` with nothing ahead.

---

## Self-Review (completed by author)

**Spec coverage:**
- Deliverable recipe (`4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml`) → Task 2.
- Super-flag carry-over + arch validation → Task 1.
- NFS model storage / HF id → Tasks 1–2 (recipe references HF id; vLLM downloads to NFS).
- Recipe discovery (subdir, no agent change) → Task 3 (verified via API + rescan endpoint).
- Preflight (4 idle nodes) → Task 4.
- Deploy via API only + staged log reading → Task 5.
- Capacity / flag risk buckets + failed-deploy cleanup → Task 5 Step 3.
- Contingency agent fix with version bump + test → Task 7.
- Inference verification (corrected to direct node:port, `/lb` unmounted) → Task 6.
- Docs on main → Task 8.

**Placeholder scan:** `MODEL_ID` and `<size>` are intentional, resolved in Task 1 before any YAML is written — not silent placeholders. `<host>`/`<user>` in Task 7 Step 6 are the deployer's compose env, same as CLAUDE.md examples.

**Type/name consistency:** `recipeFile` value (`recipes/4x-spark-cluster/nemotron-3-ultra-nvfp4.yaml`) is consistent across Tasks 2/3/5; `DEPLOY_ID`, `NODE_IP`, `PORT`, `SERVED`, `MODEL_ID`, `VLLM_REPO`, `API` are defined where first used and reused consistently. API endpoints match the verified `file:line` references in the header.
