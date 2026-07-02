# GLM-5.2 Decode Performance (4.6 → 20–34 tok/s) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring GLM-5.2 single-stream decode on the 4×GB10 DGX Spark cluster from the measured 4.6 tok/s to the 20–34 tok/s the reference config achieves on identical hardware and the identical pruned model, by adopting MTP k=3 speculative decoding + the full NCCL/RDMA env, and fix the dashboard `meanTps` metric that masked the problem.

**Architecture:** Mostly an **ops/config change** in two external places (the `kreuzhofer/community-recipe-registry` recipe, HF drafter weights on `/mnt/tank`) plus one small TDD code fix in this repo (`summarizeResults` blends prefill+decode tps into a meaningless number). Deploy/benchmark runs through the Pi manager (`192.168.44.14`) as usual. Conditional follow-on experiments (draft-TP=1, `mp` vs `ray` executor) are gated on measured results.

**Tech Stack:** vLLM (custom `vllm-node-tf5-glm52-b12x:probe` image), sparkrun (eugr launch-cluster, Ray backend), NCCL/RoCE, Prisma/Express (dgx-manager server), Vitest.

**Root-cause evidence (verified 2026-07-02):**
- Our benchmark `cmr1a4tm02pp136k42hhf2d9u`: decode (tg) **4.56 tok/s mean / 6 peak**, prefill (pp) 67 tok/s. The recorded `meanTps: 35.81` is the average of the pp and tg rows — a display bug, not a real speed.
- Reference: [CosmicRaisins/glm-5.2-gb10](https://github.com/CosmicRaisins/glm-5.2-gb10) + [NVIDIA forum thread](https://forums.developer.nvidia.com/t/glm-5-2-on-a-4x-gb10-cluster-22-tok-s-decode-256k-ctx-recipe/374125) — **same 4×GB10, same `CosmicRaisins/GLM-5.2-AWQ-INT4-15pct` model, same Triton sm12x kernels/b12x/cudagraph FULL** (our image is built from his kernels) → **20.2–21.9 tok/s decode** (pp=2048/tg=256, c=1); with draft-TP=1 + no expert parallelism **31–34 tok/s**.
- Author: "**MTP gave the biggest uplift; CUDA Graph only made a 3% or so difference.**" Without MTP his baseline is ~8 tok/s (eager).
- His `launch.sh` warns: without `--device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1`, "NCCL **silently** drops to TCP: ~12 vs 30+ tok/s". Our eugr launcher runs `--privileged --ipc=host` (devices visible) but our env lacks `NCCL_NET=IB`, `NCCL_IB_HCA` (both rails), `NCCL_IB_GID_INDEX=3`, `NCCL_CROSS_NIC=1`, `NCCL_CUMEM_ENABLE=0`.

**Key config deltas (theirs vs ours):**

| knob | reference (20–34 tok/s) | ours (4.6 tok/s) |
|---|---|---|
| speculative decoding | `--speculative-config '{"model":".../glm52-mtp-int4-aligned","method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}'` | none |
| executor | `mp` + vLLM native multi-node (`--nnodes/--node-rank`) | `ray` via eugr launch-cluster |
| NCCL env | `NCCL_NET=IB`, `NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0`, `NCCL_IB_GID_INDEX=3`, `NCCL_CROSS_NIC=1`, `NCCL_CUMEM_ENABLE=0`, 3-interface `NCCL_SOCKET_IFNAME` | `NCCL_IB_DISABLE=0`, single-rail ifname |
| memory | gpu-mem 0.93, max-model-len 262144, prefix caching | 0.88, 57344 (deployed at 32768) |

**Global constraints:**
- Manager is the **Pi** (`http://192.168.44.14:4000`); run API calls there (`localhost` when working on the Pi). Head node `gx10-01` = `192.168.44.36`.
- Recipe lives in the **separate repo** `kreuzhofer/community-recipe-registry` (`recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml`). Known gotcha: `POST /api/recipes/refresh` updates `sparkrun list` only — the **run-cache clone on every cluster node** (`~/.cache/sparkrun/registries/community-kreuzhofer/`) must be `git pull`ed.
- Container HF cache `/cache/huggingface` ↔ host `/mnt/tank/models` (agent env `HF_HOME=/mnt/tank/models`). The drafter is referenced by HF repo id and resolved from that cache.
- GLM-5.2 startup is **~45 min** (12 min shard load + ~21 min silent AWQ/MoE processing at 0 % GPU — looks wedged, isn't + cudagraph capture). Don't kill it early.
- Deploy consumes all 4 DGX nodes. First MTP deploy keeps the known-good memory config (gpu-mem 0.88, max-model-len 32768) because the drafter weights eat into the 0.88 budget; raising toward 0.93/longer ctx is a separate optional task (the head node has ~112 GiB freed since the manager moved to the Pi).
- Success gates: **≥ 18 tok/s decode** after Task 5 (reference: 20.2); stretch **≥ 28** after Task 6.

---

### Task 1: Fix `meanTps` to be decode-only (TDD)

The dashboard/API `meanTps` averages prefill and decode rows (67 + 4.56)/2 → "35.8 tok/s" for a 4.6 tok/s model. Make it decode-only and label it honestly.

**Files:**
- Modify: `packages/server/src/benchmarks/parser.ts:98-113` (`summarizeResults`)
- Test: `packages/server/src/benchmarks/parser.test.ts:80-93`
- Modify: `packages/dashboard/app/benchmarks/page.tsx:184` (column header)

- [ ] **Step 1: Update the `summarizeResults` tests to specify decode-only semantics**

Replace the existing `describe("summarizeResults", …)` block in `packages/server/src/benchmarks/parser.test.ts` with:

```typescript
describe("summarizeResults", () => {
  it("computes mean tps over decode (tg) rows only — prefill must not inflate it", () => {
    const rows = parseBenchyResults(fixture);
    const summary = summarizeResults(rows);
    // fixture tg rows tps: (84.5 + 220.3) / 2 = 152.4 — NOT the pp+tg blend 756.3
    expect(summary.meanTps).toBeCloseTo(152.4, 1);
    // ttfr is per-workload (identical on the pp and tg rows of a workload):
    // (142.3 + 410.0) / 2 = 276.15
    expect(summary.meanTtfrMs).toBeCloseTo(276.15, 1);
  });

  it("returns null meanTps when there are no tg rows", () => {
    const rows = parseBenchyResults(fixture).filter((r) => r.opType === "pp");
    expect(summarizeResults(rows).meanTps).toBeNull();
  });

  it("returns nulls when given no rows", () => {
    expect(summarizeResults([])).toEqual({ meanTps: null, meanTtfrMs: null });
  });
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `npx vitest run packages/server/src/benchmarks/parser.test.ts`
Expected: FAIL — `meanTps` is 756.3 (the blend), not 152.4.

- [ ] **Step 3: Implement decode-only summarization**

Replace `summarizeResults` in `packages/server/src/benchmarks/parser.ts` with:

```typescript
export function summarizeResults(rows: BenchmarkResultInput[]): {
  meanTps: number | null;
  meanTtfrMs: number | null;
} {
  if (rows.length === 0) return { meanTps: null, meanTtfrMs: null };
  // meanTps is a DECODE metric: average tg rows only. Averaging prefill tps
  // into it produced numbers like "35.8 tok/s" for a model that generates at
  // 4.6 tok/s (benchmark cmr1a4tm02pp136k42hhf2d9u).
  const tgRows = rows.filter((r) => r.opType === "tg");
  const meanTps =
    tgRows.length === 0
      ? null
      : tgRows.reduce((acc, r) => acc + r.tps, 0) / tgRows.length;
  // ttfr is per-workload (shared by a workload's pp and tg rows) — average it
  // over the same tg rows so both summary metrics describe the same set.
  const ttfrRows = (tgRows.length > 0 ? tgRows : rows).filter(
    (r) => r.ttfrMs !== null,
  ) as Array<BenchmarkResultInput & { ttfrMs: number }>;
  const meanTtfrMs =
    ttfrRows.length === 0
      ? null
      : ttfrRows.reduce((acc, r) => acc + r.ttfrMs, 0) / ttfrRows.length;
  return { meanTps, meanTtfrMs };
}
```

- [ ] **Step 4: Rename the dashboard column**

In `packages/dashboard/app/benchmarks/page.tsx` line 184, change:

```tsx
<th className="px-4 py-3 text-right">Mean t/s</th>
```
to
```tsx
<th className="px-4 py-3 text-right">Decode t/s</th>
```

- [ ] **Step 5: Run the full test suite**

Run: `npm test`
Expected: all green. (If another test asserts the old blended `meanTps` — e.g. an orchestrator/integration test — update its expectation to the tg-only value; the decode-only semantics are the spec.)

- [ ] **Step 6: Correct the stale headline in the benchmark doc**

In `docs/glm-5.2-inference-benchmark.md`, replace the TL;DR line

```
- **Throughput: 35.8 tok/s**, TTFR 1552 ms (`quick-smoke` preset, cudagraph FULL, c=1)
```
with
```
- **Decode: 4.6 tok/s** (prefill 67 tok/s), TTFR 1552 ms (`quick-smoke`, cudagraph FULL, c=1) —
  the previously reported "35.8 tok/s" was a prefill+decode blend from a since-fixed
  `meanTps` bug. See `2026-07-02-glm52-decode-perf.md` for the fix plan (MTP → 20+ tok/s).
```
and in the Results table replace the `mean tok/s | 35.81` row with `decode tok/s | **4.56** (peak 6.0)` plus a `prefill tok/s | 67.1` row.

- [ ] **Step 7: Commit**

```bash
git add packages/server/src/benchmarks/parser.ts packages/server/src/benchmarks/parser.test.ts \
        packages/dashboard/app/benchmarks/page.tsx docs/glm-5.2-inference-benchmark.md
git commit -m "fix(benchmarks): meanTps is decode-only, not a prefill+decode blend"
```

(No agent files touched — no agent version bump. The Pi server keeps serving the old value until its image is rebuilt; that rebuild is Task 7 Step 4, bundled with the doc updates.)

---

### Task 2: Stage the MTP drafter weights on shared storage

**Files:** none in-repo (weights on `/mnt/tank`).

**Interfaces:** Produces `models--CosmicRaisins--GLM-5.2-MTP-INT4-aligned` in the HF cache all nodes mount, consumed by the recipe's `--speculative-config` (Task 3).

- [ ] **Step 1: Confirm the cache mapping assumption**

Run: `ssh daniel@192.168.44.36 'ls /mnt/tank/models/hub | grep -i glm; ls /mnt/tank/models/nccl-2.30.4/libnccl.so.2'`
Expected: `models--CosmicRaisins--GLM-5.2-AWQ-INT4-15pct` and the libnccl path both exist — confirms host `/mnt/tank/models` is the container's `/cache/huggingface`.

- [ ] **Step 2: Download the drafter into the shared HF cache**

```bash
ssh daniel@192.168.44.36 'HF_HOME=/mnt/tank/models python3 -m huggingface_hub.commands.huggingface_cli \
  download CosmicRaisins/GLM-5.2-MTP-INT4-aligned 2>/dev/null || \
  HF_HOME=/mnt/tank/models hf download CosmicRaisins/GLM-5.2-MTP-INT4-aligned'
```
(Whichever HF CLI form the node has; the agent env already carries a valid `HF_TOKEN`.)

- [ ] **Step 3: Verify the snapshot is complete**

Run: `ssh daniel@192.168.44.36 'du -sh /mnt/tank/models/hub/models--CosmicRaisins--GLM-5.2-MTP-INT4-aligned/ && ls /mnt/tank/models/hub/models--CosmicRaisins--GLM-5.2-MTP-INT4-aligned/snapshots/*/ | head'`
Expected: a few GB; `config.json` + safetensors present in the snapshot dir.

- [ ] **Step 4: Commit — none (weights, not repo content).**

---

### Task 3: Recipe update — MTP k=3 + full NCCL/RDMA env

**Files:** `recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml` in the **`kreuzhofer/community-recipe-registry`** repo (not this repo).

**Interfaces:** Consumed by `sparkrun run` on the nodes (via the per-node run-cache clone).

- [ ] **Step 1: Clone/pull the registry repo locally**

```bash
git clone git@github.com:kreuzhofer/community-recipe-registry.git ~/src/github/kreuzhofer/community-recipe-registry 2>/dev/null || \
  git -C ~/src/github/kreuzhofer/community-recipe-registry pull
```

- [ ] **Step 2: Rewrite the recipe**

Replace the full contents of `recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml` with:

```yaml
recipe_version: "2"
model: CosmicRaisins/GLM-5.2-AWQ-INT4-15pct
runtime: vllm
container: vllm-node-tf5-glm52-b12x:probe
cluster_only: true
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 4
  gpu_memory_utilization: 0.88
  # 32768 (not 57344): the MTP drafter weights come out of the 0.88 budget,
  # shrinking KV. Raise ctx only after the MTP deploy is proven (see plan Task 7).
  max_model_len: 32768
  served_model_name: glm-5.2
env:
  LD_PRELOAD: /cache/huggingface/nccl-2.30.4/libnccl.so.2
  VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: "1800"
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB: "256"
  GLM52_BIND_HOST_TRITON: "1"
  GLM52_MQA_LOGITS_TRITON: "1"
  GLM52_PAGED_MQA_TRITON: "1"
  GLM52_PAGED_MQA_TOPK_CHUNK_SIZE: "8192"
  GLM52_B12X_MLA: "1"
  # --- NCCL/RDMA stanza from CosmicRaisins/glm-5.2-gb10 (same NIC/HCA names
  # on our Sparks: verified rocep1s0f0/roceP2p1s0f0 + enp1s0f0np0/enP2p1s0f0np0).
  # Without NCCL_NET=IB + HCA + GID index, NCCL can silently fall back to TCP
  # (~12 vs 30+ tok/s per the reference launch.sh).
  NCCL_NET: IB
  NCCL_IB_DISABLE: "0"
  NCCL_IB_HCA: "rocep1s0f0,roceP2p1s0f0"
  NCCL_SOCKET_IFNAME: "enP7s7,enp1s0f0np0,enP2p1s0f0np0"
  GLOO_SOCKET_IFNAME: enp1s0f0np0
  NCCL_IB_GID_INDEX: "3"
  NCCL_CROSS_NIC: "1"
  NCCL_CUMEM_ENABLE: "0"
  NCCL_IGNORE_CPU_AFFINITY: "1"
  # INFO for the validation deploy so transport selection is visible in logs
  # (grep "NET/IB"). Flip to WARN in plan Task 7 once RDMA is confirmed.
  NCCL_DEBUG: INFO
  PYTORCH_CUDA_ALLOC_CONF: expandable_segments:True
command: |
  vllm serve {model} --served-model-name {served_model_name} --host {host} --port {port} --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice --speculative-config '{"model":"CosmicRaisins/GLM-5.2-MTP-INT4-aligned","method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}' -tp {tensor_parallel} --pipeline-parallel-size 1 --distributed-executor-backend ray --max-model-len {max_model_len} --max-num-seqs 1 --max-num-batched-tokens 4096 --gpu-memory-utilization {gpu_memory_utilization} --kv-cache-dtype fp8_ds_mla --compilation-config '{"cudagraph_mode":"FULL"}'
```

Deliberate deviations from the reference (keep, don't "fix"): `ray` executor (sparkrun's runtime — the `mp` experiment is gated in Task 6b), gpu-mem 0.88 + 32K ctx (first-deploy safety), no `--enable-prefix-caching` yet (orthogonal to decode speed at depth 0; add later with the ctx bump).

- [ ] **Step 3: Commit + push the registry repo**

```bash
cd ~/src/github/kreuzhofer/community-recipe-registry
git add recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml
git commit -m "glm-5.2: MTP k=3 speculative decoding + full NCCL/RDMA env (target 20+ tok/s decode)"
git push
```

- [ ] **Step 4: Propagate to every cluster node's run-cache (known gotcha) + manager list**

```bash
for ip in 192.168.44.36 192.168.44.37 192.168.44.38 192.168.44.39; do
  ssh daniel@$ip 'git -C ~/.cache/sparkrun/registries/community-kreuzhofer pull --ff-only' && echo "pulled $ip";
done
curl -s -X POST http://localhost:4000/api/recipes/refresh
```
Expected: 4× `pulled`, refresh returns OK.

- [ ] **Step 5: Verify a node sees the new recipe content**

Run: `ssh daniel@192.168.44.36 'grep -c "speculative-config\|NCCL_IB_GID_INDEX" ~/.cache/sparkrun/registries/community-kreuzhofer/recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml'`
Expected: `2`.

---

### Task 4: Deploy with MTP + verify RDMA and speculative decoding are live

**Files:** none.

- [ ] **Step 1: Deploy via the Pi manager**

```bash
curl -s -X POST http://localhost:4000/api/deployments -H 'Content-Type: application/json' -d '{
  "nodeIds": "auto",
  "recipeFile": "@community-kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer",
  "displayName": "glm-5.2"
}'
```
Expected: 200/201 with a deployment id. Record it as `$DEP`.

- [ ] **Step 2: Wait out the known ~45 min startup**

Poll every few minutes: `curl -s http://localhost:4000/api/deployments/$DEP | python3 -c 'import sys,json;d=json.load(sys.stdin);print(d["status"])'`
Reminder: ~21 min of 0 % GPU during AWQ/MoE weight processing is NORMAL — do not restart. Escalate only past ~60 min or on `failed`.

- [ ] **Step 3: Verify NCCL took the IB/RDMA path — the go/no-go gate**

```bash
ssh daniel@192.168.44.36 'docker logs $(docker ps --format "{{.Names}}" | grep -i vllm | head -1) 2>&1 | grep -m5 -E "NET/IB|NET/Socket|NCCL INFO Using"'
```
Expected: lines like `NCCL INFO NET/IB : Using ... [rocep1s0f0] ...`. **If `NET/Socket` appears instead, NCCL fell back to TCP — stop and fix before benchmarking** (check `docker exec <c> env | grep -E "NCCL|GLOO"` for whether the eugr launcher's own `-e NCCL_*` flags overrode the recipe's, and whether `/dev/infiniband` is visible in the container: `docker exec <c> ls /dev/infiniband`).

- [ ] **Step 4: Verify MTP is active**

```bash
ssh daniel@192.168.44.36 'docker logs $(docker ps --format "{{.Names}}" | grep -i vllm | head -1) 2>&1 | grep -i -m5 "speculative\|mtp\|draft"'
```
Expected: vLLM logs the speculative config / drafter load. Absence = the recipe change didn't take (re-check Task 3 Step 4).

- [ ] **Step 5: Smoke completion**

```bash
curl -s -m 120 http://192.168.44.36:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":40}' | head -c 600
```
Expected: coherent completion. (Also sanity-checks the drafter didn't wreck output quality — MTP proposals are verified by the main model, so quality should be unchanged.)

---

### Task 5: Benchmark apples-to-apples vs the reference

**Files:** none.

- [ ] **Step 1: Run the reference workload (pp=2048, tg=256, c=1)**

```bash
curl -s -X POST http://localhost:4000/api/benchmarks -H 'Content-Type: application/json' -d "{
  \"deploymentId\": \"$DEP\",
  \"config\": {\"pp\":[2048],\"tg\":[256],\"depth\":[0],\"runs\":2,\"concurrency\":[1],\"latencyMode\":\"api\",\"enablePrefixCaching\":false,\"skipCoherence\":false}
}"
```
Poll `GET /api/benchmarks/{id}` to `completed` (a few minutes).

- [ ] **Step 2: Read the decode number (from `results`, not the summary)**

```bash
curl -s http://localhost:4000/api/benchmarks/$BENCH | python3 -c '
import sys,json
d=json.load(sys.stdin)
for r in d["results"]:
    print(r["opType"], "tps=%.2f" % r["tps"], "peak=", r.get("peakTps"))'
```
**Gate: tg tps ≥ 18** (reference: 20.2 at this workload; our pre-MTP baseline: 4.56).
- ≥ 18 → proceed to Task 6 (stretch) or straight to Task 7 (finalize).
- 8–18 → MTP works but something still drags: confirm Task 4 Step 3 showed `NET/IB`; check drafter acceptance in logs; then run Task 6b (ray vs mp experiment).
- < 8 → MTP likely not engaged; re-verify Task 4 Step 4 before anything else.

- [ ] **Step 3: Also re-run `quick-smoke` for continuity with the old record**

Same POST with `"presetId": "quick-smoke"` instead of `config`. Record both run ids for the docs update (Task 7).

---

### Task 6 (stretch, optional): draft-TP=1 — the 31–34 tok/s config

Forum-reported uplift on the same cluster: draft model at TP=1 instead of sharded with the target.

- [ ] **Step 1: Edit the recipe's speculative-config** (registry repo), adding one field:

```
--speculative-config '{"model":"CosmicRaisins/GLM-5.2-MTP-INT4-aligned","method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE","draft_tensor_parallel_size":1}'
```
Commit/push + re-pull the 4 node run-caches (Task 3 Step 4 loop). Note: if vLLM rejects the key (name varies across versions — try `"draft_tensor_parallel_size"`; check `vllm serve --help` inside the container for the SpeculativeConfig schema), record that and skip this task rather than fight it.

- [ ] **Step 2: Redeploy (delete old deployment first) + re-run Task 5 Step 1.**
Gate: tg tps ≥ 28 keeps the change; below that, revert to the Task 3 config (it's one git revert in the registry).

### Task 6b (conditional experiment): quantify Ray overhead with the `mp` executor

Only if Task 5 lands in the 8–18 band. The reference uses `mp` + vLLM native multi-node instead of Ray; this bypasses sparkrun, so it's a **measurement experiment, not a production config**.

- [ ] **Step 1:** Stop the sparkrun deployment. Adapt the reference `launch.sh` (already fetched to the session scratchpad: `glm52-gb10/launch.sh`) — set `NODES=(192.168.44.36 192.168.44.37 192.168.44.38 192.168.44.39)`, `SSH_USER=daniel`, `IMAGE=vllm-node-tf5-glm52-b12x:probe`, `WEIGHTS_DIR=/mnt/tank/models`, and swap its `KMOUNTS` for nothing (our image has the kernels baked in). Launch, wait, benchmark the same workload via `POST /api/benchmarks` against the raw endpoint (the benchmarks route accepts a deployment; if a raw-endpoint path is needed, use `endpointUrl` support — see `routes/benchmarks.ts:306`).
- [ ] **Step 2:** Compare vs the Task 5 number. If `mp` is ≥ 25 % faster, file a follow-up (sparkrun native-multinode runtime or recipe-level executor override) as its own spec; tear the manual containers down and redeploy via sparkrun regardless, so the fleet stays manager-managed.

---

### Task 7: Finalize — quiet logs, docs, Pi image

**Files:**
- Modify: registry recipe (NCCL_DEBUG), `docs/glm-5.2-inference-benchmark.md` (results), this plan (execution status).

- [ ] **Step 1: Flip `NCCL_DEBUG: INFO` → `WARN` in the registry recipe** (commit/push + node run-cache pull loop). No redeploy needed just for this; it takes effect next deploy.

- [ ] **Step 2: Update `docs/glm-5.2-inference-benchmark.md`** with the new decode numbers (MTP config, benchmark run ids from Task 5), replacing the "Results" section and keeping the correction note from Task 1 Step 6.

- [ ] **Step 3: Optional ctx/memory bump task note.** If desired later: gpu-mem 0.88→0.93 and max_model_len 32768→ up to 262144 + `--enable-prefix-caching` (head node freed ~112 GiB since the manager moved to the Pi; reference runs 0.93/256K **with** MTP). Treat as its own deploy+verify cycle; abort criteria = vLLM's KV-budget error at startup.

- [ ] **Step 4: Rebuild the Pi server+dashboard images** so the Task 1 fix serves:

```bash
ssh daniel@192.168.44.14 'cd ~/dgx-manager && git pull && ./scripts/build-agent-bundles.sh && \
  MANAGER_ADVERTISE_HOST=192.168.44.14 SSH_USER=daniel docker compose up -d --build'
```
(Run on the Pi directly if this session is already there: drop the ssh wrapper.)

- [ ] **Step 5: Commit** (this repo: docs + plan status).

```bash
git add docs/glm-5.2-inference-benchmark.md docs/superpowers/plans/2026-07-02-glm52-decode-perf.md
git commit -m "docs: GLM-5.2 decode results with MTP k=3 (was 4.6 tok/s, meanTps bug corrected)"
```

---

## Notes for the implementer

- **The single highest-value change is the `--speculative-config` MTP block.** If you do only one thing, do that. The author of the reference config: "MTP gave the biggest uplift; CUDA Graph only made a 3 % or so difference."
- **`NET/Socket` in NCCL logs = silent TCP fallback.** The reference documents 12 vs 30+ tok/s for TCP vs RDMA *with* MTP. Traffic on the 200 G NIC does NOT prove RDMA (TCP also rides that NIC via `NCCL_SOCKET_IFNAME`) — only the `NET/IB` log line (or rising `ethtool -S <if> | grep rdma` counters during decode) proves it.
- **Env override risk:** the eugr `launch-cluster.sh` injects its own `-e NCCL_SOCKET_IFNAME=$ETH_IF -e NCCL_IB_HCA=$IB_IF` at `docker run`; depending on flag order these may override the recipe's env. That's why Task 4 Step 3 checks the *effective* env inside the container, not the recipe.
- **displayName "glm-5.2"** on the deploy keeps benchmark model-name resolution robust even though the served-model fix (`f439761`) is in the Pi image.
- **Do not** edit `~/.cache/sparkrun/registries/...` files directly on nodes — always via the registry repo + pull, or the next refresh clobbers the change.
- Reference materials fetched during planning live in the session scratchpad `glm52-gb10/` (recipe yaml, launch.sh, MTP README, retrospective).
