# GLM-5.2 Long-Context (DCP) Build — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **This is a BUILD/OPS plan, not code-TDD.** Steps are shell commands with **verification gates** (assertions on real output), not unit tests. Tasks run largely on a **GB10 node over SSH**, not the Pi. Do NOT parallelize — each task gates the next.

**Goal:** Build a new arm64/sm_121 vLLM image on the DCP stack and serve **GLM-5.2 (15pct) at ~262K context** via `--decode-context-parallel-size 2`, as an agentic-coding daily driver — without touching the working `b12x:probe` image or the shipped `144k` recipe.

**Architecture:** Re-run the documented arm64 build (`docs/glm-5.2-custom-image-build.md`) with the **DCP-branch vLLM** (`local-inference-lab/vllm codex/dcp-*`) + **PR#72 draft patches** instead of the legacy `ab6660699` ref, overlay the CosmicRaisins sm12x DSA kernels + b12x + deep_gemm re-bind, distribute to all 4 nodes, then deploy `15pct` with `VLLM_USE_V2_MODEL_RUNNER=1` + DCP2 via a new `@dgxrun` recipe. DCP shards the MLA KV across the 4 ranks (per-rank KV ÷ 4 ⇒ ~4× context); decode is unaffected (~25–31 on code), prefill pays ~linearly and is amortized by prefix caching.

**Tech Stack:** arm64/sm_121 (GB10), Docker, vLLM (DCP fork), b12x 0.23.0, CUDA (13.0 today; 13.2/cu132 TBD by Task 1), NCCL 2.30.4, the dgxrun runner.

## Global Constraints

- **Do NOT overwrite `vllm-node-tf5-glm52-b12x:probe`** — build to a **new tag `vllm-node-tf5-glm52-b12x-dcp:probe`** (and `…-dcp:base`). The current image + `@dgxrun/glm-5.2-awq-15pct-144k` recipe stay as the fallback.
- **Runs on a GB10 node via SSH** (the Pi cannot compile the arm64 CUDA image). Use `.38` as the build/head node. Build infra: `/mnt/tank/src/github/spark-vllm-docker/build-and-copy.sh` (shared NFS, visible on every node). Overlay context: `/mnt/tank/src/glm52-overlay/`.
- **DCP stack pins:** vLLM = `local-inference-lab/vllm` branch `codex/dcp-globaltopk-sharddraft-defaults-20260622` @ `e232d26`; **PR#72** patches (`pr72-1-draft-dcp-config-propagation.patch`, `pr72-2-glm-dcp-draft-path.patch`); **b12x==0.23.0** (verify master ≥ `80eb49b` for the fp8 decode path); latest CosmicRaisins `kernels/` + `patch_deep_gemm.py`.
- **Runtime:** `VLLM_USE_V2_MODEL_RUNNER=1` (env) + `--decode-context-parallel-size 2` (serve flag). Keep from the `144k` recipe: reduced sparse-MLA chunk env (`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128`, `VLLM_TRITON_MLA_SPARSE_QUERY_CHUNK_SIZE=128`, `TOPK_CHUNK_SIZE=256`), the NCCL/RDMA stanza, MTP k=3, `fp8_ds_mla` KV, `--enable-prefix-caching`, `LD_PRELOAD=/cache/huggingface/nccl-2.30.4/libnccl.so.2`.
- **Model:** `CosmicRaisins/GLM-5.2-AWQ-INT4-15pct` (already on `/mnt/tank`, **no download**). Unpruned QuantTrio (~400 GB pull) only if a later quality eval says the prune hurts.
- **The current `144k` deploy occupies all 4 nodes** — it must be stopped before Task 3 (build contends) and Task 7 (deploy needs all 4). Commit prefix `feat(dgxrun-dcp):` / `chore(build):`.

---

### Task 1: GATE — arm64 + DCP-fork + CUDA feasibility spike (STOP if it fails)

**Goal:** Before any ~1 h build, resolve the three unknowns that can block the whole thing. Produces the concrete `--vllm-ref`, remote-config, and CUDA-base decisions the later tasks consume. **If any gate fails, STOP and escalate — do not proceed to Task 2.**

**Files:** none (investigation only; record findings in the task report).

- [ ] **Step 1: Confirm the DCP branch exists + get its commit.**
```bash
gh api repos/local-inference-lab/vllm/branches/codex/dcp-globaltopk-sharddraft-defaults-20260622 --jq '.commit.sha' 2>&1 | head -1
```
Gate: prints a SHA (expected to start `e232d26…`). If 404, the branch moved — find the current DCP branch on `local-inference-lab/vllm` (or CosmicRaisins `CHANGES.md`) and record the new ref. **Record the exact SHA as `DCP_REF`.**

- [ ] **Step 2: Determine how `build-and-copy.sh` sources vLLM (upstream vs fork).** Read how the build-arg `VLLM_REF` is used in the Dockerfile:
```bash
grep -rniE "VLLM_REF|git clone|git fetch|git checkout|github.com/.*vllm|ARG VLLM_REF" /mnt/tank/src/github/spark-vllm-docker/*.Dockerfile /mnt/tank/src/github/spark-vllm-docker/Dockerfile* 2>/dev/null | head -20
```
Gate: identify whether the Dockerfile clones `vllm-project/vllm` (upstream) or accepts a full remote. **A bare `DCP_REF` from the fork will NOT be reachable from upstream.** Record the resolution path — one of:
  (a) the Dockerfile already `git fetch`es an arbitrary ref → pass `DCP_REF` directly;
  (b) it clones upstream only → we must add a build step / build-arg to `git remote add fork https://github.com/local-inference-lab/vllm && git fetch fork <branch> && git checkout FETCH_HEAD` (Task 2 stages this);
  (c) `build-and-copy.sh` has an `--apply-vllm-pr`/`VLLM_PRS` path that can carry the DCP diff.
**Record the chosen path as `SOURCE_PLAN`.**

- [ ] **Step 3: Resolve CUDA base — cu130 (current) vs cu132.** Our `b12x:probe` is cu130; the community DCP images are cu132. Check whether the DCP branch requires cu132 and whether an arm64 wheel exists:
```bash
# what CUDA does the build base use today?
grep -riE "cuda|cu13[0-9]|nvidia/cuda|FROM " /mnt/tank/src/github/spark-vllm-docker/Dockerfile* 2>/dev/null | grep -iE "cuda|FROM" | head
# does a torch cu132 aarch64 wheel exist (the recurring GB10 blocker)?
gh api repos/local-inference-lab/vllm/contents/requirements/cuda.txt?ref=$DCP_REF --jq '.content' 2>/dev/null | base64 -d 2>/dev/null | grep -iE "torch|cuda" | head
pip index versions torch --index-url https://download.pytorch.org/whl/cu132 2>&1 | head -3 || echo "cu132 index probe (may need the build node)"
```
Gate: decide **cu130** (reuse our base, lower risk) if the DCP branch works on it, else **cu132** only if a torch+cu132 **aarch64** wheel is confirmed available. **Record as `CUDA_BASE`.** If cu132 is required but no arm64 wheel exists → **STOP and escalate** (this is the known GB10 blocker).

- [ ] **Step 4: Record the go/no-go.** Report: `DCP_REF`, `SOURCE_PLAN` (a/b/c + exact commands), `CUDA_BASE`, and a clear **GO** or **NO-GO (reason)**. Only GO proceeds to Task 2.

---

### Task 2: Stage the DCP build inputs

**Goal:** Put the DCP vLLM ref, PR#72 patches, and current CosmicRaisins kernels where the build can consume them, per `SOURCE_PLAN` from Task 1.

**Files:**
- Stage: `/mnt/tank/src/glm52-overlay-dcp/` (copy of `/mnt/tank/src/glm52-overlay/`, kernels refreshed)
- Stage: PR#72 patches under `/mnt/tank/src/spark-vllm-docker-dcp-patches/`

- [ ] **Step 1: Refresh the CosmicRaisins kernels (DCP-stack revision) into a new overlay context.**
```bash
cp -r /mnt/tank/src/glm52-overlay /mnt/tank/src/glm52-overlay-dcp
cd /mnt/tank/src/glm52-overlay-dcp/kernels
for f in sparse_mla_kernels.py sparse_mla_env.py sm12x_sparse_mla_attn.py patch_flashmla_ops.py flashmla_sparse.py sm12x_deep_gemm_fallbacks.py sm12x_mqa.py b12x_sparse_helpers.py sparse_attn_indexer.py deepseek_v2.py; do
  gh api "repos/CosmicRaisins/glm-5.2-gb10/contents/kernels/$f" --jq '.content' 2>/dev/null | base64 -d > "$f" && echo "refreshed $f"
done
```
Gate: all 10 files refreshed, non-empty (`wc -l *.py` all > 0).

- [ ] **Step 2: Fetch the PR#72 patches.** From the DCP branch / its PR (source per Task 1 — the CosmicRaisins repo vendors them, or the PR on `local-inference-lab/vllm`):
```bash
mkdir -p /mnt/tank/src/spark-vllm-docker-dcp-patches
# adjust the source to whatever Task 1 found; the CosmicRaisins repo references these by name
for p in pr72-1-draft-dcp-config-propagation.patch pr72-2-glm-dcp-draft-path.patch; do
  gh api "repos/CosmicRaisins/glm-5.2-gb10/contents/patches/$p" --jq '.content' 2>/dev/null | base64 -d > "/mnt/tank/src/spark-vllm-docker-dcp-patches/$p" && echo "got $p"
done
ls -la /mnt/tank/src/spark-vllm-docker-dcp-patches/
```
Gate: both patch files present and non-empty. If the CosmicRaisins path differs, locate them from Task 1's findings (the repo `CHANGES.md` names them). **Do not fabricate** — if not found, record as a blocker.

- [ ] **Step 3: Prepare the vLLM source per `SOURCE_PLAN`.** If path (b) (add fork remote), record the exact build hook — e.g. a wrapper that, before the wheel build, does `git remote add licvllm https://github.com/local-inference-lab/vllm && git fetch licvllm && git checkout $DCP_REF && git apply /…/pr72-1….patch /…/pr72-2….patch`. Verify the patches apply cleanly to `$DCP_REF` with `git apply --check`:
```bash
# on the build node, in a throwaway clone:
git -C /tmp/vllm-dcp-check apply --check /mnt/tank/src/spark-vllm-docker-dcp-patches/pr72-1-draft-dcp-config-propagation.patch && echo "pr72-1 applies" || echo "pr72-1 CONFLICT"
```
Gate: both patches `--check` clean against `$DCP_REF` (or record the conflict for a rebase). Commit the staged patches + overlay note to git (`docs/glm-5.2-dcp-build-notes.md` capturing `DCP_REF`, `SOURCE_PLAN`, `CUDA_BASE`).

---

### Task 3: Build the DCP base image (arm64, ~1 h)

**Goal:** Produce `vllm-node-tf5-glm52-b12x-dcp:base` from the DCP ref. Runs on `.38`.

**Files:** none (produces a Docker image).

- [ ] **Step 1: Free the cluster — stop the 144K daily driver** (the build needs the node's resources):
```bash
CUR=$(curl -s http://localhost:4000/api/deployments | python3 -c "import sys,json;print(next((x['id'] for x in json.load(sys.stdin) if x.get('status') not in ('stopped','failed')), ''))")
[ -n "$CUR" ] && curl -s -X DELETE "http://localhost:4000/api/deployments/$CUR" >/dev/null && echo "stopped $CUR"
```
Gate: `docker ps` on all 4 nodes shows no `glm52/b12x` container (wait for teardown; ~30 s).

- [ ] **Step 2: Build the base on `.38`** with the DCP ref (apply the PR#35568 curl-retry + tolerant-apply fixes from `docs/glm-5.2-custom-image-build.md` if the Dockerfile still uses that step). Use `CUDA_BASE`/`SOURCE_PLAN` from Task 1:
```bash
ssh daniel@192.168.44.38 'cd /mnt/tank/src/github/spark-vllm-docker && \
  ./build-and-copy.sh --vllm-ref '"$DCP_REF"' -t vllm-node-tf5-glm52-b12x-dcp:base --tf5' 2>&1 | tail -40
```
(If `SOURCE_PLAN` is (b): first apply the fork-remote hook per Task 2 Step 3, or pass the fork via whatever mechanism Task 1 identified.)
Gate: build exits 0; verify torch is a real CUDA build, not `+cpu`:
```bash
ssh daniel@192.168.44.38 'docker run --rm --gpus all vllm-node-tf5-glm52-b12x-dcp:base python3 -c "import torch;print(torch.__version__, torch.cuda.is_available())"'
```
Expected: a `+cu13x` version and `True`. If `+cpu` or `False` → **STOP** (build is broken).

---

### Task 4: Overlay kernels + b12x + DCP draft path → `…-dcp:probe`

**Goal:** Bake the DSA kernels + b12x + deep_gemm re-bind (and any PR#72 runtime bits not in the wheel) onto the base.

**Files:**
- Modify: `/mnt/tank/src/glm52-overlay-dcp/Dockerfile.glm52-overlay` (base tag → `…-dcp:base`, output → `…-dcp:probe`)

- [ ] **Step 1: Point the overlay Dockerfile at the DCP base.** Edit `/mnt/tank/src/glm52-overlay-dcp/Dockerfile.glm52-overlay` line 1: `FROM vllm-node-tf5-glm52-b12x-dcp:base`. Keep the `COPY kernels/…` block + `pip install --no-deps b12x==0.23.0` + `patch_deep_gemm.py` step verbatim from `docs/glm-5.2-custom-image-build.md` Step 2.

- [ ] **Step 2: Build the overlay on `.38`:**
```bash
ssh daniel@192.168.44.38 'cd /mnt/tank/src/glm52-overlay-dcp && docker build -f Dockerfile.glm52-overlay -t vllm-node-tf5-glm52-b12x-dcp:probe .' 2>&1 | tail -20
```
Gate: build exits 0; the deep_gemm re-bind `ast.parse … print("OK")` line prints `OK`.

- [ ] **Step 3: Smoke-test imports + b12x + NCCL in the image:**
```bash
ssh daniel@192.168.44.38 'docker run --rm --gpus all vllm-node-tf5-glm52-b12x-dcp:probe python3 -c "import torch,b12x,vllm; print(\"OK\", torch.cuda.nccl.version(), b12x.__version__ if hasattr(b12x,\"__version__\") else \"b12x-loaded\")"'
```
Gate: prints `OK …` (torch + b12x + vllm import cleanly). If b12x import fails → the master pin < `80eb49b`; rebuild with the correct b12x.

---

### Task 5: Distribute the image to all 4 nodes

**Goal:** `…-dcp:probe` present on `.36/.37/.38/.39`.

- [ ] **Step 1: Copy the image** (use `build-and-copy.sh --no-build --copy-to`, or `docker save | ssh docker load`):
```bash
ssh daniel@192.168.44.38 'cd /mnt/tank/src/github/spark-vllm-docker && ./build-and-copy.sh --no-build --copy-to 192.168.44.36,192.168.44.37,192.168.44.39 -t vllm-node-tf5-glm52-b12x-dcp:probe' 2>&1 | tail -15
```
Gate: each of the 4 nodes lists the image:
```bash
for ip in 36 37 38 39; do echo -n ".$ip: "; ssh daniel@192.168.44.$ip 'docker images -q vllm-node-tf5-glm52-b12x-dcp:probe | head -1'; done
```
Expected: a non-empty image id on all 4.

---

### Task 6: Write the DCP2 `@dgxrun` recipe

**Files:**
- Create: `recipes/dgxrun/glm-5.2-awq-15pct-dcp2.yaml`

- [ ] **Step 1: Create the recipe** — copy `recipes/dgxrun/glm-5.2-awq-15pct-144k.yaml` and change: `container: vllm-node-tf5-glm52-b12x-dcp:probe`; `max_model_len: 262144`; add env `VLLM_USE_V2_MODEL_RUNNER: "1"`; append `--decode-context-parallel-size 2` to the `command:` serve line (before `--max-model-len`). Keep everything else (reduced chunk env, NCCL/RDMA stanza, MTP speculative-config, `fp8_ds_mla`, `--enable-prefix-caching`, `cudagraph_mode: FULL`, gmu 0.88). Header comment: DCP2 → ~262K, new image, fallback = `144k` recipe.
Gate: `grep` confirms `decode-context-parallel-size 2`, `VLLM_USE_V2_MODEL_RUNNER`, the new container tag, and `max_model_len: 262144` are all present.

- [ ] **Step 2: Commit the recipe** — `git add recipes/dgxrun/glm-5.2-awq-15pct-dcp2.yaml && git commit -m "feat(dgxrun-dcp): GLM-5.2 15pct + DCP2 262K recipe (new dcp image)"`.

---

### Task 7: Deploy 15pct + DCP2 and verify it serves at ≥262K

**Goal:** The DCP deploy comes up and answers at 262K.

- [ ] **Step 1: Deploy via inline recipeYaml** (4 nodes head-first `.38`), with the VRAM-free wait:
```bash
python3 - <<'PY'
import json,urllib.request,urllib.error,time
N={"36":"cmno92dip006j36o3h3yo91p7","37":"cmno92lcz006s36o3k3yijvbp","38":"cmno92u96007236o3axqbpskv","39":"cmr6eqr2200lj2auha8x7us2p"}
r=open("recipes/dgxrun/glm-5.2-awq-15pct-dcp2.yaml").read()
b=json.dumps({"recipeYaml":r,"nodeIds":[N["38"],N["37"],N["39"],N["36"]],"displayName":"glm-5.2"}).encode()
for _ in range(12):
  try:
    d=json.load(urllib.request.urlopen(urllib.request.Request("http://localhost:4000/api/deployments",data=b,headers={"Content-Type":"application/json"},method="POST"),timeout=30)); print("DEPLOY",d.get("id"),d.get("status")); break
  except urllib.error.HTTPError as e:
    if e.code==409 and "VRAM" in e.read().decode(): time.sleep(15); continue
    raise
PY
```
Gate: deployment reaches `running` (poll `/api/deployments`). If it `failed` — grab the head log (`ssh .38 'docker logs --tail 60 <cid>'`) and check for the DCP-draft `topk_scores_buffer` crash (⇒ PR#72 not applied) or a KV/OOM error (⇒ 262K too high for DCP2 at gmu 0.88; retry at 229376 = 224K).

- [ ] **Step 2: Confirm it answers at high context** — warmup generation + confirm the served max len:
```bash
curl -s http://192.168.44.38:8000/v1/models | python3 -c 'import sys,json;m=json.load(sys.stdin)["data"][0];print("max_model_len:",m.get("max_model_len"))'
curl -s http://192.168.44.38:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"glm-5.2","messages":[{"role":"user","content":"Reply: READY"}],"max_tokens":16}' | python3 -c 'import sys,json;print("gen usage:",json.load(sys.stdin)["usage"])'
```
Gate: `max_model_len` ≈ 262144 and generation returns tokens.

---

### Task 8: Validate — long-context retrieval + decode + prefill

**Goal:** Prove it's actually usable at long context, not just serving.

- [ ] **Step 1: Needle-in-haystack at ~200K.** Send a ~200K-token prompt with a planted fact near the start and a question at the end; confirm the model retrieves it (DCP KV-sharding must not corrupt long-range retrieval):
```bash
python3 - <<'PY'
import urllib.request,json
filler=("The project's build system is Bazel. "*20000)  # ~200K tokens of filler
needle="IMPORTANT: the deploy secret code is PURPLE-OTTER-1979.\n"
prompt=needle+filler+"\nWhat is the deploy secret code? Answer with just the code."
b=json.dumps({"model":"glm-5.2","messages":[{"role":"user","content":prompt}],"max_tokens":64,"temperature":0}).encode()
r=json.load(urllib.request.urlopen(urllib.request.Request("http://192.168.44.38:8000/v1/chat/completions",data=b,headers={"Content-Type":"application/json"},method="POST"),timeout=1200))
print("prompt_tokens:",r["usage"]["prompt_tokens"],"| answer:",r["choices"][0]["message"]["content"])
PY
```
Gate: `prompt_tokens` > 150K **and** the answer contains `PURPLE-OTTER-1979`. If it can't retrieve → DCP sharding/config is wrong; do not ship.

- [ ] **Step 2: Decode on code content + MTP acceptance.** Read the live `SpecDecoding metrics` from the head during a code-generation request:
```bash
ssh daniel@192.168.44.38 'CID=$(docker ps --format "{{.ID}} {{.Image}}"|grep -i dcp|awk "{print \$1}"|head -1); docker logs --since 2m "$CID" 2>&1 | grep "SpecDecoding metrics" | tail -4'
```
Gate: acceptance length in the ~2.6–3.3 range (decode ~25–31 on code) — i.e. DCP did not regress decode.

- [ ] **Step 3: Prefill TTFT with prefix caching.** Send the same 200K prompt twice; confirm the second call's TTFT is far lower (prefix cache hit — the property that makes big context interactive):
Gate: 2nd-call TTFT ≪ 1st-call TTFT (order-of-magnitude), confirming prefix caching composes with DCP.

---

### Task 9: Finalize

- [ ] **Step 1: Document the build** — write `docs/glm-5.2-dcp-build-notes.md` capturing the resolved `DCP_REF`, `SOURCE_PLAN`, `CUDA_BASE`, the working `max_model_len`, and the validation numbers (needle pass, decode tok/s, TTFT). Commit.
- [ ] **Step 2: Rollback note** — confirm `b12x:probe` + `@dgxrun/glm-5.2-awq-15pct-144k` remain untouched (the fallback). Both recipes now in the catalog: `144k` (no-DCP, ~159K, fast prefill) and `dcp2` (~262K, DCP).
- [ ] **Step 3: Decide steady state** — either leave `dcp2` deployed as the daily driver, or stop it and redeploy `144k` until routinely needed. Record the choice.
- [ ] **Step 4: Commit** — `git add docs/glm-5.2-dcp-build-notes.md && git commit -m "docs: GLM-5.2 DCP build notes + validation results"`.

---

## Self-review (author checklist — completed)

- **Spec coverage:** DCP-stack pins (T1–T4) ✓; new image tag / don't-overwrite (constraints + T3/T4) ✓; V2 runner + `--decode-context-parallel-size 2` recipe (T6) ✓; 15pct-first no-download (T7) ✓; distribute to 4 nodes (T5) ✓; stop-144K-during-build (T3 S1, T7) ✓; NCCL preload + reduced-chunk env carried over (T6) ✓; long-context validation + decode + prefill (T8) ✓; rollback (T9) ✓.
- **Gate front-loaded:** Task 1 resolves the arm64/fork/CUDA unknowns and can STOP the plan before the ~1 h build — the single highest-risk thing, as required.
- **No fabricated pins:** the two genuine unknowns (fork-source mechanism, cu130-vs-cu132) are Task 1's *deliverables*, and later tasks explicitly consume `DCP_REF`/`SOURCE_PLAN`/`CUDA_BASE` rather than hard-coding guesses. The PR#72 patch fetch is guarded ("do not fabricate — record as blocker if not found").
- **Consistency:** image tags (`…-dcp:base` → `…-dcp:probe`), the recipe filename `glm-5.2-awq-15pct-dcp2.yaml`, and `max_model_len 262144` (with a documented 224K fallback) are consistent across T3–T8.
- **Honest scope:** this is a real arm64 CUDA build with a genuine "may not build" risk — Task 1 exists precisely so we don't sink an hour before knowing.
