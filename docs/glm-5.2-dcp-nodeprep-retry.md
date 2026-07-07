# GLM-5.2 DCP capture-OOM — node-prep retry spec (2026-07-07)

## Problem, restated with the root cause found
The DCP stack builds and **fully initializes** (B12X_MLA_SPARSE + DCP2 + all 4 workers + KV cache), then OOMs at ~116/124 GB **during CUDA-graph capture / first-run JIT** — for both the pruned (0.90/320K) and unpruned (0.89/256K) models — wedging the nodes to ping-only (power-cycle needed).

**It is NOT a recipe/tuning problem.** Our serve config already matches the community reference (cudagraph FULL, `max_cudagraph_capture_size` 10, `fp8_ds_mla`, B12X_MLA_SPARSE, `--max-num-seqs 1`, `--max-num-batched-tokens 4096`, vLLM `e232d26` + b12x). The community runs the **same or higher gmu (0.90–0.93)** and succeeds. The gap is **node-prep** (a unified-memory page-cache + swap problem), per the research (memory `glm52-dcp-stack-build`; sources in the build notes).

Root cause chain (GB10 unified memory — weights + KV + CUDA graphs + Linux page cache all share one ~124 GB pool):
1. Loading the checkpoint fills the **page cache** (vLLM auto-prefetches NFS checkpoints into page cache → **NFS makes this worse than local NVMe**). vLLM's own startup guard "refuses at 0.90 gmu" if caches aren't dropped.
2. The **first-run FlashInfer/Triton JIT** spikes **+20–30 GB** (6 parallel `cicc` @ 1.5–6 GB each) on top of the already-full pool → 116/124 → OOM.
3. **Swap** turns the OOM into a whole-node freeze (driver alloc fails below the CUDA runtime; kernel keeps growing page cache) → the wedge.

## The fix — node-prep + timeout, keep cudagraph FULL (for the tps)

### A. Persistent per-node prep (once; via `scripts/dgx-node-prep.sh`)
Run on all 4 nodes; persist via `/etc/sysctl.d/` + systemd:
- `vm.min_free_kbytes = 5242880` (5 GiB the page cache can't consume → guaranteed capture headroom)
- `vm.vfs_cache_pressure = 200`, `vm.dirty_ratio = 5`, `vm.dirty_background_ratio = 2`
- **`swapoff -a` + `vm.swappiness = 1`** — converts the freeze into a recoverable OOM-kill (no more power-cycles)
- **`earlyoom`** (`-m 2 -s 100 --prefer 'vllm|python3|python'`) — keeps a headless node reachable if OOM does hit
- `systemctl set-default multi-user.target` — disable desktop, frees 2–3 GB
- (Triton/Inductor cache already persisted via `TRITON_CACHE_DIR=/cache/huggingface/.tritoncache-dcp` — keep it so the JIT spike is a one-time cold-boot cost)
- NOTE: cgroup/systemd `MemoryMax` does **not** bound GPU memory on UMA — do not rely on it.

### B. Per-launch: drop caches right before deploy (the community's #1 step)
- `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` on **every node immediately before the deploy** — and ideally a background loop (`while true; do echo 3 > …/drop_caches; sleep 0.1; done`) **during weight load**, killed once serving.
- Integrate into the dgxrun launch path (agent runs drop_caches before `docker run`, spawns the loop, reaps it on ready) OR a manager pre-deploy hook. Simplest first pass: run drop_caches on all nodes via the API/exec immediately before POSTing the deploy.

### C. Readiness timeout — NO CHANGE NEEDED (verified 2026-07-07)
Originally suspected a premature readiness deadline. Reading the agent code (`packages/agent/src/index.ts:413-423`): a dgxrun deploy is marked `failed` **only when the container dies** (`!containerRunning && !status.alive`); a still-capturing but *alive* head container just stays `"starting"` with no deadline. So the failure was **100% the OOM killing the container**, not a timeout — once node-prep prevents the OOM, the code already waits out the full ~10-min capture. (Keep the monitor's poll window ≥28 min to cover 14 min load + ~10 min capture.)

### D. Recipe — REVERT the tuning band-aids
- gmu **0.90** (revert 0.89 → 0.90; community value), keep `cudagraph_mode: FULL`, `--max-num-seqs 1`, `--max-num-batched-tokens 4096`. Do NOT go eager (kills the tps) or lower gmu — those don't address the page cache.

## Retry order (after a power-cycle)
1. **Node-prep**: run `scripts/dgx-node-prep.sh` on all 4 nodes; verify sysctls + `swapon --show` empty + `MemAvailable` in `/proc/meminfo` (monitor this, NOT `nvidia-smi` — it can't read GB10 memory).
2. **Pruned 15pct + separate MTP + DCP2 @ 0.90 / 256K first** (the fast daily driver; MTP works on the pruned; unpruned MTP self-draft is a separate KeyError). Drop caches on all nodes, then deploy — API-only monitor with a ≥28-min poll window (the agent waits out the ~10-min capture on its own; no timeout, see §C).
3. **If it serves** → validate (needle retrieval + decode/MTP acceptance on code), then climb toward 320K. This is the DCP daily driver.
4. **If it STILL OOMs after drop_caches** — diagnose by error string:
   - plain `NV_ERR_NO_MEMORY` / pool exhaustion during capture → page cache still winning → add the drop_caches *loop during load*, and/or **stage weights to local per-node NVMe** (`$HOME/.cache/huggingface`, `shared_weights_nfs: false`) — the definitive NFS fix.
   - `cudaErrorStreamCaptureInvalidated` → the b12x capture-safe decode path isn't active (would mean decode falls back to a Triton kernel that allocates under capture); verify `VLLM_USE_B12X_SPARSE_INDEXER=1` + `VLLM_USE_V2_MODEL_RUNNER=1` + the `index_topk_pattern` override are all live, else fall back to `cudagraph_mode: PIECEWISE`.
5. **Unpruned + DCP2 (no MTP)** as the quality baseline once the pruned serves.

## Success gate
Pruned 15pct + DCP2 serves at ≥256K with `cudagraph FULL`, decode ~25–31 on code (MTP acceptance in the vLLM logs), and — critically — an OOM (if any) is now a **clean kill, not a node freeze** (swapoff + earlyoom). No power-cycle required to recover.

## What we deliberately are NOT doing
Lowering gmu below 0.90, going enforce-eager, or bumping `--shm-size` (a cap not a reservation; not our issue). The recipe was right; the plumbing (NFS + no drop_caches + swap + short timeout) was the gap.
