# DRAFT — vLLM upstream issue: silent post-load hang at TP=4 on Qwen3.5/3.6 family, DGX Spark cluster (sm_121 / GB10)

> **Status: DRAFT for review before filing at https://github.com/vllm-project/vllm/issues/new.**
> Written 2026-05-04 from an actual reproduction on a 4× DGX Spark cluster.
> Audience: vLLM maintainers + other DGX Spark cluster users hitting the same wall.

---

## Title (suggested)

`[Bug]: Silent post-load hang at TP=4 on Qwen3.5/3.6 family across DGX Spark cluster (Blackwell consumer / sm_121)`

---

## Your current environment

```
vLLM:                 0.20.1rc1.dev152+gc3ad791e1.d20260502.cu132 (pinned wheel)
Hardware:             4× NVIDIA DGX Spark (GB10, Blackwell consumer, sm_121)
                      122 GB unified memory per node
Interconnect:         per-node 100/200 GbE fast fabric (192.168.100.0/24)
                      plus 10 GbE management (192.168.44.0/24)
NVIDIA driver:        580.142 (per NVIDIA forum recommendation; 590.x has UMA leak on GB10)
nvidia-container-toolkit: 1.19.0-1
Docker:               29.2.1
DGX release:          7.5.0  (identical on all 4 nodes)
Kernel:               6.17.0-1014-nvidia (identical on all 4 nodes)
OS:                   Ubuntu 24.04.4 LTS (identical on all 4 nodes)
Container:            built from spark-vllm-docker (eugr fork), `vllm-node` image,
                      installs the pinned vllm wheel listed above
Runtime:              Ray 2.55.x distributed-executor-backend, fastsafetensors=NOT used
                      (see "what we ruled out" below)
```

---

## 🐛 Describe the bug

Deploying any Qwen3.5 or Qwen3.6 family model with `--tensor-parallel-size 4` across
4 DGX Spark nodes consistently hangs **silently** after weight load completes, with
no error, no exception, no NCCL timeout — the workers spin at 89–159% CPU while the
HTTP API server never binds.

**Same symptom across:**
- `Qwen/Qwen3.6-27B-FP8` (dense, multimodal, 27B params)
- `Qwen/Qwen3.5-397B-A17B-FP8` (MoE, 397B/17B active)
- `Intel/Qwen3.5-397B-A17B-int4-AutoRound` with the Marlin TP=4 patch from
  [#35924](https://github.com/vllm-project/vllm/issues/35924) applied (which gets
  past Marlin selection cleanly)

**TP=1 and TP=2 work perfectly on the same hardware/software**, so the problem is
in the cluster coordination path that engages at TP≥3.

---

## Reproduction

Minimal — using the smallest model that triggers the hang:

```bash
# On the head node, with 4 DGX Sparks reachable as Ray workers:
vllm serve Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.85 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --enable-prefix-caching \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --distributed-executor-backend ray
```

Cluster: head + 3 Ray workers, each on a separate DGX Spark, joined via Ray
(`ray start --head` + `ray start --address …`). All four nodes share an NFS
mount for the model cache, all four have identical software versions (verified
— see "what we ruled out" below).

**Outcome:** weights load successfully (66/66 shards in ~43 s on warm NFS),
kernel selection logs `Using CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod`
on rank 0, then **all log output stops**. The API server never reaches
`Started server process`. The deploy stays in this state indefinitely (verified
across 7+ hour soaks).

---

## Last log lines before the hang

```
(EngineCore pid=947) (RayWorkerWrapper pid=1206) INFO 16:30:09 [gpu_model_runner.py:4720]
    Starting to load model Qwen/Qwen3.6-27B-FP8...
(EngineCore pid=947) (RayWorkerWrapper pid=1206) INFO 16:30:09 [cuda.py:390]
    Using backend AttentionBackendEnum.FLASH_ATTN for vit attention
(EngineCore pid=947) (RayWorkerWrapper pid=1206) INFO 16:30:09 [mm_encoder_attention.py:230]
    Using AttentionBackendEnum.FLASH_ATTN for MMEncoderAttention.
(EngineCore pid=947) (RayWorkerWrapper pid=1206) INFO 16:30:09 [qwen3_next.py:202]
    Using Triton/FLA GDN prefill kernel
(EngineCore pid=947) (RayWorkerWrapper pid=1206) INFO 16:30:09 [cuda.py:274]
    Using AttentionBackendEnum.FLASHINFER backend.
(EngineCore pid=947) (RayWorkerWrapper pid=1206) INFO 16:30:09 [__init__.py:261]
    Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod
Loading safetensors checkpoint shards: 100% Completed | 66/66 [00:43<00:00,  1.52it/s]
<silence — no further log output for 7+ hours>
```

After this point: 4× `RayWorkerWrapper.execute_method` processes consume 89–159% CPU
each. No NCCL timeout, no exception, no stack trace. The only way out is to kill
the deploy.

---

## What we ruled out

We did not file this issue lightly — we eliminated the obvious suspects first.

### 1. Node parity (per-node hardware/software drift)

All four DGX Sparks are byte-identical at the OS / driver / DGX-package level:

| Component | spark-01 | spark-02 | spark-03 | spark-04 |
|---|---|---|---|---|
| OS | Ubuntu 24.04.4 | Ubuntu 24.04.4 | Ubuntu 24.04.4 | Ubuntu 24.04.4 |
| Kernel | 6.17.0-1014-nvidia | 6.17.0-1014-nvidia | 6.17.0-1014-nvidia | 6.17.0-1014-nvidia |
| NVIDIA driver | 580.142 | 580.142 | 580.142 | 580.142 |
| nvidia-container-toolkit | 1.19.0-1 | 1.19.0-1 | 1.19.0-1 | 1.19.0-1 |
| Docker | 29.2.1 | 29.2.1 | 29.2.1 | 29.2.1 |
| `dgx-release` | 7.5.0 | 7.5.0 | 7.5.0 | 7.5.0 |

Solo deploy of the same `Qwen/Qwen3.6-27B-FP8` recipe at TP=1 produces identical
performance and accuracy on every node:

| Node | Wall (s, 100 SQL examples, c=4) | Tokens generated | Aggregate TPS | Accuracy |
|---|---:|---:|---:|---:|
| spark-01 (`192.168.44.36`) | 4048 | 121,522 | 30.1 | 48% |
| spark-04 (`192.168.44.39`) | 4112 | 122,089 | 29.6 | 48% |

So the hang is **not** caused by any specific node being flaky.

### 2. fastsafetensors NCCL collective during weight load

Our first TP=4 attempt used `--load-format fastsafetensors` and **did** hit a
clean failure (NCCL broadcast error in `fastsafetensors.frameworks._torch.py:103`
on rank 3 during weight load — different node each retry). Removing
`--load-format fastsafetensors` and falling back to the default safetensors
loader: weight load completes successfully on all 4 ranks, then the silent hang
begins. **So the hang is downstream of weight loading**, not in the load itself.

### 3. CUDA graph capture

We tested with `--enforce-eager` on the larger 397B run (same model class, same
TP=4 setup). The hang was unchanged. So it is not the standard "CUDA-graph
capture deadlock at multi-node" failure mode.

### 4. Marlin TP=4 quantization issue

The 397B int4 attempt also tested the upstream
[#35924](https://github.com/vllm-project/vllm/issues/35924) workaround (port to
the post-refactor `gdn_linear_attn.py` location, see our patch in
[mods/fix-qwen35-tp4-marlin](https://github.com/kreuzhofer/spark-vllm-docker/tree/main/mods/fix-qwen35-tp4-marlin)).
With the Marlin patch applied, the int4 deploy **gets past Marlin selection
cleanly** (`Using MarlinLinearKernel for GPTQMarlinLinearMethod` on all 4 ranks,
zero `not found in params_dict` warnings) and then exhibits **the exact same
silent post-load hang**. So this hang is independent of the Marlin issue — it
applies equally to FP8 (no Marlin involved at all) and to int4 with the Marlin
fix.

### 5. Pre-existing TP=4 issues

We searched for prior reports. Closest matches:

- [#34893](https://github.com/vllm-project/vllm/issues/34893) — Qwen3.5-397B-FP8
  TP=4 fused-linear sharding. **Closed Feb 2026 with a fix that should be in our
  build.** Symptom was a clean error during weight load (RuntimeError); ours is
  silent post-load.
- [#34948](https://github.com/vllm-project/vllm/issues/34948) — Qwen3.5 CUDA
  illegal memory access in GDN kernel. Reproduces during generation, not init.
- [#39774](https://github.com/vllm-project/vllm/issues/39774) — Qwen3.5 NCCL
  "unhandled system error" at TP>1. Different error, different stage.
- [#36821](https://github.com/vllm-project/vllm/issues/36821) — sm_121 build
  support gap on aarch64. Adjacent but separate.

None of these report **silent post-load hang** as the symptom.

---

## Side observation: TP=4 is essentially uncharted in the DGX Spark community

The published cluster recipes and benchmarks all top out at TP=2:

| Source | TP support |
|---|---|
| Official vLLM Qwen3.5/3.6 recipe | single-node only |
| NVIDIA DGX Spark playbook (Ray cluster section) | TP=2 |
| [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (the recipe runner used in this report) | "Verified end-to-end on 1 and 2 Sparks, with n>2 code paths reviewed but not yet exercised on real hardware" (per project README) |
| csabakecskemeti DGX Spark community playbooks | TP=2 |
| bkrabach/dgx-spark-cluster | "dual-node" by design |
| mark-ramsey-ri/vllm-dgx-spark | "one or two" Sparks |

The only **TP=4 successes** for Qwen3.5/3.6 family on DGX Spark we could find use
**SGLang**, not vLLM:

- [Forum thread — 397B-int4-AutoRound on 4× Sparks via SGLang](https://forums.developer.nvidia.com/t/qwen3-5-397b-a17b-int4-autoround-4-x-db10-node-test/362368)
  reporting 37 tok/s single-user, 94 tok/s @ 4 concurrent.
- [Forum thread — 397B-NVFP4 on 4× Sparks via scitrera/dgx-spark-sglang](https://forums.developer.nvidia.com/t/two-multi-node-dgx-spark-wins-roce-2x-inference-throughput-qwen3-5-397b-a17b-nvfp4-serving-with-sm121-cutlass-patch/366325)
  reporting 22-101 tok/s with an SM121 CUTLASS patch.

So as of writing, **no public report exists of vLLM successfully serving a
Qwen3.5/3.6 family model at TP=4 on DGX Spark.** This issue is the first
reproducible failure report; that should make it easier to investigate (rules
out "it works for someone else, you must be holding it wrong").

---

## What we have NOT yet tried (next-step diagnostics)

If the maintainer wants more data, we can run any of these on the live hang:

- `--mm-encoder-tp-mode data` — both 27B and 397B are multimodal; the mm encoder
  defaults to column-parallel sharding which has known issues at higher TP. The
  proven SGLang TP=4 deploys above use this kind of flag.
- `VLLM_LOGGING_LEVEL=DEBUG`, `NCCL_DEBUG=TRACE`, `CUDA_LAUNCH_BLOCKING=1` to see
  whether the workers are stuck in a specific NCCL collective.
- `py-spy dump --pid <RayWorkerWrapper PID>` on each rank during the hang to get
  the actual Python stack of every spinning worker.
- Tighter test matrix: TP=2 with **just spark-03 + spark-04** (vs the
  successful spark-01 + spark-02) to confirm the working TP=2 isn't a node-pair
  artifact. (We already confirmed solo-on-each-node parity but not pair-by-pair.)

We did not include the above in this initial report to keep it tight; happy to
gather them on request.

---

## What would be most actionable for vLLM

1. Confirm whether anyone in the vLLM team has tested TP=4 on Blackwell consumer
   (sm_121) hardware at all — i.e., is this known-broken or just untested?
2. If untested, treat it as a feature-gap report for `sm_121 + multi-node TP > 2`
   in the same vein as [#36821](https://github.com/vllm-project/vllm/issues/36821)
   (sm_121 build support gap).
3. If anyone else has hit this and worked around it (e.g., via a specific flag
   combination not yet documented), that's the highest-leverage answer.

---

## Filing checklist (before posting)

- [ ] Re-run the 27B FP8 TP=4 deploy with `VLLM_LOGGING_LEVEL=DEBUG` +
      `NCCL_DEBUG=TRACE` and capture the LAST 200 lines of log before silence.
- [ ] `py-spy dump --pid <pid>` on each of the 4 worker processes during the
      hang and include the four stack traces.
- [ ] Run `--mm-encoder-tp-mode data` once and report whether it gets past the
      hang (the most plausible single-flag fix per current evidence).
- [ ] Confirm we can reproduce the same hang with `vllm/vllm-openai:cu130-nightly`
      (the official Blackwell-recommended container) — not just our pinned
      wheel — to rule out anything spark-vllm-docker-specific.
- [ ] Strip the issue body of internal IPs, anonymize anything sensitive.
- [ ] Add `sm_121` and `multi-node` labels (vLLM repo label vocabulary).

Once the four items above are gathered, post to
https://github.com/vllm-project/vllm/issues/new with this draft as the body.
