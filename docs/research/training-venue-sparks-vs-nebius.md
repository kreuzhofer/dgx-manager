# Train Qwen3.8-27B on the DGX Sparks, or on Nebius?

> Research for [#49](https://github.com/kreuzhofer/dgx-manager/issues/49). Part of the
> [Qwen3.8-27B wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/45).
>
> **Nebius pricing checked 2026-08-30, 15:24–15:32 UTC**, against `nebius.com/prices` and
> `docs.nebius.com` directly. Prices move; re-check before committing money.
> Every number below is labelled **measured**, **derived** or **extrapolated** — the
> distinction matters here, because the headline objection in the issue turns out to rest on
> an extrapolation that does not hold.

---

## Recommendation

**Train on the Sparks.** Keep Nebius as a pre-costed escape hatch with an explicit trigger.

Three findings drive this, in order of weight:

1. **The premise is an artefact.** "Multi-node ZeRO-3 on the Sparks is ~4× the wall time" was
   measured at `max_seq_length=256`. ZeRO-3's communication volume per optimizer step is
   *independent of sequence length* — it is a function of model size and gradient-accumulation
   depth only. Scale the sequence to 16k and the same ~9.5 s/step of NCCL traffic goes from
   ~63% of step time to **~3%**. The 4× penalty does not survive the change that prompted the
   question.

   *This is not just arithmetic on paper.* The model predicts the measured overhead: 2 ranks ×
   `grad_accum=4` gather 207 GiB per optimizer step, which at the measured ~25 GB/s dual-rail
   RoCE bandwidth is **8.9 s/step** — against a **measured ~9.5 s/step** of multi-node overhead
   in the 3.6-27B run (15 s/step dual-rail vs 5.5 s single-node). The agreement is close enough
   to trust the sequence-independence claim, which is the load-bearing part of the argument.
2. **For LoRA, ZeRO-3 is the wrong tool entirely.** Qwen3.8-27B is 51.7 GiB at BF16 and a GB10
   has 124.5 GB of unified memory, so the model *fits on one node*. Only 10.5 M parameters are
   trainable, so gradients are **21 MB**. Switching `ds_config.json` from `"stage": 3` to
   `"stage": 1` replaces **207 GiB** of parameter all-gathers per optimizer step with a 21 MB
   all-reduce — a ~1000× reduction in inter-node traffic, for a one-line diff.
3. **The run is small and the compute is cheap either way.** Even the pessimistic dataset
   scenario is ~100 H100-GPU-hours ≈ **$380** on Nebius, or ~8 days on an otherwise-idle
   4-Spark cluster that costs €0.58/hr fully amortised. Money is not the deciding variable;
   wall-clock iteration speed is, and the Sparks are fast enough once (1) and (2) are applied.

Two further facts push the same way:

- **The prior art is less reusable than it looks.** `nebius-slurm-ml-training-and-inference-demo`
  builds a Soperator (Managed Slurm) cluster. Per Nebius's own docs, **GPU Soperator worker
  nodes require capacity block groups**, which are *not* self-service — they need a Nebius
  account manager and a commitment-discount addendum. The self-service path on default quota is
  a plain 8-GPU VM, which is not what the demo's Terraform builds.
- **The Sparks' unified memory is a genuine fit.** 51.7 GiB of weights plus 32k-token
  activations needs ~81 GiB. That fits a 124.5 GB GB10 and **does not fit an 80 GB H100**,
  which would force you back onto sharding. This run is unusually well matched to the hardware
  already owned.

### Switch to Nebius if any of these fires

| Trigger | Threshold |
|---|---|
| The corpus turns out large *and* you need to iterate | **> ~80 M tokens/epoch** (≈ 22 h/epoch on 4 Sparks) **and** ≥ 3 planned iterations |
| The memory estimate is wrong by ~2× at 16k | You are forced back onto **cross-node** ZeRO-3 |
| The Sparks are needed for serving/eval at the same time | Contention is unavoidable for > ~1 week |

### The one-hour experiment that settles this

Everything marked *extrapolated* below collapses to *measured* with a single cheap probe.
`dgx-spark-02/03/04` are idle right now (`dgx-spark-01` is serving `qwen3.8-27b-bf16`):

> Run 20 steps of `recipes/qwen3.6-27b-base-lora-longctx` at `max_seq_length=16384` against a
> synthetic long-context dataset, with `ds_config.json` at `"stage": 1`, single node.
> Record **s/step** and **peak RSS**. Then repeat on 2 nodes.

That yields the per-step time and the memory headroom, which are the only two unknowns that
matter. **Do this before spending anything on a Nebius port.**

---

## The numbers

### Cost and time per million training tokens

The model below is a roofline, calibrated against the one measured Spark datapoint. Derivation
and confidence are in [Appendix A](#appendix-a--how-the-compute-model-was-built).

Qwen3.8-27B LoRA with gradient checkpointing costs **≈171 GFLOP per training token** at
`seq=16384` (26.9 B params × 6, plus 6% for attention over the 16 full-attention layers).

| Venue | Dense BF16 | Time / 1M tokens | Cost / 1M tokens |
|---|---:|---:|---:|
| 1 × DGX Spark (GB10) | 125 TFLOPS *(derived)* | **65 min** | €0.16 |
| **4 × DGX Spark** | 500 TFLOPS | **16 min** | **€0.15** |
| **8 × H100 SXM (Nebius)** | 7.92 PFLOPS | **1.0 min** | **$0.53** |
| 8 × H100 preemptible | 7.92 PFLOPS | 1.0 min | $0.30 |
| 8 × H200 SXM (Nebius) | 7.92 PFLOPS | 1.0 min | $0.62 |

*Times at 35% MFU. Spark € figures use this repo's own cost model from
[`qwen3.6-27b-cost-analysis.md`](../qwen3.6-27b-cost-analysis.md): €0.288/hr for 2 nodes
including 3-year capex amortisation and 100 W/node at €0.30/kWh — so €0.576/hr for 4 nodes,
of which only €0.12/hr is electricity.*

**An 8×H100 node is ~16× the whole 4-Spark cluster** — that is simply the FLOPS ratio.
Per token the two venues cost about the same money. The difference is entirely wall-clock and
fixed costs.

### What that means for a real run

The corpus size is **not yet known** — it is exactly what [#50](https://github.com/kreuzhofer/dgx-manager/issues/50)
is chartered to establish. chat3d's strategy doc mentions *"~200+ high-scoring
agent_submitted examples and growing"*, which turn-level splitting multiplies. Bracketed
scenarios, at 3 epochs (equivalently, 1 epoch × 3 experiment iterations):

| Scenario | Rows × avg tokens | Tokens ×3 ep | 4 Sparks | 8×H100 | H100 GPU-h | $ on-demand |
|---|---|---:|---:|---:|---:|---:|
| **S** — 300 trajectories, one window each | 300 × 16k | 15 M | **4.1 h** | 16 min | 2.1 | **$8** |
| **M** — turn-level ×8 | 2,400 × 10k | 72 M | **20 h** | 1.2 h | 9.9 | **$38** |
| **L** — corpus grows | 5,000 × 12k | 180 M | **49 h** | 3.1 h | 25 | **$95** |
| **XL** — large corpus | 20,000 × 12k | 720 M | **8.2 d** | 12 h | 99 | **$381** |

The Sparks are comfortable through **M** and tolerable at **L**. **XL is where they stop being
the right answer** — which is the origin of the 80 M-tokens/epoch trigger above.

Note what this table says about Nebius: **the GPU bill is never the problem.** The problem is
that a Soperator cluster bills while it *exists*, not while it computes. The demo's shape —
2 GPU workers (16 H100) plus ~8 support VMs — is **$61.60/hr in GPU alone**; left up for a week
of iteration that is **~$10,300** for a job needing 2–99 GPU-hours. If Nebius is ever used, the right shape
is a single 8-GPU VM rented per shift with second-granular billing, not a standing Slurm
cluster.

### Nebius pricing, as published on 2026-08-30

Per-GPU-hour, verbatim from `nebius.com/prices` and `docs.nebius.com/compute/resources/pricing`
(the two agree exactly). Prices exclude VAT.

| GPU | Preemptible | On-demand | 8-GPU node, on-demand | Regions |
|---|---:|---:|---:|---|
| **HGX H100** | $2.15 | **$3.85** | **$30.80/hr** | **`eu-north1` (Finland) only** |
| **HGX H200** | $2.45 | **$4.50** | $36.00/hr | `eu-north1`, `eu-west1`, `us-central1` |
| HGX B200 | $3.95 | $7.15 | $57.20/hr | `us-central1`, `me-west1` — **no EU region** |
| HGX B300 | $4.30 | $7.85 | $62.80/hr | `uk-south1` |
| RTX PRO 6000 | $0.95 | $1.80 | $14.40/hr | `us-central1` (default quota **0**) |

Operationally relevant terms:

- **Billing granularity is 1 second.** Pricing unit is 1 hour. Renting an 8×H100 node for a
  90-minute job costs 1.5 × $30.80 = $46.20, not two hours.
- **Self-service, no sales call, for plain VMs.** Default `eu-north1` quota is **32 H100 GPUs
  without reservations** = four 8-GPU nodes. Signup is OAuth + card, minimum first payment $25.
- **GPU Managed Soperator is sales-gated.** Docs: *"If you need worker nodes with GPUs, make
  sure that you have capacity block groups that reserve GPUs"*, and capacity block groups
  require *"a request to your Nebius manager"* plus a commitment addendum. The Soperator
  software itself is free; the VMs under it bill normally, **including the ~8 support nodes**.
- **Ingress is free. VM egress is free.** Only Object Storage egress is charged ($0.015/GiB).
- **Shared filesystem $0.08/GiB-month** (4 TiB default quota in `eu-north1`).
- **No German region.** `eu-north1` (Finland) is the nearest, and is the *only* place H100
  exists on Nebius at all. It is EU/EEA, so GDPR-covered.
- Committed discounts are *"up to 35%"*, company-only, prepaid, and sized for the likes of
  *"64 VMs, each with 8 NVIDIA H100 GPUs"*. Not applicable at this scale.

There is **no published general free trial or signup credit**. The startup programme requires
*"at least $5M+ USD from an approved VC partner"*.

---

## Sub-question 2 — what actually has to change in the recipe

This is the most decision-relevant section, and the headline is: **less of the patch stack is
DGX-Spark-specific than its own docstring claims, and the largest body of work is
venue-independent and does not exist yet.**

`lib/patches.py` opens with *"DGX Spark hardware workarounds"*. Auditing it item by item
against what the code actually does:

### Category 1 — genuinely DGX-Spark-specific (delete on Nebius)

| Item | Where | Why it is Spark-only |
|---|---|---|
| `patch_pynvml()` | `lib/patches.py` | GB10 does not implement `nvmlDeviceGetMemoryInfo`; the patch fakes it from `/proc/meminfo`. H100 implements it. Harmless if left (it only fires on `NVMLError`) but pointless. |
| `patch_safetensors_cache()` | `lib/patches.py` | `posix_fadvise(DONTNEED)` after each shard, to stop the NFS page cache eating unified memory that the GPU also needs. On a discrete-VRAM box this only causes needless re-reads. |
| `flush_page_cache()` | `lib/patches.py`, called twice in `train.py` | Writes `/proc/sys/vm/drop_caches`; needs root and a privileged container. Fails soft elsewhere. Same unified-memory motivation. |
| `unset NCCL_IB_HCA` + the dual-rail RoCE block | `launch.sh` | ~25 lines reasoning about DGX Spark's two 100G MACs per QSFP port across PCI domains 0000/0002. Nebius is InfiniBand (H100: 400 Gbps, 8× ConnectX-7) with a fabric ID. Delete wholesale. |
| Custom NCCL 2.28.9 + `sm_121` build from `/workspace/nccl-build` | `launch.sh` | Built for GB10's Blackwell codegen. Stock NCCL in the NGC image is correct for `sm_90`. |
| `sync && echo 3 > /proc/sys/vm/drop_caches` | `launch.sh` | As above. |
| Hostfile rank discovery by matching local IPs | `launch.sh`, ~30 lines | Replaced by Slurm's `SLURM_NODEID` / `SLURM_NNODES` / `scontrol show hostnames`, or by `torchrun --rdzv_backend=c10d` on a single VM. Also `--nproc_per_node=1` → `8`. |
| `NCCL_SOCKET_IFNAME=enp1s0f0np0`, `NCCL_IB_HCA=rocep1s0f0`, `--device=/dev/infiniband/` | `packages/agent/src/runtime/finetune.ts:556-561, 662-667` | Hardcoded Spark interface names, on **both** the head and worker launch paths. |
| `apt-get install openssh-server` + sshd on :2233 + host `~/.ssh` mount | `entrypoint.sh` | Exists only because the agent SSHes head→worker to start the worker container. Slurm/pyxis does the launching. Deleting it also removes `PermitRootLogin yes` and `StrictHostKeyChecking no`. |

### Category 2 — *not* Spark-specific, despite where it lives (keep)

| Item | Where | Why it is generally needed |
|---|---|---|
| **`patch_nvtx_dummy_domain()`** | `lib/patches.py` | Filed under "DGX Spark hardware workarounds", but its own docstring gives it away: *"Single-rank runs don't trip this… Multi-node runs reproducibly hit it."* It is a **DeepSpeed × nvtx arity bug in the NGC image**, triggered by multi-rank ZeRO-3 parameter partitioning — the same thing you would run on Nebius. Keep it (or make it moot by dropping ZeRO-3). |
| **The 4-hour NCCL timeout** | `train.py` (`default_pg_timeout`), `launch.sh` (`TORCH_NCCL_ASYNC_ERROR_HANDLING=1`) | Motivated by slow ZeRO-3 `from_pretrained` broadcasts, but the Nebius demo hit the same class of problem and set `NCCL_TIMEOUT=3600`. **Keep the concept, lower the value, and drop `ASYNC_ERROR_HANDLING=1`** — it disables the timeout entirely, and the agent's own comments note this leaves the head hanging forever if a worker dies. On metered hardware that hang is billed. |
| PEFT torchao dispatcher disable | `patch_peft_for_clippable_linear()` | NGC `pytorch:25.11` ships torchao 0.14.0 but PEFT requires ≥ 0.16.0. That is an **image** property, identical on x86. Keep while on this image. |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | `launch.sh` | Motivated by GB10 fragmentation, but harmless-to-helpful anywhere with variable-length activations. Keep. |
| Liger fused linear cross-entropy | `train.py` | **Load-bearing at long context on any hardware.** At `seq=16384` and vocab 248,320 the logits tensor is 7.6 GiB in BF16 and 15.2 GiB after `ForCausalLMLoss` casts to fp32. Keep. |
| `per_device_eval_batch_size=1` | `train.py` | Same reason — Liger only patches the training forward, so eval uses stock CE. Keep. |
| Drop `token_type_ids` / `mm_token_type_ids` | `train.py` | A Gemma-4 artefact of `lib/dataset.py`, unrelated to hardware. Keep until the dataset layer is rewritten (see Category 3), which removes the need. |
| `keep_in_memory=True` on `datasets.map()` | `lib/dataset.py` | Justified as an NFS Arrow-cache fix, and Nebius shared FS is also network storage, so the race is not obviously gone. **But see Category 3 — with images this becomes actively harmful.** |
| Per-rank shell `tee` to `$output_dir/train.log` | `lib/setup_logging.sh` | This is what makes the manager's log view and loss parser work. **Keep it in both venues.** |
| `LogMetricsCallback` emitting `[TRAIN]` / `[EVAL]` lines | `lib/logging.py` | The observability contract. Keep. |

### The container is a non-issue

`nvcr.io/nvidia/pytorch:25.11-py3` is a **multi-arch manifest** — verified directly:

```
$ docker manifest inspect nvcr.io/nvidia/pytorch:25.11-py3
linux amd64 sha256:e14cf0da7ca0d878d0874eb81062b77df275491d4a8d030a2a7463a4e8b07f01
linux arm64 sha256:4a85d8cf6fb3a943280960b8948cf4e9b6eca77b4414c68c9b2c7bb863f79b70
```

The same tag pulls an x86 image on Nebius. The "arm64 container lineage" concern in the issue
is confined to things *built locally* (the `sm_121` NCCL) and to aarch64 wheel availability —
and **moving to x86 makes wheel availability strictly better**, not worse.

### The DeepSpeed config — the one change worth making *regardless of venue*

`ds_config.json` is `"stage": 3`. For a LoRA run on a model that fits in one device's memory,
that is pure overhead:

| | ZeRO-3 (current) | ZeRO-1 (proposed) |
|---|---|---|
| Param all-gather per micro-step | 25.9 GiB × 2 (fwd + bwd) = **51.7 GiB** | none |
| Traffic per optimizer step (`grad_accum=4`, 2 ranks) | **207 GiB** | **21 MB** (10.5 M trainable × BF16) |
| Fits on one GB10 (124.5 GB)? | yes | yes — 51.7 GiB weights + 10.0 GiB activations at 16k |
| Fits on one 80 GB H100? | yes (sharded) | **no** at 32k |

Change `"stage": 3` → `"stage": 1`, and drop `stage3_prefetch_bucket_size`,
`stage3_param_persistence_threshold` and `stage3_gather_16bit_weights_on_model_save`.
One caveat: without sharding, every rank loads the full 51.7 GiB from `/mnt/tank`
concurrently — which is precisely what `patch_safetensors_cache()` was written for, so keep it
on the Sparks.

*Note the asymmetry this creates: on an 80 GB H100 you would keep ZeRO-3, because NVLink at
900 GB/s makes the gathers nearly free. On a GB10 or an H200 (141 GB) you would not.*

### Category 3 — needed in **both** venues, and not built yet

This is the largest item in the whole analysis and it is **orthogonal to the venue question**.
The recipe as it stands cannot train what this map calls for:

1. **No image path at all.** `lib/dataset.py::format_example()` calls
   `tokenizer.apply_chat_template(...)` then `tokenizer(text)`. It produces `input_ids` and
   `attention_mask` only, and hardcodes `token_type_ids`/`mm_token_type_ids` to zeros. There is
   no `AutoProcessor`, no `pixel_values`, no `image_grid_thw`. `train.py` loads
   `AutoModelForCausalLM`, not `AutoModelForImageTextToText`. **Render screenshots cannot
   currently reach the model.** The map's central premise — that the vision path keeps getting
   gradient — depends on work that does not exist.
2. **No assistant-turn loss masking.** `DataCollatorForLanguageModeling(mlm=False)` derives
   labels from `input_ids`, so loss is computed on *every* token including user turns and tool
   output. The map specifies *"turn-level, loss masked to assistant turns"*. That needs a
   custom collator.
3. **`keep_in_memory=True` becomes a hazard with images.** A decoded 768×768 RGB frame is
   1.7 MB; 10,000 of them is ~17 GB held in RAM — on a GB10 that RAM is the same pool the model
   is using. Images must stay on disk and be decoded in the collator.
4. **No tool-call round-trip test.** `_normalize_openai_messages_for_qwen()` exists for Qwen
   tool-call templates, but nothing verifies that the chat template round-trips *images plus*
   tool calls together, which is what an agentic trajectory contains.

Estimate: this is several days of work either way, and it is the **critical path**. Spending
the porting budget on the venue while this is unbuilt would be the wrong order.

---

## Sub-question 3 — can dgx-manager drive a remote Nebius job?

**Yes, and more cheaply than expected — because the observability layer is already
venue-neutral.** The loss curves and phase tracking are not coupled to DGX hardware at all;
they are coupled to a *log format that we control*.

### How the observability actually works

- `packages/agent/src/runtime/finetune.ts` (1,030 lines) contains a **pure line parser**,
  `parseProgress(line, currentPhase)`, matching:
  - `/\[TRAIN\]\s+step=(\d+)\/(\d+)\s+loss=([\d.eE+\-?]+)(?:\s+lr=([\d.eE+\-]+))?/`
  - `/\[EVAL\]\s+eval_loss=([\d.]+)/`
  - plus tqdm-percentage fallbacks routed by the current phase.
- Those lines are emitted by `LogMetricsCallback` in the recipes repo's `lib/logging.py`.
- Phases are `downloading → loading → tokenizing → training → eval`, inferred from log text.
- Results reach the dashboard as WebSocket messages: `finetune:log`, `finetune:status`,
  `finetune:merge-progress`, `finetune:merge-status`, `finetune:quantize-progress`,
  `finetune:quantize-status`, plus `agent:training-recipes` at handshake.

**Nothing in that chain requires a DGX node.** It requires a process that writes our log format
to stdout and something that relays the lines.

### The real coupling is the shared filesystem, not the hardware

`packages/server/src/routes/finetune.ts` (1,299 lines) anchors everything on `SHARED_STORAGE`
(`/mnt/tank`): datasets resolve there, outputs go to `${SHARED_STORAGE}/outputs/{jobId}`, and
the merge and FP8-quantize steps read and write there. The manager and the nodes share a
filesystem. **Nebius does not share a filesystem with the manager** — that, not GPU metrics, is
what a remote venue breaks.

### Three options, costed

| Option | What you get | What you lose | Effort |
|---|---|---|---|
| **(a) Run the dgx-manager agent on a Nebius login node** | Registers as a node; live everything | Drags the whole deploy/reconcile surface along; `metrics.ts` works but sparkrun/dgxrun paths do not; still no shared FS with the manager; the hardcoded `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` / `--device=/dev/infiniband/` need per-venue branching | Medium, high ongoing friction |
| **(b) A server-side "remote job" executor** — submit over SSH, tail the log, feed the *existing* parser | Live loss curves, phases, log streaming, job status, DB history, the dashboard unchanged | Dataset staging needs an upload step; merge/quantize still assume shared storage | **~500–800 LOC** + tests; a `venue` column on `FineTuneJob` |
| **(c) Run it like the demo repo, import afterwards** | Full loss curve and phase history reconstructed post-hoc by replaying the downloaded `train.log` through the same pure parser | **Live** streaming only | **~150–200 LOC** |

The important conclusion: **Nebius does not mean losing the loss curves.** Because the parser is
pure and the log format is ours, option (c) recovers the entire curve for a couple of hundred
lines. It only costs you *live* streaming. That materially weakens "observability" as an
argument for either venue — but it also means the observability you get for free on the Sparks
is worth ~500–800 LOC to replicate live, which is not nothing for a run this small.

---

## Sub-question 4 — data movement

**Technically a non-issue. The open question is governance, not bandwidth.**

- **Size.** Row counts are blocked on [#50](https://github.com/kreuzhofer/dgx-manager/issues/50).
  Bracketing: one 768×768 render is **677 prompt tokens** (measured on this fleet, 2026-08-28)
  and typically 200–600 KB as PNG. At 400–20,000 images that is **0.1–6 GB**. The text side is
  smaller: a 16k-token row is ~64 KB of JSON, so 20,000 rows ≈ 1.3 GB.
- **Transfer cost: $0.** Nebius publishes *"Networking: Egress/Ingress traffic — Free"*, and
  the VPC docs confirm the service is free of charge. Only Object Storage egress is billed
  ($0.015/GiB), which does not apply to uploading a dataset onto a shared filesystem.
- **Transfer time: minutes.** A 40 Mbit/s domestic upstream moves ~18 GB/hr, so 6 GB is ~20
  minutes. Not a blocker.
- **Storage cost: negligible.** 6 GB on the Nebius shared filesystem at $0.08/GiB-month is
  ~$0.50/month.
- **Acceptability is Daniel's call, not a technical finding.** The renders are of chat3d users'
  CAD geometry. The best available answer on residency is `eu-north1` — Finland, EU/EEA,
  GDPR-covered — and note there is **no German Nebius region**, and H100 exists *only* in
  `eu-north1`. That the trajectories were already generated by sending user prompts to a
  frontier API is a precedent for per-request processing, but a **bulk upload of a user-content
  corpus to a third party for model training is a different act** and should be decided
  explicitly, not inherited. This research does not resolve it.

Training on the Sparks sidesteps the question entirely, which is a real if unglamorous point in
its favour.

---

## Appendix A — how the compute model was built

**Per-token training cost.** LoRA with gradient checkpointing, so the dense-matmul term is
forward `2P` + input-gradient backward `2P` + checkpoint recompute `2P` = `6P`. Weight
gradients are skipped for the frozen base; the LoRA branch is negligible at 10.5 M params.

- `6 × 26.9e9 = 1.614e11` FLOP/token.
- Attention adds `96 × S × d_q` FLOP/token over the **16 full-attention layers**, with
  `d_q = num_attention_heads × head_dim = 24 × 256 = 6144` (read from the local
  `Qwen/Qwen3.8-27B` `config.json`). At `S=16384` that is `9.66e9` — **+6.0%**. At 32k, +12%.
  The hybrid architecture (48 of 64 layers are linear-attention) and GQA (4 KV heads) keep this
  small.
- The vision tower (depth 27, hidden 1152 ≈ 0.41 B params) costs `~1.7e12` FLOP per image
  against `~2.8e15` for a 16k-token sequence — **0.06%**. Ignorable.

**Total: ~171 GFLOP/token at 16k, ~181 at 32k.**

**Hardware peaks, dense BF16.** H100 SXM = 989.5 TFLOPS (datasheet's 1,979 is with sparsity);
H200 SXM has identical tensor cores. GB10 = **125 TFLOPS, derived** from the DGX Spark "1
PFLOP at FP4" figure via Blackwell's FP4:BF16 = 4:1 dense ratio and the sparsity factor of 2.
This is not a datasheet BF16 row and has not been measured locally — **treat it as ±25%**.

**Calibration against the one measured datapoint.** The 3.6-27B run measured **5.5 s/step**
single-node at `grad_accum=4, bs=1, seq=256` = 1,024 tokens/step = **186 tokens/s**. Times
`1.614e11` gives **30 TFLOP/s = 24% of the derived 125 TFLOPS peak**. That is a credible MFU
for 256-token micro-batches, and it cross-validates the peak: if GB10's true BF16 peak were
60 TFLOPS, the implied MFU would be 50% on tiny GEMMs, which is not believable.

### Confidence

| Claim | Status |
|---|---|
| Nebius prices, regions, quotas, Soperator gating | **Measured** — first-party pages, 2026-08-30 15:24–15:32 UTC |
| 3.6-27B step times at seq=256; 51.7 GiB BF16; 677 tokens/image; multi-arch manifest; config dims | **Measured** on this fleet or read directly from the artefact |
| ZeRO-3 comm volume is sequence-independent; LoRA gradients are 21 MB; activation-memory arithmetic | **Derived**, high confidence — properties of the algorithm and the config, not of the hardware |
| **Step time at 16k/32k, and therefore every wall-clock figure** | **Extrapolated, medium confidence.** The FLOP model is sound, but MFU at long sequence has *never been measured on GB10*, and 24% at seq=256 is the only anchor. A 25–45% MFU band puts the per-1M-token times at **±40%**. Longer sequences should raise MFU (bigger GEMMs), so the estimates are more likely conservative than optimistic — but that is reasoning, not measurement. |
| GB10 = 125 TFLOPS dense BF16 | **Derived** from a marketing FP4 figure. ±25%. |
| Corpus token count, hence total wall time | **Unknown** — blocked on [#50](https://github.com/kreuzhofer/dgx-manager/issues/50) |

The one-hour probe in the recommendation converts rows 3 and 5 to *measured*. Nothing else in
this document is worth more than that experiment.

---

## Appendix B — if Nebius wins later, the concrete checklist

Ordered, assuming the self-service path (a plain 8-GPU VM in `eu-north1`, **not** Soperator):

1. **Account.** OAuth signup, add a card, $25 minimum first payment. Default quota already
   allows 32 H100 GPUs in `eu-north1`. No sales call. *(Sales is required only if you want
   Managed Soperator with GPU workers, or committed pricing.)*
2. **Storage.** Create a shared filesystem (default quota 4 TiB, $0.08/GiB-month). Upload the
   dataset — ingress is free.
3. **Recipe port.** Delete every Category-1 item above; keep every Category-2 item; lower the
   NCCL timeout and drop `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`. Replace hostfile rank discovery
   with `torchrun --nproc_per_node=8` on the single VM. Pull the same
   `nvcr.io/nvidia/pytorch:25.11-py3` tag (amd64 digest). Bake the `pip install` from
   `entrypoint.sh` into an image rather than running it per job.
4. **DeepSpeed.** Keep ZeRO-3 on 80 GB H100s (NVLink makes it cheap); use ZeRO-1 on H200.
5. **Observability.** Implement option (c) first — post-hoc import of `train.log` through the
   existing parser, ~150–200 LOC. Only build option (b) if live streaming proves necessary.
6. **Discipline.** Rent per shift, not per week. Billing is per second; a standing cluster is
   what makes cloud expensive at this scale.

The prior-art repo `/mnt/tank/src/github/nebius-slurm-ml-training-and-inference-demo` remains
useful for the **`.envrc` auth bootstrap** (IAM token, project/subnet lookup, service account,
S3 Terraform backend) and for its `train_lora.py` / sbatch patterns — but its Soperator
Terraform targets a cluster shape that is not self-service, and its recorded runs
(`max_length=1024`, sql-create-context) are too short-sequence to calibrate this workload.

---

## Sources

- Nebius pricing, `https://nebius.com/prices` and `https://docs.nebius.com/compute/resources/pricing` — checked 2026-08-30 15:24–15:32 UTC
- Nebius regions, quotas, Soperator and capacity-block-group docs, `https://docs.nebius.com/` — same session
- [`docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md`](../qwen3.6-27b-fine-tuning-on-dgx-spark.md) — the measured 3.6-27B step times and the multi-node investigation
- [`docs/qwen3.6-27b-cost-analysis.md`](../qwen3.6-27b-cost-analysis.md) — the Spark €/hour cost model
- [`docs/qwen3.8-model-survey.md`](../qwen3.8-model-survey.md) §3, §11 — architecture identity with 3.6-27B, 51.7 GiB BF16, 677 tokens/image
- `kreuzhofer/dgx-manager-fine-tune-recipes` — `lib/patches.py`, `lib/dataset.py`, `lib/logging.py`, `lib/setup_logging.sh`, `recipes/qwen3.6-27b-base-lora{,-longctx}/`
- `packages/server/src/routes/finetune.ts`, `packages/agent/src/runtime/finetune.ts`, `packages/server/src/ws/agent-hub.ts`
- `/mnt/tank/src/github/nebius-slurm-ml-training-and-inference-demo` — `README.md`, `DEMO_SUMMARY.md`, `demo/scripts/`
- chat3d `docs/local-model-strategy.md` (2026-06-14) — the 16k/32k sequence-length requirement and the dataset filter rule
