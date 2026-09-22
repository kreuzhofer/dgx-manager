# Serving Qwen-Image-2.1 on the GB10 Spark cluster

**Research date:** 2026-09-20 · **Asked as:** *"latest Grace Blackwell is arm so why shouldn't there be support?"*

**Why this document exists.** Issue #116 was filed the same day claiming vLLM-Omni is amd64-only
and therefore that all four Sparks were excluded from serving any diffusion model. That claim was
wrong. This document is the correction, and the question that prompted it was the right one: Grace
Blackwell *is* aarch64, so an "arm64 unsupported" statement about a Grace-targeting project should
never have survived first contact.

**Method.** Primary sources only — project source code, build files, registry manifests, NVIDIA
support matrices, and upstream recipes. Three parallel agents researched packaging, sm_121 kernel
gating, and the non-vLLM alternatives. Registry and Hugging Face facts below were then
**re-verified first-hand** against the Docker Hub v2 API and `huggingface.co/api/models/<repo>?blobs=true`,
because the agents disagreed on one of them (§A.1).

**Labelling.** `[verified]` = read in the owning source. `[inference]` = my arithmetic or reasoning.
`[unconfirmed]` = could not close. Every sm_100 → sm_121 extrapolation is called out, because that
is the single most likely way to be wrong about this hardware.

---

## ⚠️ MEASURED OUTCOME (2026-09-21) — read this before quoting anything below

Everything from §1 down is the **pre-measurement research**, kept as written so predictions can be
compared with what happened. It has since been run on real hardware. **Where the two disagree, the
measurements win.**

**Setup.** `dgx-spark-03` (GB10, 121 GiB usable), image `vllm/vllm-omni:qwen-image21-arm64-cu130`
(26 GB on disk), vLLM 0.29.0, `vllm serve Qwen/Qwen-Image-2.1 --omni --port 8091`, single rank,
weights on `/mnt/tank`. Container flags copied from a live dgxrun deploy (`--ipc=host`,
`--network=host`, 32 GiB shm, `CAP_IPC_LOCK`, `/dev/infiniband`, `label=disable`). Run outside
manager control as an unowned container — see the note at the end.

| § | prediction | measured | verdict |
| --- | --- | --- | --- |
| §3 | cuDNN 9.20 may lack sm_121 kernels; upgrade to ≥9.24 | **works as shipped** — `Defaulting to diffusion attention backend CUDNN_ATTN (Blackwell sm_121, cuDNN 92000, head_dim 128)`, graphs captured, output correct | ❌ prediction too cautious |
| §2 | sm_120 cubins should run on sm_121 by family compatibility | confirmed — no `no kernel image`, device init clean | ✅ |
| §6 | ~30–60 s per 1024²/40 steps | **55.7 s cold, 52.9 s warm** | ✅ |
| §6 | "≥~70 s" floor at 2048² `[inference]` | **271.2 s** | ❌ **~4× optimistic** |
| §5 | BF16 fits in ~121 GiB, FP8 optional | peak **77 GiB** of 121 | ✅ |
| §8.3 | readiness probe "probably free" | **free** — `/v1/models`, `/health`, `/metrics` all 200; `/v1/images/generations` 405 on GET (POST-only) | ✅ |
| §4 | image has an ENTRYPOINT trap | **no entrypoint** — `Entrypoint: None`, command passes straight through | ❌ non-issue |

**The 2K number, which did not previously exist anywhere:** **271.2 s** for 2048×2048 at 40 steps.
That is **4.9× the 1024² cost** against a 4× pixel ratio — so it scales slightly worse than area,
not catastrophically. My §6 arithmetic floor of ~70 s was wrong by ~4×, and it was wrong in the
direction that flatters the model. Treat the scaled-floor method as discredited for diffusion.

**Output correctness.** Verified numerically rather than by eye, because the encoder returns PNGs at
**compression level 0** (IDAT ratio exactly 1.000), which looks alarming and is not: raw pixels
compress to **0.519** and the mean horizontal neighbour delta is **1.33** (noise would be ~85).
Alpha came back 254.9 ± 0.3 — a valid opaque RGBA surface. So: real images, not the silent-garbage
failure mode #7759's open review warns about. This is *not* a check against a reference
implementation — it rules out noise, not subtle drift.

**Costs not in the original research.** First-touch weight download was **1179 s (19.7 min)** for
31 GB onto `/mnt/tank` (~26 MB/s) — cached thereafter. Total cold start to serving was ~23 min, of
which only ~3 min was load plus graph capture.

**Operational note.** The probe ran as a container the manager does not own, which exposed a real
gap: admission reads live metrics (`vram.ts:168`) so a *fresh* deploy is correctly refused, but
`loadConflicts` builds its list from deployment rows, so the 409 names nothing (#101). Worse, on a
**restart** `reclaimableMB = vramUsed − otherFootprint` credits an unowned container's memory to the
restarting deployment, so the node reads as empty and the restart is admitted into contention. Tear
down unowned probes before restarting anything on that node.

## §0 Verdict

**Qwen-Image-2.1 is servable on the Sparks via vLLM-Omni, and the Sparks are the *better* host than
the RTX 5090.** There is no architecture exclusion. There is one live version gate (cuDNN, §3), two
operational gates (§4), and one genuine open risk that is about speed rather than feasibility (§6).

The decision this research does **not** make: whether a ~30–60 s/image generator, at an unmeasured
and probably much worse 2K, is worth a Spark. That is a product call, and §6 is the input to it.

---

## §1 The false blocker, and how it happened

The claim in #116 came from NVIDIA's Dynamo backend docs at **v1.2.1**:

> "vLLM-Omni is currently only installed on `amd64` builds. On `arm64`, the container build skips
> the install and vLLM-Omni features are unavailable."

Three independent things are wrong with treating that as an upstream capability statement:

1. **It describes Dynamo's own container build, not vLLM-Omni.** `[verified]` The literal gate is in
   `container/deps/vllm/install_vllm.sh`:
   `if [ -n "$VLLM_OMNI_REF" ] && [ "$ARCH" = "amd64" ]; then … else echo "⚠ Skipping vLLM-Omni (no ref provided or ARM64 not supported)"`,
   plus a `{% if platform == "amd64" %}` guard on the `vllm_omni` site-packages COPY in
   `vllm_runtime.Dockerfile`. No comment anywhere justifies it technically. The doc sentence was
   added by a **doc-only** PR (ai-dynamo/dynamo#8373, "No code changes", +4 lines) whose linked
   internal ticket is titled *"vllm-omni module missing from arm64"* — an artifact defect.

2. **Dynamo removed the gate four months before I quoted it.** `[verified]` PR #7648 (merged
   2026-05-15) moved to upstream `vllm/vllm-openai` base images and installs vllm-omni from PyPI
   unconditionally; the only surviving gate is on *device* ("CUDA-only"). `platform == "amd64"`
   occurs **2× at v1.2.1 and 0× from v1.3.0 through v1.4.2**. The sentence lingered stale in the
   docs through v1.3.1 and is gone by v1.4.0. **The URL I read is a pinned historical page.**

3. **vLLM-Omni has never had architecture gating at all.** `[verified]` No `ext_modules` in
   `setup.py`, no `CMakeLists.txt`, no `CMAKE_CUDA_ARCHITECTURES`, `TORCH_CUDA_ARCH_LIST` only in
   `collect_env.py`. **All 21 PyPI releases (0.11.0rc1 → 0.29.0rc1) are `py3-none-any` universal
   wheels.** It is pure Python layered over vLLM. Its `docker/Dockerfile.cuda` is 20 lines with zero
   arch logic, and its docs state one hardware requirement: *"GPU: compute capability 7.0 or higher"*.

The two issues cited in #116 as corroboration, vllm-omni **#195** and **#1571**, are both explicitly
about *container images* — "Support **Container Image Format** → ARM64" and "There are no ARM64
**Docker image** releases on Docker Hub". `[verified]` Both are still open and stale. The decisive
comment is maintainer **ywang96** on #1571:

> "Since `vllm-omni` only has python code, you should be able to simply take the corresponding
> `vllm` arm64 docker images and build one yourself fairly easily."

The asker replied the same day: *"Yep built fine and working, thanks!"*

**The lesson, for next time.** A distribution gap was read as a hardware exclusion, and the
contradiction was sitting inside the same evidence — a GB300 measurement quoted one paragraph from
the claim that arm64 was unsupported. For this fleet the question is never *arm-vs-x86*; the whole
fleet is arm64 and building custom images is routine here (`build-glm53-flash-image.sh` is eight
thin layers over an arm64 base; `build-glm52-image.sh` compiles vLLM from source on arm64). The
question that actually bites is **sm_121 vs sm_100**.

---

## §2 What actually runs on aarch64 — the positive evidence

**Official multi-arch images exist and include this exact model.** `[verified first-hand, Docker Hub
v2 API, 2026-09-20]`

| tag | architectures | last pushed |
| --- | --- | --- |
| `vllm/vllm-omni:qwen-image21` | **amd64 + arm64** | 2026-09-20 13:04 |
| `vllm/vllm-omni:qwen-image21-arm64-cu129` | arm64 | 2026-09-20 13:02 |
| `vllm/vllm-omni:qwen-image21-arm64-cu130` | arm64 | 2026-09-20 13:02 |
| `vllm/vllm-omni:latest` | amd64 + arm64 | 2026-09-10 |
| `vllm/vllm-omni:nightly` | amd64 + arm64 | 2026-09-20 |
| `vllm/vllm-omni:v0.28.0` | amd64 + arm64 | 2026-08-31 |

vllm-omni PR #3428 added a `Build release image - aarch64` step on `arm64_cpu_queue_release`.

**Upstream maintains a DGX Spark recipe for a diffusion workload.** `[verified first-hand]`
`recipes/MiniMaxAI/MiniMax-H3-Spark-GB10.md` in the vllm-omni repo, 15 KB:

> Validated on: — Host: **DGX Spark (GB10), aarch64** — GPUs: 1, or 2 hosts with one unified-memory
> GPU each — Driver: 580.173.02 — vLLM: 0.26.0 — vLLM-Omni: `main` at `e1aa6eae…`

It carries measured peak-allocator high-water marks (97.7 GiB T2VA, 102.8 GiB Ref2VA, 75.45/87.17
GiB per rank two-host), and it is unified-memory-aware in the way that matters:

> "GB10 is a unified-memory platform, so unlike the discrete-GPU recipes it uses **no offload of any
> kind**."

The two-host variant uses **text-encoder TP2 and DiT USP2** — directly relevant, since we have four
Sparks and a RoCE fabric.

**The one real historical arm64 blocker was a transitive dep.** `[verified]` `fa3-fwd` 0.0.1 was
x86_64-wheel-only, which is the actual DGX Spark failure reported in #195. An aarch64 wheel shipped
in 0.0.2 (2026-03-26); the repo now pins `fa3-fwd==0.0.3`.

**Caveat worth keeping.** `[verified]` aarch64 is *built* in release CI but **not functionally
GPU-tested** — no `arm64`/`aarch64` entry in any `.buildkite/cuda/` pipeline, and no GB200/GB300/GB10
SKU marker in the pytest matrix. The GB10 recipe above is hand-validated, not CI-validated.

---

## §3 The one live arch gate: cuDNN 9.20 vs sm_121

This is the finding to act on, and it is the exact pattern this fleet keeps hitting.

`[verified]` vLLM-Omni selects `CUDNN_ATTN` **automatically on sm_121**, gated on
`cudnn_version >= 90500` with the comment *"cuDNN 9.5+ ships Blackwell FMHA kernels"*.

`[verified]` `torch==2.13.0` — vLLM 0.29.0's pin — bundles `nvidia-cudnn-cu13==9.20.0.48`.

`[verified]` Read directly from three NVIDIA cuDNN support matrices: **9.20 → no CC 12.1. 9.22 → no
CC 12.1. 9.24 → CC 12.1 present.** Support enters in the 9.23.x line.

So the version check passes ~18 minor versions before NVIDIA documents the architecture. The gate is
satisfied by a cuDNN that does not claim to support our GPU.

**`[unconfirmed]`** whether 9.20 actually *fails* on sm_121 or is merely undocumented. Evidence
against alarm: the in-repo MiniMax-H3 GB10 recipe ran `CUDNN_ATTN` successfully on vLLM 0.26.0. But
"undocumented and it worked for someone else on a different model" is a thing to pin, not to hope
about.

**Fix, one line:** `pip install -U 'nvidia-cudnn-cu13>=9.24'` — and pin it in the recipe with this
paragraph as the reason.

---

## §4 The operational gates

**Take the cu130 wheel, not cu129.** `[verified by read-only gencode probe on dgx-spark-01]`
`sm_120`, `sm_120f` and `sm_121` all run; **`sm_120a` fails**, PTX included. A cu129 wheel therefore
does not refuse to load — it *silently* loses its Marlin / CUTLASS-FP8 kernel families. This matches
NVIDIA's own DGX Spark playbooks, which are 99-to-0 `cu130`. Since both `-arm64-cu129` and
`-arm64-cu130` tags are published (§2), this is a tag choice, not work.

**PR #7759 is still open.** `[verified]` Updated 2026-09-20, TODO list incomplete, model id
provisional, and reviewers have flagged correctness bugs in KV-cache aliasing and condition-latent
ordering plus incompatibility with the advertised continuous batching
(`--step-execution --max-num-seqs 8`). The published `:qwen-image21` image is built from that branch.
Our own standing rule from #100 — *pin a commit, never a branch head* — applies with extra force when
the branch has open correctness review.

---

## §5 Memory: it fits, comfortably

`[verified first-hand, HF blobs API]` The checkpoint is **33.1 GB / 30.9 GiB** across 27 files:

| component | size |
| --- | ---: |
| `text_encoder` (Qwen3-VL 8B) | 17.5 GB |
| `transformer` (7.1 B single-stream DiT, 32 layers) | 14.2 GB |
| `vae` (64-ch RGBA, 16× spatial compression) | 1.4 GB |

Against a Spark's **~121 GiB usable** unified memory, BF16 fits with room to spare — so FP8, the
least-verified path in the pipeline, is **optional** rather than mandatory. Upstream's only anchor
measurement is 34.0 GB peak on a GB300 at 1024²/40 steps.

`[inference]` For contrast, this is why aihost01 is the wrong host: 30.9 GiB of weights against
31.8 GiB usable on the RTX 5090 leaves essentially nothing for activations. #116 originally proposed
the 5090 as the only candidate; it is the one node that cannot hold this model.

---

## §6 Performance — the real open question

Feasibility is settled. Speed is not, and it is the reason to hesitate.

| hardware | model | resolution | time | source |
| --- | --- | --- | ---: | --- |
| GB300 | Qwen-Image-2.1 | 1024² / 40 steps | **4.5 s** | upstream recipe |
| DGX Spark | Qwen-Image v1 (SGLang) | 1024² / 40 steps | **35.36 s** | third-party measured |
| DGX Spark | Qwen-Image v1 (diffusers) | 1024² / 40 steps | **53.2 s** | third-party measured |
| DGX Spark | Qwen-Image-2512 (20B, diffusers) | 1024² | **61 s** | third-party measured |
| DGX Spark | FLUX.2-klein-9B | 1024² | 4.4 s | third-party measured |
| DGX Spark | Z-Image-Turbo | 1024² | 7.2 s | third-party measured |
| DGX Spark | SDXL | 1024² | 11.3 s | third-party measured |

Budget **~30–60 s/image at 1024², roughly 8–12× a GB300.** `[inference]` That gap is bandwidth:
~273 GB/s of LPDDR5X against HBM measured in TB/s, and a 40-step denoise is a bandwidth problem.

**Nobody has published a 2K number, on any hardware.** `[verified — searched]` Every figure in the
record, everywhere, is 1024². Native 2K is the entire reason to prefer this model over its
predecessors, and it is the operating point with zero data. A scaled floor is **≥~70 s** on a Spark
`[inference, arithmetic only — do not quote as a measurement]`.

Two cheap levers are documented, both on the diffusers path: `DIFFUSERS_ATTN_BACKEND=_native_cudnn`
took one model from 39.3 → 13.9 s, and NVFP4 gave a further ~20%.

---

## §7 The alternatives, and why vLLM-Omni still wins

**NVIDIA NIM for Visual GenAI** already solves the API-shape problem — and for the wrong models.
`[verified]` It ships **aarch64** containers exposing `/v1/images/generations`, `/v1/health/ready`,
`/v1/metrics` and Prometheus `nv_*` on port 8002 — exactly the three surfaces our stack probes. DGX
Spark has its own row in the support matrix with prebuilt FP4/FP8 TensorRT engines at 128 GB, and is
named in the release notes (1.3.2/1.3.3 FLUX.1, 1.4.1 SD3.5-Large). **There is no Qwen-Image-2.1
NIM.** Worth remembering the day we want FLUX or SD3.5 — it would be zero integration work.

**Plain `diffusers`** is architecturally the cleanest: `[verified]` no arch gate, stock SDPA
attention, and the docs state it *"needs no compilation and works on any PyTorch build"*.
`QwenImage21Pipeline` requires diffusers from git — PR #14804 merged 2026-09-18, absent from v0.40.0.
Cost: we would write 100% of the serving layer, with no `/v1/*`, no health, no metrics.

**ComfyUI** is the weakest option. `[verified]` Qwen-Image-2.1 support is master-only, landed
2026-09-19/20 *after* tag v0.36.0 — and NVIDIA's DGX Spark playbook pins v0.33.2, so taking this path
leaves NVIDIA's tested branch. Reading `server.py` directly: no `/v1/*`, no `/metrics`, no `/health`
(zero hits for all three). `POST /prompt` is async, takes a *workflow graph* rather than a prompt
string, and returns a `prompt_id` to poll via `/history/{id}` then `/view`. The two public
OpenAI-compatible shims are 0-star personal repos, neither shipping health or metrics. A documented
unified-memory bug also halved the machine — ComfyUI's safetensors `mmap` loader double-allocated on
GB10, capping usable memory at ~64 of 128 GB; confirmed by an NVIDIA moderator, fixed upstream,
closed 2026-06-15, but an operational hazard to know about. `[unconfirmed]` its `comfy_kitchen`
aarch64 wheel embeds `sm_120f` but not `sm_121`; NVIDIA's family-compatibility rule suggests
`sm_120f` covers GB10, but no NVIDIA doc names sm_121 as a valid `sm_120f` cubin target. Affects only
the int8/NVFP4 path.

**Why vLLM-Omni wins anyway:** it is `vllm serve … --omni`, so it is the vLLM OpenAI server. We get
`/v1/models` and `/metrics` for free, which is what the agent's `apiReady` gate already probes — and
an arm64 image for this exact model already exists.

---

## §8 What it costs in this repo

1. **Gateway: a third forwarded path.** `FORWARDED_PATHS` in `packages/server/src/gateway/proxy.ts:131`
   is exactly `chat/completions` and `embeddings`. Image generation is `/v1/images/generations` —
   different request shape, different response shape (b64/URL), far longer time-to-first-byte.
   `gateway/models.ts` eligibility must learn that an image model is not a chat model.
2. **Recipe: the shape already exists.** `arch: amd64` / `arch: arm64` and `cluster_only` are
   recipe-declared (`dgxrun-catalog.ts:15`), and an arm64 Spark recipe is the default case. The
   entrypoint-clearing overlay pattern applies if the published image sets one.
3. **Readiness probe: probably free.** Because it is the vLLM server, `/metrics` and `/v1/models`
   should both answer. Confirm rather than assume.

---

## §9 License — settle before building

**Qwen Research License: non-commercial purposes only**, plus a "Built with Qwen" attribution
requirement in any product built on it. Every other model in this catalog is Apache-2.0 or MIT. If
chat3d is the intended consumer, this is a product decision and it should be taken before the build
work, not after.

---

## §10 What to measure first

Ordered so the cheapest thing that could kill it runs first.

1. **Pull `vllm/vllm-omni:qwen-image21-arm64-cu130` on one Spark and generate one 1024² image.**
   Settles §3's cuDNN question empirically in minutes. If `CUDNN_ATTN` faults, upgrade cuDNN to
   ≥9.24 and repeat.
2. **Generate at 2048².** The number nobody has. This is the decision.
3. **Check output correctness, not just that it ran** — PR #7759 has open review findings on
   KV-cache aliasing and condition-latent ordering, which are silent-wrong-output classes, not
   crashes. Verify RGBA transparency actually round-trips.
4. Only then: recipe, gateway path, and whether it earns a Spark.

---

## §11 One cross-finding worth keeping

`[verified]` **pytorch/pytorch#192209 (open)**: `MapAllocator`'s hardcoded `PROT_WRITE` OOM-kills
checkpoint loads on unified-memory systems, naming GB10 specifically. That is a plausible contributor
to our own recorded incident where a download OOM-killed a serving replica (#92). Unrelated to this
model; worth chasing separately.

Also `[verified]`, and worth internalizing: PyTorch **deliberately** ships no sm_121 cubin
(pytorch#178891 closed with *"We do not need to compile PyTorch for 12.1 as it is compatible with
12.0"*). `get_arch_list()` on our Spark returns `['sm_80','sm_90','sm_100','sm_110','sm_120']` — and
it works. Relatedly: all 18 NGC PyTorch release notes and NVIDIA's Blackwell Compatibility Guide
mention GB10/sm_121 **zero** times. In NVIDIA documentation, "Blackwell" means GB200.

---

## §A Appendix: where the sources disagreed

**§A.1 — arm64 images.** The packaging agent reported official multi-arch amd64+arm64 tags; the
sm_121 agent independently re-checked and reported "no ARM64 image", citing #195/#1571 as open.
**I resolved this myself against the Docker Hub v2 API** and the tags table in §2 is that query's
output: multi-arch since at least v0.28.0 (2026-08-31), with model-specific arm64 tags pushed
2026-09-20. The open issues are stale and do not track reality. *Where a delegated claim and a
registry query disagree, the registry wins.*

**§A.2 — GB300 vs GB200.** The vLLM recipes YAML reports the measurement on a GB300; PR #7759's own
prose says GB200. Both are Grace, so the aarch64 point stands either way, and §2's GB10 recipe makes
the inference unnecessary.

**§A.3 — GB300 ⇒ aarch64 was an inference**, via NVIDIA's "144 Arm Neoverse V2 cores" for Grace; no
`uname -m` is published. Non-load-bearing given §2.

**§A.4 — one delegated performance claim** (a "25–30% throughput penalty") was flagged by the agent
that produced it as probably wrong, and is excluded here.
