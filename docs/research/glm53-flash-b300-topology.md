# 2 × 8 NVIDIA B300: does a frozen-base LoRA of GLM-5.3-Flash run, and at what EP/CP?

> Supersedes the verdict in
> [`docs/research/glm53-flash-topology-reconcile.md`](https://github.com/kreuzhofer/dgx-manager/blob/research/glm53-flash-topology-reconcile/docs/research/glm53-flash-topology-reconcile.md)
> (branch `research/glm53-flash-topology-reconcile`), which answered "one 8×H200 node in
> `eu-north1`" for [#71](https://github.com/kreuzhofer/dgx-manager/issues/71). The hardware
> actually available is **2 nodes × 8 = 16 NVIDIA B300**, so that answer is moot. Part of the
> [GLM-5.3-Flash wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/65).
>
> **Read on 2026-09-26. Primary sources only** — NVIDIA documentation on `docs.nvidia.com` /
> `developer.nvidia.com/cuda-gpus` / NVIDIA datasheet PDFs, `NVIDIA-NeMo/Automodel` `main`
> at **`1358302c`** (2026-09-25), `fla-org/flash-linear-attention` `main` at **`954438d1`**
> (2026-09-21), `deepseek-ai/FlashMLA` `main` at **`ba89a346`** (2026-09-15),
> `deepseek-ai/DeepEP` at NeMo's pin `10d4dd73`, `pytorch/pytorch` at tag `v2.13.0`, the
> GitHub issue trackers of those repos, PyPI, and `docs.nebius.com` / `nebius.com`.
> No blog, no news article, no secondary write-up.
>
> **Every number is labelled.**
> **measured** = read off the live cluster, off a vendor's published result, or off an API/source file;
> **derived** = arithmetic over measured values, arithmetic shown;
> **extrapolated** = a model with a stated assumption in it;
> **unknown** = said so plainly.
>
> Facts carried forward from #66/#67/#69/#71 without re-derivation are marked *(carried)*.

---

## VERDICT

**Yes. 2 nodes × 8 B300 runs it, at `ep_size: 16`, `tp_size: 1`, `pp_size: 1`, `cp_size: 1`
up to ~8k and `cp_size: 2` at 16k. Memory is a non-issue: 36.5 GiB/GPU of frozen weights
against a measured 268.6 GiB. But do the first probe on ONE node at `ep_size: 8` — it is also
comfortable (73.1 GiB of 268.6) and it deletes the entire inter-node surface from the first
failure.**

**The single biggest risk is not memory, not topology, and not FlashMLA. It is that
Flash-Linear-Attention's Triton kernels are known to be silently miscompiled on sm_103 —
the exact compute capability of a B300 — in exactly the Triton versions PyTorch 2.13.0
pins, and the KDA path that carries 34 of the model's 45 layers is *only partially* guarded
against it.**

### The three things that changed versus #71

| | #71's answer (8×H200) | This answer (16×B300) |
|---|---|---|
| **Memory** | the load-bearing constraint — 73.1 GiB/GPU of 140.4, and a **derived ~110 GiB/GPU load peak** that was called *"the single most likely place a one-node run fails"* | **gone.** 36.5 GiB/GPU at EP16 of a **measured** 268.6, load peak 54.8. Even single-node EP8 is 73.1 of 268.6 with a 109.6 GiB load peak — 41% fill. |
| **Kernels** | a non-issue; H200 is sm_90, the most-exercised training target in the ecosystem | **the whole problem.** sm_103 is new silicon with an active, open, *silent-corruption* bug class in the KDA kernel path. |
| **FlashMLA / cuDNN** | available (sm_90a), worth +13.26% | **also available** — and the naive "sm_100f ≠ sm_103" reading is wrong. See §2. |

### What the probe must do first, before anything else

Run a **bitwise determinism check on `fla.ops.kda.chunk_kda` forward+backward at production
shapes on the B300**, before believing any loss curve. The failure mode in this bug class is
not a crash — it is identical inputs returning different outputs on every call, and NaN
gradients that propagate without raising. §3.6 gives the script. If that check fails, the run
is not "slow", it is *wrong*, and no amount of wall-clock measurement will tell you.

### The shape, in one table

| | |
|---|---|
| Hardware | 2 × (8 × NVIDIA B300 SXM6 AC), **sm_103**, 275,040 MiB/GPU, NVSwitch intra-node, 800 Gb/s ConnectX-8 per GPU inter-node *(measured on the cluster)* |
| Topology | `tp_size: 1`, `pp_size: 1`, `ep_size: 16`, `cp_size: 1` (≤8k) / `2` (16k) / `8` (32k, dp2) |
| Weights/GPU | **36.5 GiB** at EP16, **73.1 GiB** at EP8-on-8 *(derived)* |
| Load peak/GPU | **54.8 GiB** at EP16, **109.6 GiB** at EP8-on-8 *(derived)* |
| First probe | **1 node, EP8, `dispatcher: torch`, `attn: sdpa`, CP1, packed 2k** |
| Mandatory gate | KDA bitwise-determinism check (§3.6) |
| Highest-value single mitigation | **Triton 3.8.0** (released 2026-08-28) instead of the `triton==3.7.1` that torch 2.13.0 pins |
| Second mitigation | FLA from `main` (0.6.0), not PyPI 0.5.2 — 0.5.2 is missing two Blackwell fixes |
| Still unknown | **s/step, tokens/s, MFU.** Nothing on this map has ever measured them. #72 is what settles it. |

---

## §1 — What a B300 is

The coordinator supplied the cluster's own `nvidia-smi` / `torch` output, which is ground
truth and is used in preference to any document. This section exists to confirm it against
NVIDIA's own sources and to name the parts of it that documents get wrong.

### 1.1 Compute capability: **10.3 / sm_103**. Confirmed three ways.

**measured on the cluster:** `torch.cuda.get_device_capability()` → `(10, 3)`.

**measured, NVIDIA's own product→CC table** — [developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus),
which the CUDA Programming Guide names as the authority (*"The CUDA GPU Compute Capability page
provides a comprehensive mapping from NVIDIA GPU models to their compute capability"*,
[compute-capabilities.html](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html) §5.1.1):

| Compute Capability | Data Center |
|---|---|
| **10.3** | **NVIDIA GB300, NVIDIA B300** |
| 10.0 | NVIDIA GB200, NVIDIA B200 |
| 12.1 | NVIDIA GB10 (DGX Spark) |
| 9.0 | NVIDIA GH200, NVIDIA H200, NVIDIA H100 |

**measured, NVIDIA's own source repo** — [NVIDIA/cutlass `README.md`](https://github.com/NVIDIA/cutlass/blob/main/README.md),
columns *GPU | CUDA Compute Capability | Minimum CUDA Toolkit Required by CUTLASS-3*:

```
|NVIDIA B200 Tensor Core GPU            |10.0|12.8|
|NVIDIA B300 Tensor Core GPU            |10.3|13.0|
|NVIDIA DGX Spark                       |12.1|13.0|
```

Note the third column: **B300 needs CUDA Toolkit ≥ 13.0 for CUTLASS-3**, where B200 needs only
12.8. The cluster's system `nvcc` is **12.9.86** *(measured)*. That matters in §2.4.

Corroborating, from the [CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
(13.4 Update 1), verbatim, which treat "sm_103" and "B300/GB300" as the same thing:

> *"Improved performance on Blackwell (`sm_100` and `sm_103`) via heuristics tuning for FP32
> GEMMs…"* · *"Removed unnecessary overhead related to loading kernels on GPUs with compute
> capability 10.3."* · *"Fixed an issue in cublasLtMatmul that could lead to incorrect results
> for NVFP4 precision **on B300 and GB300 GPUs** when the m dimension was not a multiple of 64."*

The nvcc driver doc's own warning is worth keeping in view
([nvcc §5.1](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)), verbatim:

> *"While GPU generations are often referred to by product names such as Hopper or Blackwell,
> these don't always have a direct correspondence to major compute capability version."*

### 1.2 HBM: the docs disagree, and the measurement settles it

**measured on the cluster: 275,040 MiB per GPU.**

That is `275040 × 2²⁰ = 288,358,563,840 bytes` = **268.59 GiB** = **288.4 GB decimal**
*(derived, unit conversion only)*.

| Source | States | Matches the cluster? |
|---|---|---|
| **Cluster `nvidia-smi`** | **275,040 MiB = 268.6 GiB = 288.4 GB** | — (this is the measurement) |
| `docs.nebius.com/compute/virtual-machines/types` | `288 GB HBM3e \| 10 TB/s` | **Yes**, exactly, in decimal GB |
| NVIDIA Blackwell Ultra Datasheet, **HGX B300** column | `270 GB HBM3E \| 7.7 TB/s` | No (270 GB = 251.5 GiB) |
| NVIDIA Blackwell Ultra Datasheet, **GB300 NVL72** column | `279 GB HBM3E \| 8 TB/s` | No (279 GB = 259.8 GiB) |
| `nebius.com/compute/b300` (FAQ) | *"Each GPU carries 270 GB of HBM3e memory"* | No |

**I cannot reconcile NVIDIA's own 270 GB HGX-B300 figure with a part that reports 288 GB.**
The datasheet states both 270 (HGX B300) and 279 (GB300 NVL72) in the *same table*, so the
distinction is deliberate on NVIDIA's side, and this cluster is neither. **Plan against the
measured 268.6 GiB.** Note that #69's earlier "288 GB" for `gpu-b300-sxm` was right and
should not have been talked down to 270.

One trap worth recording so nobody re-imports it: `nvidia.com/en-us/data-center/hgx/` does
show **`288 GB HBM4 | 22 TB/s`** — but that is the **HGX Rubin NVL8** column (NVLink 6th gen),
not B300. The coincidence of "288" is an accident.

### 1.3 Node shape and the Nebius SKU

**measured on the cluster:** 8 GPUs/node, every GPU pair `NV18` in one NVSwitch domain, one
dedicated ConnectX-8 (MT4126) per GPU at 800 Gb/s `PIX` to its GPU and GPUDirect-capable,
NCCL verified across both nodes with `NCCL_IB_DISABLE=0` / `NCCL_NET_GDR_LEVEL=5`,
~2.43 TiB host RAM, 192 logical CPUs (2 × Xeon 6776P), driver 580.159.04,
torch 2.13.0+cu130, system nvcc 12.9.86, Slurm 25.11.3 on Nebius "soperator",
**`/tmp` is node-local — shared scratch must be `/mnt/data`**.

The matching Nebius catalogue entry, **measured** from `docs.nebius.com` on 2026-09-26 — the
coordinator could not determine the SKU on the machine, so this is offered as *consistent with*
the measurements, not as an identification:

| | |
|---|---|
| Platform ID | `gpu-b300-sxm` — *"NVIDIA® B300 NVLink with Intel Granite Rapids"*, host CPU **Intel Xeon 6776P** |
| Preset | `8gpu-192vcpu-2768gb` (8 GPU / 192 vCPU / **2768 GiB** RAM) — matches the measured 192 CPUs and ~2.43 TiB |
| Regions | `uk-south1`, `eu-west2`, `us-north1`; IB fabrics `uk-south1-a`, `eu-west2-a`, `us-north1-a` |
| Cluster-compatible | yes, the 8-GPU preset only (*"Other presets and platforms are not compatible with GPU clusters"*); the 1-GPU preset is not |
| Self-service | yes — GPU-cluster prerequisites are verbatim *"If you use the web console, you don't need to complete any prerequisites."* `nebius.com/compute/b300`: *"Is HGX B300 available in self-service? **Yes.** … (Note: GB300 NVL72 is currently available through the sales team only.)"* |
| Default quota | **32 B300 GPUs** (= 4 nodes) and **5 GPU clusters** per region |
| Local NVMe | **6 × 3.84 TB**, ephemeral, on the 8-GPU preset in all three regions. *"When the VM is stopped or deleted, the data is lost."* |
| Boot image | exactly one: `ubuntu24.04-cuda13.0` |
| Price | `$7.85`/GPU-hour on-demand — **`nebius.com/prices` publishes `$9.50` effective 2026-10-01**, which `docs.nebius.com` does not yet show. Cost is not a decision variable on this map *(carried, #69)*, but 8 GPUs × $9.50 = $76/hr is worth knowing. |

`gpu-b300-sxm` is **HGX B300 on x86_64**, not GB300/Grace/arm64 — settled by the Intel Xeon
6776P host CPU (which is also what NVIDIA's own DGX B300 page lists), by Nebius modelling
GB300 as a separate `nvl-instance-groups` API resource with `type: GB200|GB300`, and by the
measured `2x Xeon 6776P`. **The arm64 container question #49 worried about does not arise.**

---

## §2 — FlashMLA on B300: covered, and the naive reading was wrong

### 2.1 The arch-suffix rule, from NVIDIA, verbatim

This is the crux the coordinator flagged, and it is the same shape as the two errors already
corrected on this map. **`sm_100f` does cover `sm_103`.** Three NVIDIA primary sources say so.

**CUDA Programming Guide, Appendix "Compute Capabilities" §5.1.2.3 "Feature Set Compiler
Targets"** ([docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html)),
verbatim — this is the load-bearing sentence:

> *"The `compute_100f` family-specific compilation target allows the use of the subset of
> architecture-specific features that are common across the GPU family. This target will only
> be compatible with devices that are part of the GPU family. **In this example, it is
> compatible with devices of Compute Capability 10.0, 10.3, and 10.7.**"*

and contrastingly:

> *"The `compute_100a` architecture-specific compilation target allows the use of the complete
> set of architecture-specific features in Compute Capability 10.0 devices. **This target will
> only be compatible with devices of Compute Capability 10.0 and no others.**"*

and the baseline:

> *"The `compute_100` compilation target does not allow the use of architecture-specific
> features. This target will be compatible with all devices of compute capability 10.0 and later."*

**Table 28 "Family-Specific Compatibility"**, verbatim:

| Compilation Target | Compatible with Compute Capability |
|---|---|
| `compute_100f` | **10.0, 10.3, 10.7** |
| `compute_103f` | 10.3, 10.7 |
| `compute_120f` | 12.0, 12.1 |

**PTX ISA 9.4 §11.1.2 "PTX Module Directives: `.target`"**
([docs.nvidia.com/cuda/parallel-thread-execution](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#ptx-module-directives-target)),
verbatim:

> *"Target architectures with suffix "`a`", such as `sm_90a`, include architecture-specific
> features that are supported on the specified architecture only, hence such targets do not
> follow the onion layer model. Therefore, PTX code generated for such targets cannot be run on
> later generation devices."*
>
> *"Target architectures with suffix "`f`", such as `sm_100f`, include family-specific features
> that are supported only within the same architecture family. Therefore, **PTX code generated
> for such targets can run only on later generation devices in the same family.**"*

**Table 70 Architecture Families**, verbatim:

| Family | Target SM architectures included |
|---|---|
| `sm_10x` family | `sm_100f, sm_103f, sm_107f`, future targets in `sm_10x` family |
| `sm_12x` family | `sm_120f, sm_121f`, future targets in `sm_12x` family |

**nvcc driver doc §4.2.9.1.14 (`ptxas --gpu-name`)**
([docs.nvidia.com/cuda/cuda-compiler-driver-nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)),
verbatim:

> *"PTX for `.target sm_XY` can be compiled to all GPU targets sm_MN, sm_MNa, SM_MNf where
> MN >= XY. PTX for `.target sm_XYf` can be compiled to GPU targets sm_XZ, sm_XZf, sm_XZa where
> Z >= Y and sm_XY and sm_XZ belong in same family. PTX with `.target sm_XYa` can only be
> compiled to GPU target sm_XYa."*

**The general rule, stated for reuse:**

| target | runs on sm_100 (B200) | runs on sm_103 (B300) | runs on sm_121 (GB10) |
|---|:---:|:---:|:---:|
| `sm_100` (baseline) | ✓ | ✓ | ✗ (different major) |
| `sm_100f` (family) | ✓ | **✓** | ✗ |
| `sm_100a` (arch) | ✓ | **✗** | ✗ |
| `sm_103a` | ✗ | ✓ | ✗ |

**The dangerous case is `a`, not `f`.** A wheel shipping *only* `sm_100a` will not load on a
B300 — that is the silent-at-build, hard-fail-at-load trap to check for.

### 2.2 FlashMLA today emits `sm_103a` outright

`deepseek-ai/FlashMLA` `main` at **`ba89a346`**, `setup.py` **L41-48**, verbatim:

```python
    arch_flags = []
    if not DISABLE_SM100:
        # We use architecture-specific (sm_100a / sm_103a) targets instead of the family-specific one (sm_100f) for better SASS code generation
        arch_flags.extend(["-gencode", "arch=compute_100a,code=sm_100a"])
        arch_flags.extend(["-gencode", "arch=compute_103a,code=sm_103a"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
    return arch_flags
```

**measured.** Note that the code comment independently confirms the §2.1 reading: `sm_100f`
is described as the alternative they moved *away from*, "for better SASS code generation" —
a performance choice, not an enablement one. This landed silently inside commit `07a108985`
(*"Add kernels for DeepSeek v4.1 (#221)"*, **2026-09-10**) — six days after the 2026-09-04
re-check that recorded `sm_90a`/`sm_100f`, and un-announced in the title, README or any
changelog. There is no `csrc/sm103/` directory; `sm_103a` recompiles the same
`csrc/kernels/sm100/**` sources a second time, with `KERUTILS_ENABLE_SM103A` gated on
`__CUDA_ARCH__ >= 1030` (`csrc/kerutils/include/kerutils/device/common.h` L64-66).

FlashMLA's README has **not** caught up — L67 still says *"SM90 / SM100 (See the support matrix
below)"* and never mentions SM103 or B300. Do not read the README as the authority here; read
`setup.py`.

### 2.3 But NeMo pins an older FlashMLA — which is still fine

The coverage doc `docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx` **L98-103** pins, verbatim:

```bash
git clone --recursive https://github.com/deepseek-ai/FlashMLA.git /tmp/FlashMLA
git -C /tmp/FlashMLA checkout b7643bd54521f563b839b98289b5cd048c062ba2
```

That commit is **2026-04-30** *(measured, GitHub API)* and its `setup.py` L41-46 emits:

```python
    arch_flags = []
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100f,code=sm_100f"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
```

**`sm_100f` + `sm_90a`.** Per §2.1 Table 28 that **runs on sm_103**, at the family-common
feature subset. **So the NeMo-pinned FlashMLA covers B300 as-is.** Upgrading to `main` is a
*performance* option, not a requirement, and it is not free: `csrc/` was reorganised
(`csrc/sm100/…` → `csrc/kernels/sm100/…`) and `docker/common/verify_native_runtime.py` **L63**
hard-asserts the version string:

```python
    if flash_mla_version != "1.0.0+b7643bd":
        raise RuntimeError(f"FlashMLA 1.0.0+b7643bd is required, found {flash_mla_version}")
```

**Recommendation: build the pin. Do not chase `main` for the first probe.**

### 2.4 The one build-time constraint: nvcc 12.9 is exactly at the floor

FlashMLA `setup.py` L38-39 (identical in both revisions), verbatim:

```python
    if major < 12 or (major == 12 and minor <= 8):
        assert DISABLE_SM100, "sm100 compilation for Flash MLA requires NVCC 12.9 or higher. ..."
```

The cluster's nvcc is **12.9.86** *(measured)* → `12 == 12 and 9 <= 8` is False → the assert
does not fire, and the SM100 group compiles. **It passes, by one minor version.**

Counter-flag: CUTLASS's own README states **CUDA Toolkit 13.0** as the minimum for B300 under
CUTLASS-3, and FlashMLA vendors CUTLASS as a submodule. The `sm_100f` pin does not target
sm_103 in CUTLASS terms so 12.9 should suffice; **building `main`'s `sm_103a` path may need a
CUDA 13 toolkit.** Nebius's only B300 boot image is `ubuntu24.04-cuda13.0`, so a CUDA 13
toolkit is presumably obtainable — but the *measured* system nvcc on this cluster is 12.9.86,
and that is what a source build will pick up unless `CUDA_HOME` is pointed elsewhere.
**Unverified: whether a CUDA 13 nvcc is present on these nodes.**

### 2.5 The fallback, and what it costs

If FlashMLA or cuDNN Frontend is missing, NeMo **raises, it does not silently degrade** —
`nemo_automodel/components/models/glm5_next/layers.py` **L750-754**, verbatim:

```python
        if not is_cudnn_sparse_attention_available():
            raise RuntimeError(
                "backend.attn='cudnn' requires the optional cuDNN sparse-attention "
                "and FlashMLA runtimes, but they are unavailable in this environment."
            )
```

The fallback is a config change you make by hand — coverage doc L114-115, verbatim: *"If
FlashMLA or cuDNN Frontend is unavailable, change `model.backend.attn` to `sdpa`; KDA still
requires the FLA dependency."*

**The price of that fallback is NVIDIA's own measured 13.26%**, coverage doc L155-161
(**measured**, EP72/CP2 on 72 H100, MedPix packed-2k, steps 10-99):

| Backend | Mean TPS | Median TPS | Mean / Peak Memory | Training Loop |
|---|---:|---:|---:|---:|
| SDPA | 8,378.15 | 8,417.33 | 57.160 / 57.68 GiB | 37:15 |
| cuDNN | 9,489.04 | 9,538.12 | 57.185 / 57.78 GiB | 33:18 |

> *"The cuDNN path improved mean throughput by 13.26% and median throughput by 13.32%"*

**That 13.26% is on H100. It has never been measured on Blackwell and should not be assumed
to transfer.** It applies only to the **11** DSA layers; the 34 KDA layers are on FLA
regardless of `backend.attn`.

### 2.6 Nothing in NeMo gates against Blackwell

The only device-capability check on the GLM-5.3-Flash cuDNN path is
`nemo_automodel/components/models/common/cudnn_sparse_attention.py` **L74-77**, verbatim:

```python
    major, minor = torch.cuda.get_device_capability(device)
    if major < 9:
        raise RuntimeError(f"{operation} requires SM90 or later, got SM{major}{minor}.")
    return major, minor
```

**No upper bound.** sm_103 has major 10; it passes. The one arch-dependent branch,
`_padded_head_count` L109-122, splits on `major >= 10` — sm_103 takes the same branch as
sm_100. A repo-wide grep for `get_device_capability` / `major <` / `major >=` found nothing
that would block sm_103 on this model's path. (`minimax_m3_vl` has a `require_sm100` gate;
different model, irrelevant here.)

Two peripheral B300 gaps, neither fatal:

- `nemo_automodel/_transformers/mfu.py` **L42-62** `_DEVICE_FLOPS` has `GB200` and `B200` but
  **no B300 entry**; L92-93 docstring: *"Returns `float("inf")` for unknown devices."* MFU
  reporting will be meaningless unless you set the documented override
  (`docs/guides/configuration.mdx` L193-194).
- `docker/Dockerfile` builds Transformer Engine (L106 `TE_CUDA_ARCHS="80;90;100;120"`),
  DeepEP/HybridEP (L161 `TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0"`) and torchao (L202, same) with
  **no `10.3` and no `+PTX`**. Those produce baseline `sm_100` cubins, which per §2.1 run on
  sm_103 — but adding `10.3` to those three lines and rebuilding is the cheap, zero-risk move.

**Also relevant that NeMo already knows about sm_103 toolchain skew elsewhere** —
`nemo_automodel/components/models/minimax_m3_vl/msa_bindings.py` **L125-128**, verbatim:

```python
# sm_103a needs nvcc 12.9; MSA hard-codes both SM100 targets in a joined string with no module-level
# constant to override (jit.py:200-201), so an older toolkit fails the whole compilation.
_SM103A_TARGET = "-gencode=arch=compute_103a,code=sm_103a"
_MIN_SM103A_NVCC = (12, 9)
```

**Verdict on Q2: FlashMLA covers B300 both as NeMo pins it (`sm_100f`, by family
compatibility) and upstream (`sm_103a`, natively). There is no fallback penalty to pay. The
prior "sm_90a/sm_100f therefore not covered" reading was wrong on both counts.**

### 2.7 Second flag: does torch 2.13.0 ship sm_103 SASS? No — and it does not need to

`pytorch/pytorch` `main`, `.ci/wheel/linux/build_env_setup.py` **L76-98**, verbatim:

```python
TORCH_CUDA_ARCH_LIST_TABLE: dict[str, dict[str, set[int]]] = {
    ...
    "13.0": {
        "x86_64": {75, 80, 86, 90, 100, 120},
        "aarch64": {80, 90, 100, 110, 120},
    },
    ...
}

# Architectures we additionally emit PTX for on nightly/dev builds
# (forward-compat for newer GPUs). Release/RC wheels ship SASS-only to keep
# libtorch_cuda.so size down; see _ptx_arches().
_PTX_ARCHES: set[int] = {120}
```

with `_ptx_arches()` returning `set()` on a release build. **So a release torch wheel for
CUDA 13.0 / x86_64 contains SASS for 7.5/8.0/8.6/9.0/10.0/12.0 and no PTX at all — no sm_103
SASS, and nothing to JIT from.** *(measured from `main`; the file does not exist at the
`v2.13.0` tag, so this is the current build recipe rather than the exact one used for 2.13.0.
Confirm on the cluster with the one-liner in §8.)*

**The consequence is benign, and it is the opposite of what was feared.** There is no
PTX-JIT startup cost, because there is no PTX. What runs on the B300 is the **baseline
`sm_100` SASS**, which per §2.1 is *"compatible with all devices of compute capability 10.0
and later"*, and which nvcc §5.1 backs with the sm_80→sm_86/sm_89 precedent. Torch agrees
explicitly — `torch/cuda/__init__.py` **L436-441** at tag `v2.13.0`, verbatim:

```python
    supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]
    for idx in range(device_count()):
        cap_major, cap_minor = get_device_capability(idx)
        # NVIDIA GPU compute architectures are backward compatible within major version
        supported = any(sm // 10 == cap_major for sm in supported_sm)
```

`100 // 10 == 10 == cap_major` → **supported, no warning emitted.**

**What it does mean:** torch's CUDA kernels on this cluster are sm_100-tuned SASS, never
sm_103-tuned, and PyTorch's release CI has never run on sm_103. That is a *performance and
coverage* caveat, not a correctness one — and it is a minor sibling of the real problem in §3.

---

## §3 — THE BIG ONE: FLA / KDA kernels on sm_103

**Read this section before provisioning anything. 34 of 45 layers go through it.**

### 3.1 The claim being chased, and where it actually leads

The 2026-09-04 re-check recorded, from FLA issue #913, a report of *"roughly 3-5x slower per
training step"* on B200/B300 and filed it as a *performance* datapoint about a torch fallback.
Followed to its source, **that quote is the footnote, not the finding.** The finding is why
they were on a torch fallback at all.

The full quote, [`fla-org/flash-linear-attention#913`](https://github.com/fla-org/flash-linear-attention/issues/913),
`@tommyliautaud`, 2026-06-04, verbatim (**measured**, third-party, on real hardware):

> *"For Qwen3.5/Qwen3.6 MoE training, we currently patch the model to swap FLA kernels for the
> PyTorch reference implementations: […]* `"""Swap FLA's Triton kernels for torch reference
> implementations. **Tested on B200 (sm_100) and B300 (sm_103)**; converges through 12K+
> steps."""` *[…] This converges to expected loss trajectories on B200 and B300, but the torch
> fallback is approximately **3-5x slower per training step** than the FLA fused kernel path on
> H200. For 12K-step training runs of a 35B MoE, this is roughly the difference between about
> 2 days on H200 and 5-6 days on Blackwell fallback."*

They are on the torch reference **because the Triton kernels return wrong answers on
Blackwell**, silently. Same issue, same author, same day, verbatim:

> *"1. `chunk_gated_delta_rule` backward often completes but returns **silent NaN gradients**.
> 2. `fused_recurrent_gated_delta_rule` crashes later during `.backward()`, but
> `compute-sanitizer` shows the first detected memory error is actually an out-of-bounds read
> in the forward kernel."*
>
> *"The supported stack, torch 2.12 + Triton 3.7, is especially risky for training reliability
> because `chunk_gated_delta_rule` backward completes but returns NaN gradients silently.
> **Without explicit finite-gradient checks, a training run can diverge without a Python
> exception.**"*

Reproduced across torch 2.11/2.12, Triton 3.5.1/3.6.0/3.7.0, FLA 0.4.2/0.5.0/0.5.1/HEAD,
drivers 570.172/580.126 — **byte-identical NaN counts across all of them**. Not fixed by a
version bump in any direction. Issue #913 was **auto-closed as stale by a bot on 2026-08-16**;
no maintainer ever answered it.

**So: the "3-5×" number is real, it is a Blackwell-datacentre datapoint, and it explicitly
names B300/sm_103 — but it is the cost of the *workaround*, not the cost of the hardware. The
finding that matters is the correctness bug it works around.**

### 3.2 The root cause, and it is sm_103-specific and *wider* than sm_100

[`triton-lang/triton#10590`](https://github.com/triton-lang/triton/issues/10590), opened
2026-06-12, closed 2026-06-15. Environment, verbatim (**measured**):

> *"GPU | **NVIDIA B300 SXM6 AC — sm_103 / capability (10, 3), 148 SMs** · Driver | 580.126.09
> (CUDA 13.0) · triton | **3.6.0** (torch 2.10.0+cu130) and **3.7.0** (torch 2.12.0+cu130) —
> both reproduce · dtype | bf16 inputs, fp32 accumulators"*

> *"A Triton kernel that carries an fp32 accumulator state across a sequential `for` loop of
> `tl.dot` updates (the standard "chunked linear-attention recurrence" pattern) produces
> **bitwise-different outputs on every invocation with identical inputs** on NVIDIA B300
> (sm_103). […] The kernel has no cross-CTA communication (each program writes a disjoint
> output slice), so this is a compiler/scheduling race, not a kernel logic error."*

**The sm_103 failure envelope is strictly larger than sm_100's** — the issue's own comparison
table, verbatim:

| | #9871 (sm_100, GB200/B200) | This report (sm_103, B300) |
|---|---|---|
| `BV=32, w4, s2/s3` | FAIL | FAIL |
| `BV=64, w4, s2` | **pass** (BV≤32 only) | **FAIL** |
| `BV=32, w8, s2/s3` | **pass** (verified formula: w4 only) | **FAIL** |
| Grid threshold | ≥ ~160 CTAs | FAIL from 128–192 CTAs (reliable at ≥ 256) |
| Status after Triton 3.7.0 | fixed | **still failing (verified)** |

> *"Failures begin once the grid exceeds roughly one wave on this part (148 SMs)"*

**This matters more than any other single sentence in this document:** every mitigation FLA
has shipped was calibrated on the *narrower* sm_100 envelope, and B300 is the wider one.

**The fix exists only on Triton `main` / 3.8**, and nobody knows what fixed it. Closing
comment, 2026-06-15, verbatim — the reporter re-running the same matrix:

> *"Ah interesting, so confirming this seems to be fixed on main: `sm=(10, 3) dev=NVIDIA B300
> SXM6 AC triton=3.8.0 torch=2.12.0+cu130 … FAILING CONFIGS: none`"*

and, three weeks later, asked what changed:

> *"@michaelroyzen … can you tell me what changed with main branch that fixed your bug?"* →
> *"Not sure, it just seems to work now."*

**An unexplained disappearance is not a fix. Treat it as latent.**

### 3.3 **torch 2.13.0 pins `triton==3.7.1` — the broken band**

*(measured)* `pytorch/pytorch` tag `v2.13.0`:

- `.ci/docker/triton_version.txt` → `3.7.1`
- PyPI `torch 2.13.0` `requires_dist` → `triton==3.7.1; platform_system == "Linux" and python_version < "3.15"`

**Triton 3.8.0 was released on PyPI on 2026-08-28** *(measured)* — but torch pins 3.7.1
**exactly**, so getting onto 3.8 means overriding a hard `==` pin, which PyTorch neither tests
nor supports.

**This is the single highest-leverage decision on the whole map**, and it is a one-line
install choice:

| Stack | sm_103 `tl.dot`-in-loop miscompile |
|---|---|
| torch 2.13.0 default → **Triton 3.7.1** | **present** (3.7.0 verified failing on B300; 3.7.1 not separately tested by anyone) |
| Force **Triton 3.8.0** | verified clean on a B300 by the #10590 reporter, and independently by #1228: *"On a Triton 3.8 build the standalone matrix above is fully deterministic and our production runs show zero flips with or without the gate."* |

**unknown:** whether Triton 3.8.0 is API-compatible with torch 2.13.0 and with FLA 0.6.0 in
practice. That is a 20-minute experiment on the cluster and it should be the second thing
anyone does.

### 3.4 What FLA has actually guarded — and the KDA path is only partly covered

FLA's response to this bug class has been **per-kernel `num_warps` caps gated on
`IS_NVIDIA_BLACKWELL`**, added one at a time as each kernel was reported. `IS_NVIDIA_BLACKWELL`
is `torch.cuda.get_device_capability()[0] in (10, 12)` (`fla/utils/_device.py` L162), so where
a gate exists **it does apply to sm_103**. The problem is coverage, not detection.

**Guarded (and KDA benefits):**

`fla/ops/common/chunk_delta_h.py` **L26-35**, verbatim — the fix for
[#945](https://github.com/fla-org/flash-linear-attention/issues/945)
(*"`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` produces non-deterministic output on
**NVIDIA B300 (sm_103)** when autotune selects `num_warps=4`"*, PR #953):

```python
# TODO: Triton mainline fixes a Blackwell tl.dot recurrence race.
# Keep this kernel on num_warps=2 for Blackwell until Triton 3.8 is released
# and we re-validate the wider config space.
if IS_NVIDIA_BLACKWELL:
    GATED_DELTA_RULE_FWD_H_NUM_WARPS = [2]
```

KDA imports this kernel directly (`fla/ops/kda/chunk_fwd.py` L10, `chunk_bwd.py` L13), so
**this one protects the KDA chunked-state forward.** Good.

`fla/ops/kda/chunk_bwd.py` **L132-133** — the fix for
[#727](https://github.com/fla-org/flash-linear-attention/issues/727)
(*"CUDA Illegal Memory Access in `chunk_kda_bwd` on NVIDIA B200"*, PR #1109, merged
**2026-08-09**):

```python
        if not (IS_NVIDIA_HOPPER and BK == 32 and num_warps == 4)
        if not (IS_NVIDIA_SM100 and BK == 32 and num_warps != 2)
```

**Note two things.** First, it prunes only `BK == 32` — at `BK == 64` the 4- and 8-warp
configs survive, and `BV=64/w4/s2` is *precisely* the config #10590 showed fails on sm_103 and
passes on sm_100. Second, **2026-08-09 is after FLA 0.5.2's release date of 2026-07-27**
*(measured, GitHub releases + PyPI)* — **so this KDA fix is in no released FLA version.**

**Ungated, in the KDA path, with the same `tl.dot`-in-a-loop-into-an-fp32-accumulator shape
and 4/8 warps still in the Blackwell config space** *(all measured from FLA `main` `954438d1`)*:

| kernel | file:line | autotune space on Blackwell | role in KDA |
|---|---|---|---|
| `chunk_gla_fwd_kernel_o` | `fla/ops/gla/chunk.py` L336-341 | `BK{32,64} × BV{64,128} × warps{2,4,8} × stages{2,3,4}` | **the KDA forward output kernel**, reached via `chunk_gla_fwd_o_gk` (`kda/chunk_fwd.py` L13) |
| `chunk_kda_bwd_kernel_dAv` | `fla/ops/kda/chunk_bwd.py` L36-38 | `warps{2,4,8} × stages{2,3,4}` | KDA backward dA/dv |
| `chunk_kda_bwd_kernel_intra` | `fla/ops/kda/chunk_intra.py` L392-394 | `warps{1,2,4,8} × stages{2,3,4}` | KDA backward intra-chunk |
| `chunk_kda_fwd_kernel_intra_sub_chunk` | `fla/ops/kda/chunk_intra.py` L684-686 | `warps{1,2,4,8} × stages{2,3,4}` | KDA forward intra-chunk |
| `recompute_w_u_fwd_kda_kernel` | `fla/ops/kda/wy_fast.py` L26-28 | `warps{2,4,8} × stages{2,3,4}` | KDA WY recompute |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | `fla/ops/common/chunk_delta_h.py` (bwd block) | `BV{32,64} × warps{2,4} × stages{2,3,4}` | **KDA backward chunked state** — `BV=64/w4/s2` is the sm_103-only failing config |

**The single most telling item is the first row.** The fix for
[#1228](https://github.com/fla-org/flash-linear-attention/issues/1228)
(*"`chunk_fwd_kernel_o` is non-deterministic at `num_warps=8` on **GB300 (sm_103)**"*, merged
2026-09-09) drops the 8-warp config in `fla/ops/common/chunk_o.py`:

```python
CHUNK_FWD_O_BLACKWELL_DROPPED_CONFIGS = [] if IS_NVIDIA_BLACKWELL else [
    triton.Config({'BK': 128, 'BV': 128}, num_warps=8, num_stages=3),
]
```

**KDA does not use that file.** It uses `fla/ops/gla/chunk.py::chunk_gla_fwd_kernel_o`, which
is the structural twin and is **not** gated. And #1228's own closing ask, verbatim:

> *"The backward kernels in `chunk_o.py` (`chunk_bwd_kernel_dqkwg`, `chunk_bwd_kernel_dv`) and
> `chunk_bwd_dqkwg`'s siblings share the dot-in-loop shape and, on non-Hopper, an 8-warp entry
> in `NUM_WARPS`. **We did not test them in production. They may deserve the same audit #945
> asked for.**"*

**That audit has not been done, and the KDA siblings are exactly what it would have covered.**

### 3.5 Why this went unnoticed: there is no Blackwell in FLA's CI

*(measured)* `fla-org/flash-linear-attention` `.github/workflows/` contains runners
`nvidia-h100-1`, `intel-b580`, `amd-mi300` and `linux-aarch64-a2-1` (Ascend). **There is no
B200, B300, GB200 or GB300 runner anywhere.** Every Blackwell bug in this class was found by a
user in production, months apart:

| issue | date | hardware | kernel | symptom |
|---|---|---|---|---|
| #607 / #612 / #618 / #638 / #639 | 2025-10 → 2025-11 | B200 | GDN, **KDA** | backward errors, hangs, IMA |
| [#727](https://github.com/fla-org/flash-linear-attention/issues/727) | 2026-01-23 | B200 | **`chunk_kda_bwd`** | illegal memory access during autotune |
| [#790](https://github.com/fla-org/flash-linear-attention/issues/790) | 2026-03-23 | Blackwell | `chunk_..._fwd_kernel_h_blockdim64` | incorrect outputs for certain autotune configs |
| [#913](https://github.com/fla-org/flash-linear-attention/issues/913) | 2026-05-21 → stale-closed 2026-08-16 | **B200/B300**, GB10, RTX PRO 6000 | GDN fwd/bwd | **silent NaN gradients**, IMA, misaligned address |
| [#945](https://github.com/fla-org/flash-linear-attention/issues/945) | 2026-06-12 | **B300 (sm_103)** | `chunk_..._fwd_kernel_h_blockdim64` | bitwise non-determinism, 30/30 distinct outputs |
| [#999](https://github.com/fla-org/flash-linear-attention/issues/999) | 2026-07-01 | B200 | `prepare_wy_repr_bwd` | hang during autotune → NCCL watchdog timeout |
| [#1228](https://github.com/fla-org/flash-linear-attention/issues/1228) | **2026-09-07** | **GB300 (sm_103)** | `chunk_fwd_kernel_o` | 13–23% of tokens change MoE route between two identical forwards |

**Three weeks ago.** This is not a historical problem.

#945's description of the downstream damage is the reason this section leads the document —
verbatim:

> *"Causal chain: this kernel's non-deterministic `h`/`v_new` → GatedDeltaNet output differs
> between checkpoint forward and recompute → MoE router logits shift by a few ULPs across the
> sequence → ~0.4% of top-k expert assignments flip […] **Without checkpointing (or with eager
> MoE), training on B300 appears to "work" while every linear-attention forward is silently
> non-reproducible.**"*

GLM-5.3-Flash has **activation checkpointing on** (recipe L86 `activation_checkpointing: true`)
and **288 routed experts with top-8 routing**. That is the exact configuration #945 and #1228
describe.

### 3.6 Does it hit KDA specifically? **Unknown, and that is the answer.**

Being scrupulous about what is and is not established:

- The #913 forward corruption was bisected to
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel` in `fla/ops/gated_delta_rule/chunk_fwd.py`.
  **KDA does not use that kernel** — it has its own analogue,
  `chunk_kda_fwd_kernel_inter_solve_fused` in `fla/ops/kda/chunk_intra.py`, which fuses the
  same kkt + `solve_tril` operation in a different file.
- #945, #1228, #10590 are all GDN-path or common-path kernels. **No public report exists of
  `fla.ops.kda.chunk_kda` producing wrong answers on sm_103.**
- But #727 and #639 *are* KDA-on-Blackwell crashes (B200), and the structural precondition —
  `tl.dot` into an fp32 accumulator inside a sequential loop, at 4 or 8 warps, above one wave
  of CTAs — is present throughout the KDA kernels listed in §3.4.

**So: the bug class is live in the KDA path by construction; whether it fires at GLM-5.3-Flash's
particular shapes on this particular cluster is unknown until measured.** That is not a reason
to relax. #945 and #1228 both describe kernels that looked fine in isolation and raced only
under production load, and #1228 could not reproduce its own production failure standalone:

> *"So the 128x128-tile 8-warp race needs full training load. We can show it only with the
> production measurement below. #945 had the same experience in reverse."*

**The gate, and it is mandatory.** Before trusting a single loss value:

```python
# Run on ONE B300, at the real KDA head config, BEFORE any training.
# GLM-5.3-Flash: linear_attn_config.num_heads 64, head_dim 128 (config.json, measured).
import torch, torch.nn.functional as F
from fla.ops.kda import chunk_kda

def bithash(t):
    v = t.detach().contiguous().view(torch.int16 if t.dtype==torch.bfloat16 else torch.int32).flatten().to(torch.int64)
    w = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 8191 + 1
    return int((v * w).sum().item())

B, T, H, K, V = 1, 8192, 64, 128, 128      # scale T until NT*B*H >= 1024 launched blocks
torch.manual_seed(0)
q = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
k = torch.randn(B, T, H, K, device="cuda", dtype=torch.bfloat16, requires_grad=True)
v = torch.randn(B, T, H, V, device="cuda", dtype=torch.bfloat16, requires_grad=True)
g = -F.softplus(torch.randn(B, T, H, V, device="cuda", dtype=torch.float32)).requires_grad_()
beta = torch.rand(B, T, H, device="cuda", dtype=torch.bfloat16).sigmoid().requires_grad_()

fwd, bwd = set(), set()
for _ in range(25):
    for t in (q, k, v, g, beta):
        t.grad = None
    o, _ = chunk_kda(q, k, v, g, beta, use_qk_l2norm_in_kernel=True, state_v_first=True)
    fwd.add(bithash(o))
    o.sum().backward()
    assert torch.isfinite(q.grad).all(), "NaN/Inf in dq -- FLA KDA backward is corrupt on this stack"
    bwd.add(bithash(q.grad))
print("distinct fwd hashes:", len(fwd), " distinct bwd hashes:", len(bwd))   # BOTH MUST BE 1
print("selected configs:")   # re-run with TRITON_PRINT_AUTOTUNING=1 and record them
```

Run it at `T ∈ {1024, 2048, 4096, 8192, 16384}` and with `TRITON_PRINT_AUTOTUNING=1`. Record
which `num_warps` the autotuner picks for each kernel — if it picks 4 or 8 anywhere, you are
inside the #10590 envelope. Repeat on Triton 3.7.1 **and** 3.8.0 and keep the comparison.

If it fails, the ordered mitigations are:

1. **Triton 3.8.0** — the only actual fix, verified on a B300 (§3.3).
2. **FLA from `main` (0.6.0)**, not PyPI 0.5.2 — picks up #1109 (KDA SM100 backward) and #1228
   (`chunk_fwd_o`). NeMo pins only `flash-linear-attention>=0.4.2` (`pyproject.toml` L147,
   **measured**), so a plain resolve lands on 0.5.2 and gets neither. `chunk_kda` on `main`
   still accepts NeMo's deprecated `transpose_state_layout` kwarg (`kda/chunk.py` L403-411),
   so `main` is drop-in for NeMo.
3. **Cap the autotune space by hand** — monkeypatch `num_warps ≤ 2` on the §3.4 kernels, the
   same move #945/#1228 made. #1228 measured **zero step-time cost** for its gate (*"Step time
   is the same with and without the gate in our runs"*), but that was one kernel on one shape.
4. **Torch reference** — correct, converges (#913: *"converges through 12K+ steps"*), and
   **3–5× slower per step** across 34 of 45 layers. This is the option the "3-5x" quote is
   actually about, and it is the floor, not the expectation.

**If the penalty in (4) had to be paid across the whole KDA path, it would dominate every
wall-clock estimate on this map.** It is not currently established that it must be — but it is
also not established that it need not be, and no amount of reading settles it.

### 3.7 Two smaller Blackwell facts in FLA, for completeness

*(measured, `fla/ops/utils/op.py` L40-58)* On Blackwell, FLA wraps every `safe_dot` in inline
asm to defeat a Triton pass:

> *"On SM100 datacenter and SM120 consumer Blackwell GPUs, wraps the result in inline assembly
> to prevent the TritonGPUHoistTMEMAlloc pass from incorrectly fusing add and dot operations.
> See: fla-org/flash-linear-attention#638 … TODO: Remove this workaround once the Triton
> compiler bug is fixed. Track upstream issue at: triton-lang/triton#8695"*

**KDA does not use `safe_dot`** (grep: only `delta_rule`, `precond_gated_delta_rule`,
`mesa_net` do) — which is one more way KDA is outside the blast radius of the fixes that have
been made.

*(measured, `fla/ops/kda/backends/flash_kda.py`)* Moonshot's fused CUTLASS `FlashKDA` backend
exists but is **inference-only**: `if torch.is_grad_enabled(): return False, "FlashKDA only
supports inference mode"`. Irrelevant to training. A TileLang backend exists for one KDA
backward kernel (`fla/ops/kda/backends/tilelang/`) but requires `tilelang` + a usable `nvcc`;
NeMo does not install it.

---

## §4 — Sequence packing and KDA boundaries: packing **is** available and **is** correct

The operator's flag — that Qwen3.8 could not use packing because a Gated DeltaNet recurrent
state cannot be reset mid-sequence — **does not transfer to NeMo's GLM-5.3-Flash path.**

**FLA's chunked kernels are varlen-native.** `fla/ops/common/chunk_delta_h.py` **L88-96**,
the kernel KDA's forward and backward both call, verbatim:

```python
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        ...
    else:
        bos, eos = i_n * T, i_n * T + T
```

with `'IS_VARLEN': lambda args: args['cu_seqlens'] is not None` (L46). **One program per
document, state initialised at that document's `bos`.** The recurrent state does not carry
across a document boundary because each document is a separate program instance with its own
accumulator. That is a structurally different mechanism from a dense `[B, T, …]` loop, and it
is why packing works here and did not for Qwen3.8's `transformers` GDN path (whose torch
reference `torch_chunk_gated_delta_rule` has no `cu_seqlens` argument at all).

**NeMo passes `cu_seqlens` through.**
`nemo_automodel/components/models/glm5_next/layers.py` **L378-397**, verbatim:

```python
        if _CHUNK_KDA_OK and hidden_states.is_cuda:
            kernel = _chunk_kda if cp_context is not None or hidden_states.shape[1] > 64 else _recurrent_kda
            ...
            output, _ = kernel(
                q=q, k=k, v=v, g=gate, beta=beta,
                initial_state=None,
                output_final_state=cp_context is None,
                cu_seqlens=cu_seqlens,
                **kernel_options, **kernel_kwargs,
            )
```

**NVIDIA states it as a property of the path**, coverage doc **L59-61**, verbatim:

> *"KDA layers use Flash Linear Attention (FLA) kernels. For CP, FLA carries the recurrent
> state across contiguous sequence shards **while preserving packed document boundaries**."*

**And NeMo tests it** *(measured)*:
`tests/unit_tests/models/glm5_next/test_model.py::test_packed_documents_are_attention_isolated`
(perturb one document, assert the other's logits are unchanged) and
`::test_tiny_hybrid_packed_forward_backward_is_finite`; plus
`tests/unit_tests/models/glm5_next/test_cp.py` covering `doc_ids_from_cu_seqlens` /
`segment_cu_seqlens` round-trips.

**The shipped recipe packs**, `examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml`:

```yaml
packed_sequence:
  pretokenize: true
  max_length: 2048
  pack_size: 2048
  collate_max_length: 2048
  packing_ratio: 0.9
  drop_long_samples: true
  balance_media_tokens: true
  packing_format: neat
```

and the validated configuration is *"packed THD sequences of 2,048 tokens"* (coverage doc L73),
with CP1/CP8 parity measured over 100 matched steps on the packed workload (L136-147).

**Resolving the #67 tension:** #67's *"fundamentally incompatible with indexer"* was about
`transformers`' own test suite skipping packing for the **DSA indexer** — the 11 sparse-MLA
layers — on the `transformers` tree. NeMo does not use `transformers`' implementation; it owns
its own (`glm5_next/layers.py`, `_forward_document`), and it packs. **The two statements are
about two different codebases and do not conflict.**

**One residual caveat I will not paper over:** all of NVIDIA's packing validation is at
**2,048-token packs** (and the CP-parity runs at the same). Longer packs are untested by the
vendor. And `balance_media_tokens: true` plus `drop_long_samples: true` are dataset-shaping
knobs that will interact with whatever chat3d's corpus looks like.

**Throughput consequence: packing is ON, so the pessimistic "no packing" throughput scenario
does not apply.** Good news, and it is one fewer unknown.

---

## §5 — Memory at EP16 on 16 B300

Per the coordinator, memory is no longer the question. This section is short, and it exists
only to close it out.

### 5.1 The parameter budget *(carried from #71; expert term is exact, dense term ±1%)*

| Component | value |
|---|---:|
| Routed-expert params loaded (42 sparse layers × 288 × 25,165,824) | 304,405,807,104 *(derived, exact)* |
| Everything else, MTP dropped (`num_nextn_predict_layers: 0`) | ≈ 9.45 B *(derived, bounded 9.42–9.52)* |
| **Model NeMo loads** | **≈ 313.85 B = 584.6 GiB BF16** *(derived)* |
| — of which expert weights | 567.0 GiB |
| — of which dense (FSDP2-sharded) | 17.6 GiB |

The base **cannot stay FP8** *(carried, #71 §2.3, verified in source)* —
`dequantize_base_checkpoint` is a loading flag; `state_dict_adapter.py` `_dequantize` converts
to `self.dtype` = bfloat16, and the coverage doc says the same from the other side (L23):
*"Training Precision: BF16 after FP8 checkpoint dequantization"*.

### 5.2 Residency, against the **measured** 268.6 GiB/GPU

EP shards the experts; FSDP2 shards the dense remainder over `dp_shard × cp`.

| GPUs | `ep_size` | experts/GPU | expert GiB | dense GiB | **weights GiB/GPU** | fill of 268.6 | free |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 (1 node) | 8 | 36 | 70.9 | 2.20 | **73.1** | 27% | 195.5 GiB |
| 16 (2 nodes) | 8 (`ep_shard 2`) | 36→18 | 35.4 | 1.10 | **36.5** | 14% | 232.1 GiB |
| **16 (2 nodes)** | **16** | **18** | **35.4** | **1.10** | **36.5** | **14%** | **232.1 GiB** |

All **derived**. For reference, the same arithmetic gave 73.1 GiB against an H200's 140.4 GiB
in #71 — a 52% fill that was the binding constraint. On B300 it is a rounding error.

### 5.3 The load-time dequantize peak — no longer the top risk

#71 identified the FP8→BF16 dequantize transient as *"the single most likely place a one-node
run fails"*. The mechanism is unchanged — `nemo_automodel/components/moe/state_dict_mixin.py`
**L1058-1060**, verbatim:

> *"Quantization casts each split with `value.to(float8_e4m3fn)`, creating storage separate
> from the model's grouped weights. DCP must not treat that cast as model weight memory …
> Quantized loads therefore rebuild the grouped expert tensor after the read."*

It scales linearly with the EP shard, because the FP8 destination is half the size of the BF16
weights it sits beside:

| | BF16 model | + FP8 destinations | **load peak/GPU** | of 268.6 GiB |
|---|---:|---:|---:|---:|
| 8 GPUs, EP8 | 73.1 | 36.5 | **109.6** | **41%** |
| **16 GPUs, EP16** | 36.5 | 18.3 | **54.8** | **20%** |
| 72 GPUs, EP72 (NVIDIA's) | 8.1 | 4.1 | 12.2 | — |

All **derived**. Against H200's 140.4 GiB the EP8 figure was a 78% fill and genuinely
frightening; against 268.6 it is comfortable at either topology. **This risk is retired.**

### 5.4 Sequence length

CP is validated for this model — the coverage doc's CP1/CP8 parity table (**measured**, 18
nodes / 144 H100, EP144, peaks 38.89 vs 41.04 GiB, final losses 1.2344 vs 1.2328, mean |Δ|
0.001879 over 100 matched steps). Legal meshes at 16 GPUs, `tp1 pp1`, by the constraint in
§6.1 (`non_pp_size % ep_size == 0`, `288 % ep_size == 0`):

| target seq | `cp_size` | `dp_size` | `ep_size` | `non_pp` | tokens/rank | weights/GPU |
|---:|---:|---:|---:|---:|---:|---:|
| ≤ 8k | 1 | 16 | 16 | 16 ✓ | 8,192 | 36.5 GiB |
| 16k | 2 | 8 | 16 | 16 ✓ | 8,192 | 36.5 GiB |
| 32k | 4 | 4 | 16 | 16 ✓ | 8,192 | 36.5 GiB |
| 32k | 8 | 2 | 16 | 16 ✓ | 4,096 | 36.5 GiB |
| 64k | 16 | 1 | 16 | 16 ✓ | 4,096 | 36.5 GiB |

The non-weight term remains **unknown**; the only anchor is ≈10.6 GiB/GPU at 4,096 tokens
*(derived from NVIDIA's measured 38.8 GiB peak for Nemotron 3.5 Super VL, a different
architecture — carried from #71, still ±2×)*. GLM carries `hc_mult: 4` residual streams and an
`index_topk: 2048` DSA indexer that inflate it by unknown factors. **But with 232 GiB free per
GPU, even a 20× error in that extrapolation still fits.** Provision on the weight column and
let the probe find the ceiling.

**An independent sanity check from the cluster itself** *(measured, supplied by the
coordinator)*: an 8-GPU multimodal LoRA ran there at **peak 55 GiB/GPU allocated**, 75
optimizer steps in 14.1 min. Different model, but it confirms the node and the allocator work.

---

## §6 — Two nodes: what it actually needs

### 6.1 The EP constraints are still exactly two, and neither bites

`nemo_automodel/components/distributed/mesh_utils.py` **L316-319**, verbatim *(measured,
line numbers current on `1358302c`)*:

```python
    non_pp_size = dp_size * cp_size * tp_size
    if non_pp_size % ep_size != 0:
        raise ValueError(f"{non_pp_size=} must be a multiple of {ep_size=}")
    ep_shard_size = non_pp_size // ep_size if ep_size < non_pp_size else 1
```

and `nemo_automodel/components/moe/parallelizer.py` **L1216-1219** / `moe/experts.py`
**L491-493**:

```python
assert self.n_routed_experts % ep_size == 0, (
    f"Number of experts must be divisible by ep_size (ep_size={ep_size})")
```

**16 % 16 == 0 ✓ and 288 / 16 = 18 ✓.** There is no third constraint — no `ep_size` upper
bound, no minimum node count, no EP-size table. `supports_ep: True` with no cap
(`glm5_next/model.py` L230), and `grep -n "ep_size\|ep_mesh" nemo_automodel/components/models/glm5_next/`
returns zero hits: EP for this model is entirely generic.

**Nothing in the mesh assumes a single node.** `mesh_utils.py` builds everything from the
*global* `world_size` and global rank indices; it contains no `LOCAL_WORLD_SIZE`, no
`nproc_per_node`, no `LOCAL_RANK` (verified by grep over the whole file). EP is deliberately
carved out of the flattened `DP_REPLICATE × DP_SHARD × CP × TP` block, i.e. it is designed to
span nodes. The only `local_rank` arithmetic in the MoE tree lives in
`nemo_automodel/components/moe/uccl_ep/` — a *different* dispatcher, irrelevant unless selected.

### 6.2 The one real 2-node wall, and the condition that trips it

`nemo_automodel/components/moe/megatron/fused_a2a.py` **L247-253**, verbatim:

```python
    if not nvshmem and group.size() > 8:
        raise RuntimeError(
            f"DeepEP was compiled without NVSHMEM support (SM90 features disabled), "
            f"but expert parallelism group size {group.size()} > 8 requires internode "
            f"RDMA communication. Recompile DeepEP with NVSHMEM or reduce ep_size to "
            f"fit within a single node (max 8 GPUs)."
        )
```

**`ep_size: 16` across 2 nodes fires this unless DeepEP was built with NVSHMEM.** NeMo's own
image does build it that way (`docker/Dockerfile` L156 `pip install nvidia-nvshmem-cu13==3.6.5`),
so with the shipped container it is inert. **If you build your own environment or
`pip install deep_ep` without `NVSHMEM_DIR`, this is where 2 nodes stops.**

DeepEP's own `setup.py` at NeMo's pin `10d4dd73`, L220, verbatim:

> *"Warning: `NVSHMEM_DIR` is not specified, and the NVSHMEM module is not installed. All
> internode and low-latency features are disabled"*

### 6.3 What `dispatcher: hybridep` is, and its cost at 2 nodes

`nemo_automodel/components/moe/megatron/token_dispatcher.py` **L364-367**, verbatim:

> *"A manager class to handle fused all-to-all communication processes for MoE models using
> HybridEP backend. See https://github.com/deepseek-ai/DeepEP/tree/hybrid-ep for more details."*

HybridEP is a **branch of DeepEP**, not a separate project; DeepEP's own
`docs/README_Hybrid-EP.md` describes it as *"an optimized implementation developed by NVIDIA
that uses TMA instructions and warp-level pipeline parallelism to minimize SM usage while
maximizing network bandwidth. It supports both intra-node (NVLink) and inter-node (RDMA)
communication."* The NVLink-domain split is explicit, via
`NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` (documented in NeMo's `ling_1t_sft.yaml` L16 for
EP64; **not set** by the GLM-5.3-Flash EP72 recipe, and its threshold for EP16 is undocumented).

Build requirements *(measured, `docker/Dockerfile` L134-165)*: `rdma-core` + `libibverbs-dev`,
`nvidia-nvshmem-cu13==3.6.5`, `HYBRID_EP_MULTINODE=1`, `RDMA_CORE_HOME`,
`TORCH_CUDA_ARCH_LIST="9.0 10.0 12.0"`. **Add `10.3` to that list.**

**Two operational traps at 2 nodes:**

1. **Cold JIT on a new arch.** HybridEP JIT-compiles per
   `f"deep_ep-{version}_cuda-{torch.version.cuda}_sm{major}{minor}"`
   (`fused_a2a.py` L569). On `sm103` that fingerprint has never been seen, so the first step
   pays a full compile. NeMo's own DeepSeek-V4.1 coverage doc warns this *"can stretch the
   first step past the NCCL timeout"*. **Point the cache at `/mnt/data`, not `/tmp`** (which
   the coordinator confirmed is node-local), and raise `dist_env.timeout_minutes`.
2. **NCCL PXN + the container network plugin.** NeMo documents a real multi-node InfiniBand
   failure (`docs/model-coverage/llm/deepseek-ai/deepseek-v41-flash.mdx` L125-133, NeMo issue
   #3959): *"With the container's default NCCL network plugin and PXN both active, the Engram
   row-owner all-to-all received byte-shifted token IDs from the first node's ranks […] with
   NCCL 2.30.5 and 2.30.7 alike."* Mitigation is to disable PXN or bypass the plugin. Not this
   model, but the same fabric and the same plugin.
   Also, `fused_a2a.py` L256, verbatim: `# NOTES: the adaptive routing configuration of the network **must be off**`.

**The zero-build escape hatch is `dispatcher: torch`** — `moe/layers.py` L811-813 falls through
to `GroupedExperts` using DTensor all-gather/reduce-scatter, i.e. plain NCCL. Slower, no
DeepEP, no NVSHMEM, no RDMA build. **That is what the first probe should use.**

### 6.4 Launching it

`docs/launcher/slurm.mdx` *(measured)*. Interactive single node: `automodel --nproc-per-node 8 config.yaml`.
Multi-node is `sbatch` over a copied `slurm.sub`, which *"Runs `torchrun -m nemo_automodel.cli.app $CONFIG`
on each node via `srun`"*:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=13742
srun ... torchrun \
    --nproc-per-node=${SLURM_GPUS_PER_NODE:-8} \
    --nnodes=${SLURM_NNODES:-1} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    -m nemo_automodel.cli.app ${CONFIG}
```

The cluster is Slurm 25.11.3 on Nebius soperator *(measured)*, so this is the native path.
2-node recipes are common in NeMo's tree (`qwen3_moe_30b_*_gb200.yaml` at `num_nodes: 2`,
`custom_qwen2_5_32b_peft_benchmark_2nodes.yaml`), so **2 nodes is a well-trodden shape** — just
never at EP16 on this model, and never on sm_103.

---

## §7 — The runnable configuration

### 7.1 Escalation ladder — do these in order, not all at once

| Step | Shape | What it adds to the failure surface | Gate to pass |
|---|---|---|---|
| **0** | one GPU, no model | nothing | **§3.6 KDA determinism check**, on Triton 3.7.1 and 3.8.0 |
| **1** | **1 node, EP8, `dispatcher: torch`, `attn: sdpa`, CP1, packed 2k** | FSDP2 + EP + the FP8→BF16 load | model loads; `torch.cuda.max_memory_allocated()` after `load_base_model`; one optimizer step; `assert` the freeze (§7.3) |
| **2** | 1 node, EP8, `dispatcher: hybridep`, `attn: cudnn` | DeepEP NVLink leg + FlashMLA + cuDNN | step time vs step 1; cuDNN-vs-SDPA delta on *this* hardware |
| **3** | **2 nodes, EP16**, `dispatcher: hybridep` | NVSHMEM + RDMA + inter-node JIT | loss curve matches step 2 bitwise-ish; no NCCL timeout on step 1 |
| **4** | 2 nodes, EP16, CP2 @ 16k | CP state-carry across shards | peak memory; loss sanity |

**Step 0 is not optional and it is cheap.** Everything after it is worthless if it fails.

### 7.2 The recipe

Compose onto `examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml`.
Only the deltas are shown; everything else stays as shipped.

```yaml
# ---- STEP 1: one node, EP8, maximum-safety backends ----
# (STEP 3 deltas are in comments)

distributed:
  strategy: fsdp2
  tp_size: 1          # "TP and PP are not supported for this model" (coverage doc L168)
  pp_size: 1
  ep_size: 8          # STEP 3: 16   (288/16 = 18 experts/GPU; 16 % 16 == 0)
  cp_size: 1          # STEP 4: 2 at 16k, 4 or 8 at 32k
  sequence_parallel: false
  activation_checkpointing: true
  defer_fsdp_grad_sync: false
  moe:
    reshard_after_forward: false
    wrap_outer_model: true
    ignore_router_for_ac: true

model:
  _target_: nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained
  pretrained_model_name_or_path: zai-org/GLM-5.3-Flash   # the FP8 repo, 328.3 GB -- NOT the BF16 sibling
  torch_dtype: bfloat16
  attn_implementation: sdpa
  use_liger_kernel: false
  text_config:
    output_hidden_states: true
    num_nextn_predict_layers: 0        # drops the MTP layer; 313.85 B loaded
  backend:
    _target_: nemo_automodel.components.models.common.BackendConfig
    attn: sdpa          # STEP 2: cudnn  (needs FlashMLA @ b7643bd + nvidia-cudnn-frontend[cutedsl])
    linear: torch
    rms_norm: torch_fp32
    experts: torch_mm
    dispatcher: torch   # STEP 2/3: hybridep  (needs DeepEP built WITH NVSHMEM for ep_size>8)
    rope_fusion: false
    gate_precision: float32
    fake_balanced_gate: false
    enable_hf_state_dict_adapter: true
    enable_fsdp_optimizations: true

checkpoint:
  enabled: false
  model_save_format: safetensors
  save_consolidated: false
  dequantize_base_checkpoint: true    # loading flag only -- the base is BF16 in HBM

dist_env:
  backend: nccl
  timeout_minutes: 180      # raised from 120: first step pays a cold sm103 JIT

step_scheduler:
  global_batch_size: 8      # 1 node: local_batch 1 x dp 8 x 1 microstep
  local_batch_size: 1
  max_steps: 20             # a probe, not a run

# ---- targets: every glob anchored on the language-model prefix ----
# Verified in #71 by executing NeMo's own ModuleMatcher against the real module paths:
# zero matches under model.visual.*, full coverage of all 45 LM layers.
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
    # --- KDA linear attention: 34 of 45 layers. glm_5.2_lora.yaml reaches NONE of these. ---
    - "*.language_model.layers.*.self_attn.q_proj"
    - "*.language_model.layers.*.self_attn.k_proj"
    - "*.language_model.layers.*.self_attn.v_proj"
    # optional, cheap, KDA-only gating/decay projections:
    # - "*.language_model.layers.*.self_attn.f_a_proj"
    # - "*.language_model.layers.*.self_attn.f_b_proj"
    # - "*.language_model.layers.*.self_attn.g_a_proj"
    # - "*.language_model.layers.*.self_attn.g_b_proj"
    # - "*.language_model.layers.*.self_attn.b_proj"

    # --- KPool-DSA / sparse MLA: the other 11 layers ---
    - "*.language_model.layers.*.self_attn.q_a_proj"
    - "*.language_model.layers.*.self_attn.q_b_proj"
    - "*.language_model.layers.*.self_attn.kv_a_proj_with_mqa"
    - "*.language_model.layers.*.self_attn.kv_b_proj"

    # --- output projection: present on BOTH attention kinds, all 45 layers ---
    - "*.language_model.layers.*.self_attn.o_proj"

    # --- dense MLP, layers 0-2 (first_k_dense_replace: 3) ---
    - "*.language_model.layers.*.mlp.gate_proj"
    - "*.language_model.layers.*.mlp.up_proj"
    - "*.language_model.layers.*.mlp.down_proj"

    # --- shared expert, the 42 sparse layers ---
    - "*.language_model.layers.*.mlp.shared_experts.gate_proj"
    - "*.language_model.layers.*.mlp.shared_experts.up_proj"
    - "*.language_model.layers.*.mlp.shared_experts.down_proj"

    # --- 288 routed experts: OFF for the first probe.
    #     5.17 GiB/GPU at dim 32, and GroupedExpertsDeepEPLoRA has never run on a VLM tree.
    # - "*.language_model.layers.*.mlp.experts"

  dim: 32
  alpha: 64
  dropout: 0.0
  use_memory_efficient_lora: true
  use_triton: true
  # moe_rank_scaling: true   # only meaningful once *.mlp.experts is enabled

freeze_config:
  freeze_vision_tower: true   # belt: freezes the whole model.visual.* subtree by path substring
  freeze_audio_tower: true
  freeze_language_model: false
  # do NOT add freeze_embeddings -- FreezeConfig has no such field and parse_freeze_config
  # raises on unknown keys. It was removed in a9bc22f9 and a unit test asserts its absence.
```

**Why the target globs are what they are** *(carried from #71, which executed NeMo's matcher
against the real module paths)*: `glm_5.2_lora.yaml`'s `*.mlp.gate_proj` **does** match
`model.visual.blocks.7.mlp.gate_proj`, and its MLA-only attention globs reach **zero** of the
34 KDA layers' q/k/v. `target_modules` and `exclude_modules` are **mutually exclusive**
(`module_matcher.py` L110-113 raises), so the fix must be anchored globs, not an exclude list.
`freeze_vision_tower` is the belt: it freezes `model.visual.*` by path substring *after*
adapter injection, so the tower cannot be trained even by a sloppy glob — but a sloppy glob
still allocates and serialises dead adapters.

At `dim: 32`, attention + shared experts + dense MLP is **≈112 M trainable params, 0.036% of
the model, ~0.03 GiB/GPU** *(derived)* — the same order as the 177 M / 0.15% NVIDIA shipped
for Nemotron 3.5 Super VL.

### 7.3 Environment

```bash
# --- shared scratch: /tmp is node-local on this cluster ---
export TRITON_CACHE_DIR=/mnt/data/cache/triton
export HYBRID_EP_CACHE_DIR=/mnt/data/cache/hybridep     # cold sm103 fingerprint on first step
export HF_HOME=/mnt/data/hf
export TRITON_PRINT_AUTOTUNING=1                        # record which num_warps get picked

# --- the two mitigations that matter, in priority order ---
pip install --no-deps --force-reinstall 'triton==3.8.0'            # the actual fix for sm_103
pip install --no-deps 'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention@main'
#   PyPI 0.5.2 (2026-07-27) predates BOTH the KDA SM100 backward fix (#1109, 2026-08-09)
#   and the chunk_fwd_o Blackwell gate (#1228, 2026-09-09). NeMo pins only >=0.4.2.

# --- build DeepEP/HybridEP for sm_103, with NVSHMEM (required for ep_size > 8) ---
export TORCH_CUDA_ARCH_LIST="10.0 10.3"    # NeMo's Dockerfile uses "9.0 10.0 12.0" -- no 10.3
export HYBRID_EP_MULTINODE=1
export RDMA_CORE_HOME=/opt/rdma-core/build
export NVSHMEM_DIR=...                     # or: pip install nvidia-nvshmem-cu13==3.6.5

# --- FlashMLA, for STEP 2 only. nvcc must be >= 12.9 (cluster has 12.9.86) ---
git clone --recursive https://github.com/deepseek-ai/FlashMLA.git /tmp/FlashMLA
git -C /tmp/FlashMLA checkout b7643bd54521f563b839b98289b5cd048c062ba2
git -C /tmp/FlashMLA submodule update --init --recursive
pip install --no-build-isolation /tmp/FlashMLA        # emits sm_100f -> runs on sm_103 (CPG Table 28)

# --- multi-node NCCL, per NeMo's own #3959 warning ---
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=5
# adaptive routing on the fabric must be OFF (fused_a2a.py L256)
# if the first all-to-all returns garbage: disable PXN or bypass the container NCCL plugin
```

### 7.4 Assertions the probe must make

```python
# 1. The freeze actually held. Run AFTER apply_model_infrastructure returns --
#    the trainability policy is re-resolved three times; only the last reflects post-shard reality.
from collections import defaultdict
buckets = defaultdict(lambda: [0, 0])
for name, p in model.named_parameters(remove_duplicate=False):
    key = ("visual" if ".visual." in name or name.startswith("model.visual")
           else "language_model" if "language_model" in name else "other")
    buckets[key][1] += p.numel()
    if p.requires_grad:
        buckets[key][0] += p.numel()
assert buckets["visual"][0] == 0, "vision tower is trainable -- the freeze did not hold"
assert buckets["other"][0] == 0, "lm_head/embeddings trainable -- target globs are too broad"

# 2. The load peak, before step 1.
print("load peak GiB/GPU:", torch.cuda.max_memory_allocated() / 2**30)   # expect ~110 @ EP8, ~55 @ EP16

# 3. Gradients are finite, EVERY step. This is the sm_103 tripwire -- see §3.
for n, p in model.named_parameters():
    if p.requires_grad and p.grad is not None:
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {n} -- suspect FLA/Triton on sm_103"

# 4. Determinism under activation checkpointing (#945's exact failure mode).
#    Run the SAME batch twice with the same weights; compare logits bitwise.
#    Any difference means the KDA forward is racing and MoE routing will flip.
```

---

## §8 — What I could NOT determine without running it

Ordered by how much each one moves the verdict.

1. **Whether FLA's KDA kernels are correct on sm_103.** No public report exists either way
   (§3.6). The bug class is present by construction — same kernel shape, same ungated 4/8-warp
   config space, and B300's failure envelope is *wider* than the B200 envelope the existing
   gates were calibrated on. **Measure: the §3.6 determinism script, on Triton 3.7.1 and
   3.8.0, at T ∈ {1k…16k}, with `TRITON_PRINT_AUTOTUNING=1`.** Nothing else on this list
   matters if this fails.

2. **Whether Triton 3.8.0 works with torch 2.13.0 and FLA 0.6.0.** torch pins `triton==3.7.1`
   exactly; 3.8.0 exists on PyPI. Overriding is untested by PyTorch. **Measure: install and
   run the §3.6 script plus one training step.**

3. **s/step, tokens/s, and MFU — still completely unknown, as they have been since #69.**
   NeMo's published TPS (9,489 cuDNN / 8,378 SDPA), training-loop time (33:18 / 37:15) and
   batch size (144 × 2048-token packs on 72 H100) are **not mutually consistent under any
   reading**, and the doc never defines whether TPS is global or per-GPU: 144 × 2048 / 19.98 s
   implies ~14,800 tok/s globally, not 9,489. **#69's 20% MoE MFU has no support anywhere in
   NeMo's results**, and NeMo has no B300 entry in `_DEVICE_FLOPS` so its own MFU reporting
   will print `inf`. **I refuse to compute a wall-clock.** #72 is what settles it. **Measure:
   s/step with the optimizer step included, at a real sequence length, on this cluster.**

4. **Whether the FP8→BF16 load actually peaks where the arithmetic says.** ~110 GiB/GPU at
   EP8, ~55 at EP16 (**derived** from the source's own comment). Whether all destinations are
   live simultaneously depends on DCP's planner, which cannot be settled by reading. It no
   longer threatens the run on 268.6 GiB cards, but the number is worth capturing.

5. **Whether `hybridep` performs well at EP16 across 2 nodes.** NVIDIA validated EP72/9 nodes
   and EP144/18 nodes. **EP16 on 2 nodes has never been run for this model by anyone.**
   `NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN` is documented as needed at EP64; its threshold
   for EP16 is undocumented. **Measure: step time at EP16/2-node vs EP8/1-node.**

6. **Whether `backend.attn: cudnn` is worth it on Blackwell.** NVIDIA's +13.26% is an H100
   number at EP72/CP2 with 2,048-token packs. It has never been measured on sm_103, and
   FlashMLA's sm_100f cubin runs at the *family-common* feature subset there rather than the
   full sm_103a set. **Measure: A/B at step 2 of the ladder.**

7. **Whether a CUDA 13 toolkit is available on these nodes.** System nvcc is 12.9.86
   *(measured)*, which clears FlashMLA's floor but is below CUTLASS-3's stated B300 minimum of
   13.0, and Nebius's only B300 boot image is `ubuntu24.04-cuda13.0`. **Measure: `ls /usr/local/cuda*`,
   `nvcc --version`, and `python -c "import torch; print(torch.cuda.get_arch_list())"` — the
   last of those also settles §2.7 definitively rather than by build-recipe inference.**

8. **Whether `GroupedExpertsDeepEPLoRA` binds on the VLM tree.** Every link is verified by
   reading *(carried, #71 §4.5)*, but `"*.mlp.experts"` appears in exactly one recipe repo-wide
   and that one is an LLM. Optional for the first probe — leave expert LoRA off.

9. **Non-weight memory at 16k and 32k.** Extrapolated ±2× from one datapoint on a different
   architecture. **Now decision-irrelevant** — 232 GiB of headroom absorbs any plausible error
   — but worth capturing as a three-point curve at (8k, CP1), (16k, CP2), (32k, CP8).

10. **The Nebius SKU/region of this specific cluster.** The coordinator could not determine it
    on the machine and instructed me not to guess. §1.3's `gpu-b300-sxm` entry is offered as
    *consistent with* the measurements, not as an identification. One open question if it does
    matter later: `nebius.com/prices` publishes a rise to **$9.50/GPU-hour effective
    2026-10-01** that `docs.nebius.com` does not yet show.

11. **Whether packing behaves at packs longer than 2,048.** NVIDIA validated packing and
    CP1/CP8 parity only at 2,048-token packs. The mechanism is boundary-correct (§4); the
    *tuning* at longer packs is untested by the vendor.

12. **Image-bearing examples end to end.** Coverage doc L167: *"Image training is supported;
    video training is not."* NeMo's VLM dataset builders and collators exist, but the map's own
    `lib/dataset.py` still has no image path ([#55](https://github.com/kreuzhofer/dgx-manager/issues/55)).
    Out of scope here; still the critical path for an actual run.

---

## §9 — Corrections to the record

| Claim on the map | Status |
|---|---|
| #71: *"one node, 8×H200 `gpu-h200-sxm`, `eu-north1`"* | **Moot.** The hardware is 2 × 8 B300. The *reasoning* survives — a frozen base has no gradients or optimizer state, so the budget collapses to weights + activations — but every memory number it was tight against is now slack. |
| #71: *"the FP8→BF16 load peak is the single most likely place a one-node run fails"* | **Retired.** True at 109.6 GiB against an H200's 140.4. At 268.6 GiB it is a 41% fill, and at EP16 a 20% fill. |
| 2026-09-04 re-check: *"FlashMLA compiles `sm_90a`/`sm_100f`"* | **Correct on that date, now superseded twice over.** `main` has emitted `sm_100a` + **`sm_103a`** since commit `07a108985` (2026-09-10). And `sm_100f` **already covered sm_103** by family compatibility — CPG Table 28, `compute_100f` → CC 10.0, **10.3**, 10.7. |
| Implied: *"sm_103 ≠ sm_100, so an sm_100f build does not cover B300"* | **Wrong**, and it is the third instance of this error shape on this map. The dangerous suffix is **`a`**, not `f`: `sm_100a` runs on CC 10.0 *"and no others"*. |
| 2026-09-04 re-check: the *"roughly 3-5x slower per training step"* B200/B300 datapoint | **Real, and correctly attributed — but filed under the wrong heading.** It is the cost of a *workaround* for silent NaN gradients in FLA's Blackwell Triton kernels, not a property of Blackwell hardware. The finding is the corruption; 3–5× is its price. |
| #67: *"sequence packing is fundamentally incompatible with indexer"* | **True of the `transformers` tree, false of NeMo's.** NeMo owns its own `glm5_next` implementation, packs at 2,048 tokens in the validated recipe, tests document isolation, and FLA's KDA kernels are varlen-native via `cu_seqlens`. The operator's Qwen3.8 packing problem does not transfer. |
| #69 / agent research: *"B300 is 270 GB per GPU"* (NVIDIA HGX B300 datasheet) | **Does not match this hardware.** `nvidia-smi` reports **275,040 MiB = 268.6 GiB = 288.4 GB decimal**, which matches `docs.nebius.com`'s "288 GB HBM3e" exactly. NVIDIA's datasheet states 270 GB (HGX B300) and 279 GB (GB300 NVL72); I cannot reconcile either with the measurement. **Plan on the measured number.** |
| Operator's flag: *"no torch 2.13.0 wheel ships sm_103 SASS"* | **Correct, and benign.** Release wheels are SASS-only with **no PTX at all** (`_PTX_ARCHES` is empty on release builds), so there is no JIT and no startup cost — the baseline `sm_100` SASS runs on sm_103 by major-version compatibility, and torch's own `_check_cubins` says so (`sm // 10 == cap_major`). The real cost is that torch is never sm_103-tuned and PyTorch CI has never run on sm_103. |
| #69: *"B300 is in `uk-south1` only (public), `eu-west2`/`us-north1` private"* | **Superseded.** `docs.nebius.com/overview/regions` today lists all three as public, and B300 rows carry no private marker. Nebius's own changelog has no "now public" entry for the latter two, so the docs are internally inconsistent; the Regions page is the current-state doc. Local NVMe (6 × 3.84 TB) is likewise now in all three regions, not just `uk-south1`. |
| #69: preset field `gpu_cluster_compatible` | **Field name is `allow_gpu_clustering`** (`nebius/api` `compute/v1/platform.proto`). |

---

## Reproducing these checks

```bash
# --- The arch-suffix crux (§2.1) ---
curl -sL https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html \
  | grep -A4 'compute_100f'          # Table 28: compute_100f -> 10.0, 10.3, 10.7
curl -sL https://docs.nvidia.com/cuda/parallel-thread-execution/index.html \
  | grep -B20 'Table 70 defines the architecture families'
curl -sL https://developer.nvidia.com/cuda-gpus | grep -B2 -A2 'B300'
curl -sL https://raw.githubusercontent.com/NVIDIA/cutlass/main/README.md | grep 'B300\|B200\|Spark'

# --- FlashMLA, then and now (§2.2-2.4) ---
curl -sL https://raw.githubusercontent.com/deepseek-ai/FlashMLA/main/setup.py | sed -n '25,48p'
curl -sL https://raw.githubusercontent.com/deepseek-ai/FlashMLA/b7643bd54521f563b839b98289b5cd048c062ba2/setup.py | sed -n '25,46p'

# --- The FLA / sm_103 bug class (§3) ---
gh api repos/triton-lang/triton/issues/10590 --jq '.title, .state, .body' | head -60
gh api repos/fla-org/flash-linear-attention/issues/1228 --jq '.title, .created_at'   # 2026-09-07, GB300 sm_103
gh api repos/fla-org/flash-linear-attention/issues/945  --jq '.title, .created_at'   # 2026-06-12, B300 sm_103
gh api repos/fla-org/flash-linear-attention/issues/727  --jq '.title'                # chunk_kda_bwd on B200
gh api repos/fla-org/flash-linear-attention/releases --jq '.[0].tag_name, .[0].published_at'   # v0.5.2, 2026-07-27
gh api -X GET repos/fla-org/flash-linear-attention/commits -f path=fla/ops/kda/chunk_bwd.py \
  --jq '.[] | "\(.commit.committer.date[0:10]) \(.commit.message|split("\n")[0])"' | head   # #1109 is 2026-08-09

git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git && cd flash-linear-attention
ls .github/workflows/                                    # no Blackwell runner
sed -n '24,40p' fla/ops/kda/chunk_bwd.py                 # NUM_WARPS [2,4,8] on Blackwell; SM100 filter is BK==32 only
sed -n '26,38p' fla/ops/common/chunk_delta_h.py          # the ONE gate KDA inherits (#945/PR#953)
sed -n '330,345p' fla/ops/gla/chunk.py                   # chunk_gla_fwd_kernel_o -- KDA's o-kernel, UNGATED
sed -n '10,17p'  fla/ops/kda/chunk_fwd.py                # proves KDA imports chunk_gla_fwd_o_gk, not chunk_o.py
grep -n 'num_warps in' fla/ops/kda/*.py                  # the ungated KDA config spaces
sed -n '155,165p' fla/utils/_device.py                   # IS_NVIDIA_BLACKWELL = major in (10, 12) -> covers sm_103

# --- torch's Triton pin and arch list (§2.7, §3.3) ---
curl -sL https://raw.githubusercontent.com/pytorch/pytorch/v2.13.0/.ci/docker/triton_version.txt   # 3.7.1
curl -s https://pypi.org/pypi/triton/json | python3 -c "import json,sys;print(json.load(sys.stdin)['info']['version'])"  # 3.8.0
curl -sL https://raw.githubusercontent.com/pytorch/pytorch/main/.ci/wheel/linux/build_env_setup.py | sed -n '70,120p'
# on the cluster, definitively:
python -c "import torch, triton; print(triton.__version__, torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"

# --- NeMo: packing, EP, dispatcher, launcher (§4-6) ---
git clone --depth 1 https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel   # HEAD 1358302c
sed -n '57,65p;67,76p;136,147p;149,163p;165,175p' docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx
sed -n '370,400p' nemo_automodel/components/models/glm5_next/layers.py    # cu_seqlens -> chunk_kda
# and back in the fla clone, the varlen boundary reset KDA relies on:
#   sed -n '86,98p' fla/ops/common/chunk_delta_h.py                       # IS_VARLEN -> per-document bos/eos
sed -n '287,353p' nemo_automodel/components/distributed/mesh_utils.py     # the only two EP constraints
sed -n '778,813p' nemo_automodel/components/moe/layers.py                 # dispatcher selection
sed -n '240,256p' nemo_automodel/components/moe/megatron/fused_a2a.py     # the ep_size>8 NVSHMEM wall
sed -n '59,77p;109,122p' nemo_automodel/components/models/common/cudnn_sparse_attention.py
grep -rn 'get_device_capability\|major <\|major >=' --include=*.py nemo_automodel/ | head -20
grep -n 'TE_CUDA_ARCHS\|TORCH_CUDA_ARCH_LIST' docker/Dockerfile           # 9.0 10.0 12.0 -- no 10.3
sed -n '50,70p;95,112p' docs/launcher/slurm.mdx
```

---

## Confidence summary

| Claim | Label |
|---|---|
| B300 = compute capability 10.3 / sm_103; 275,040 MiB/GPU; 8/node; NVSwitch; 800 Gb/s CX-8; NCCL verified across nodes | **measured** — the live cluster, corroborated by developer.nvidia.com/cuda-gpus and CUTLASS |
| `compute_100f` is compatible with CC 10.0, **10.3**, 10.7; `compute_100a` with 10.0 only | **measured** — CUDA Programming Guide §5.1.2.3 + Table 28, PTX ISA §11.1.2 + Table 70, nvcc §4.2.9.1.14 |
| FlashMLA `main` emits `sm_103a` (since `07a108985`, 2026-09-10); NeMo's pin `b7643bd` emits `sm_100f` | **measured** — both `setup.py` files, read today |
| **⇒ FlashMLA covers B300 at both revisions; no fallback penalty to pay** | **derived** from the two above |
| cuDNN-over-SDPA = +13.26% mean throughput | **measured by NVIDIA — on H100, at EP72/CP2. Not transferable to sm_103 without measuring.** |
| Triton 3.6/3.7 miscompile `tl.dot`-in-loop fp32-accumulator kernels on sm_103 at 4/8 warps above ~1 wave; envelope is wider than sm_100's; clean on Triton 3.8 | **measured** — triton#10590 on a "NVIDIA B300 SXM6 AC", third-party, config matrices published |
| torch 2.13.0 pins `triton==3.7.1`; Triton 3.8.0 released 2026-08-28 | **measured** — `.ci/docker/triton_version.txt` at tag `v2.13.0`, PyPI `requires_dist`, PyPI release index |
| FLA has Blackwell `num_warps` gates on `chunk_delta_h` (KDA inherits) and `chunk_o.py` (KDA does **not** use); KDA's own `chunk_gla_fwd_kernel_o`, `chunk_kda_bwd_kernel_dAv`, `chunk_kda_bwd_kernel_intra`, `chunk_kda_fwd_kernel_intra_sub_chunk`, `recompute_w_u_fwd_kda_kernel` are ungated at 4/8 warps | **measured** — FLA `main` `954438d1`, line-cited |
| FLA's KDA SM100 backward fix (#1109, 2026-08-09) is in **no released version** (0.5.2 shipped 2026-07-27); NeMo pins only `>=0.4.2` | **measured** — GitHub commit dates, PyPI, NeMo `pyproject.toml` L147 |
| FLA CI has no Blackwell runner | **measured** — `.github/workflows/` |
| **Whether FLA's KDA kernels are correct on sm_103 at GLM-5.3-Flash's shapes** | **unknown.** No public report either way. The precondition is present by construction. Settled only by running §3.6. |
| Packing is available and boundary-correct for KDA on the NeMo tree | **measured** — FLA `IS_VARLEN` kernels, NeMo's `cu_seqlens` pass-through, NVIDIA's own statement, NeMo's own isolation test |
| 313.85 B loaded = 584.6 GiB BF16; 36.5 GiB/GPU at EP16; 73.1 at EP8-on-8 | **derived** *(carried from #71; expert term exact, dense term ±1%)* |
| Load peak 54.8 GiB/GPU at EP16, 109.6 at EP8 | **derived** from `state_dict_mixin.py`'s own comment; simultaneity of destinations **unverified** |
| Non-weight memory at 16k / 32k | **extrapolated, ±2×, from one datapoint on a different architecture** — and now decision-irrelevant given 232 GiB of headroom |
| EP16 on 16 GPUs is legal; nothing in the mesh assumes one node; the only 2-node wall is DeepEP-without-NVSHMEM at `group.size() > 8` | **measured** — `mesh_utils.py`, `parallelizer.py`, `experts.py`, `fused_a2a.py`, line-cited |
| Anchored target globs reach all 45 LM layers and zero of `model.visual.*` | **measured** *(carried from #71, which executed NeMo's own `ModuleMatcher`)* |
| **s/step, tokens/s, MFU, wall-clock** | **unknown, and unchanged since #69.** NeMo's three published throughput numbers are mutually inconsistent; the 20% MoE MFU has no support anywhere; NeMo has no B300 FLOPS entry. **#72 settles it. Do not provision on a number nobody has measured — that is how this map got the wrong hardware once already.** |
