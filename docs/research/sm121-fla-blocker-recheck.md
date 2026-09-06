# Re-check: the sm_121 FLA / `causal_conv1d` "training blocker"

**Re-check date:** 2026-09-04 · **Original finding date:** 2026-08-31
**Under test:** issues [#66](https://github.com/kreuzhofer/dgx-manager/issues/66) / [#67](https://github.com/kreuzhofer/dgx-manager/issues/67), memory `gb10-cannot-train-kda-models`
**Method:** upstream source, official docs, repo build configs, real issues/PRs only. One claim was
settled empirically by downloading a wheel and parsing its CUDA fatbin headers (§2.1, method disclosed).

---

## BOTTOM LINE

**The 2026-08-31 conclusion is substantially wrong on its stated evidence, and the specific "sharp catch"
it was proudest of is a plain misreading. The Nebius decision survives — but on entirely different
grounds, and it should stop being justified by the kernel argument.**

| Claim | Verdict |
|---|---|
| **B.** NeMo's cuDNN "requires SM90 or later" excludes sm_121; "the version numbers do not order the way you'd guess" | **Overturned — misreading.** The gate is literally `if major < 9: raise`. sm_121 has major 12 and **passes**. The numbers order exactly the way you'd guess. It also applies to the 11 DSA layers, not the 34 KDA layers, and NeMo ships a portable fallback costing 13.26% throughput. (§1) |
| **A(b).** No `causal_conv1d` build for sm_121 | **Overturned — factually false, and was already false a year before the claim.** `Dao-AILab/causal-conv1d` has compiled `sm_121` since **2025-08-29** (PR titled "Add support Thor, **Spark** and GB300"). I downloaded NVIDIA's prebuilt aarch64/CUDA-13 wheel and **verified 3 `sm_121` cubins inside it**. (§2.1) |
| **A(a).** No FLA build for sm_121 | **Category error.** FLA is a pure-Python Triton library — PyPI ships only `py3-none-any` wheels, `setup.py` builds no extension. There are no per-arch builds *for any arch*, so "no sm_121 build" cannot be true or false. FLA's own source names GB10 and classes 12.1 as Blackwell. (§2.2) |
| **A, salvaged form.** FLA's kernels are broken on GB10 for training | **Partially supported, much narrower, unverified today.** Exactly one first-hand GB10 report exists (FLA #913, 2026-06-10): FLA **0.5.0** crashes on the first backward, FLA **0.4.2 works**. Auto-closed as stale, never confirmed fixed or unfixed. It is a **GDN** kernel; GLM-5.3-Flash uses **KDA**. (§2.3) |
| **3.** The `transformers` quote exists | **Yes, verbatim.** But it means *"silently falls back to a slower path"*, not *"refuses to run"* — and the same PR that wrote it **shipped a GB10 fast path**, hardware-verified on a DGX Spark, on 2026-06-19. That fast path does **not** cover GLM-5.3-Flash. (§3) |
| **Conclusion.** "Sparks cannot TRAIN GLM-5.3-Flash by any method, dtype-independent" | **Overturned as stated.** The kernel argument does not establish "cannot". Measured penalties for the torch fallback are 3–5× to ~7× per training step — **slow, not impossible**. (§4) |

**What this means for Nebius.** *Keep renting.* But the load-bearing reason must change. The venue
question was never actually close, and three grounds that this re-check did **not** disturb settle it
without any kernel argument:

1. **Weights alone do not fit.** 321.32 B params in BF16 ≈ 598.5 GiB versus ~498 GB pooled across four
   Sparks (from #67; not re-derived here). Before optimizer state, activations, or gradients.
2. **Topology.** EP must evenly divide 288 routed experts, and NeMo's *only* validated topologies are
   **9 nodes / 72 H100** (EP72/CP2) and **18 nodes / 144 H100s** — verified below, §1.4.
3. **NeMo does not support the shapes a Spark cluster would need.** Verbatim from its coverage doc:
   *"TP and PP are not supported for this model"* and *"Full-model single-GPU checkpoint loading and
   training are not supported."*

The honest revision to the memory note: **"the Sparks are not a training venue for GLM-5.3-Flash because
the model does not fit and the only framework that trains it is validated at 72–144 H100s"** — not
"because sm_121 has no kernels". The kernel claim is wrong in its `causal_conv1d` half, meaningless in
its FLA half, and its one real residue (an FLA 0.5.x backward bug on Blackwell, with a 0.4.2 downgrade
workaround) is a *bug with a workaround*, not an architectural impossibility.

**Consequence for the wider rule.** The memory's "how to apply" instruction — *"If it has
linear-attention/SSM layers, the answer is no on sm_121 regardless of parameter count or quantization"* —
should be **withdrawn**. It would have wrongly ruled out training a small GDN/KDA model on the Sparks,
which is a thing HuggingFace, NVIDIA's own Jetson AI Lab wheel index, and at least one DGX Spark owner
in the FLA issue tracker are all demonstrably doing.

---

## §1 — Claim B: the "SM90 or later" reading

**Verdict: overturned. This is a misreading, not a sharp catch.**

### 1.1 What NeMo Automodel's doc actually says, and what it applies to

Two sentences, both real, both quoted correctly by the 2026-08-31 researcher — but read out of context.

From `docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx`, under **"Attention Backends"**
([source](https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx),
[rendered](https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thudm/glm-5-3-flash)),
lines 59–65:

> - KDA layers use Flash Linear Attention (FLA) kernels. For CP, FLA carries the
>   recurrent state across contiguous sequence shards while preserving packed
>   document boundaries.
> - Sparse DSA layers support an SDPA numerical-reference path.
> - **On SM90 or later, `backend.attn: cudnn` uses FlashMLA for the sparse forward
>   pass and cuDNN Frontend for backward.** This is the backend selected by the
>   published recipe.

And under **"Current Scope"**, lines 175–176:

> - **The cuDNN sparse-attention path requires SM90 or later, cuDNN Frontend, and
>   FlashMLA. Use `backend.attn: sdpa` as the portable reference path.**

Note the structure the original read missed: these are **three separate bullets about three separate
things**. "SM90 or later" is scoped to `backend.attn: cudnn` — the **sparse DSA** path, which is the
**11 DSA layers**. The **34 KDA layers** are the first bullet, and they use FLA regardless of which
`backend.attn` is chosen. The doc says so explicitly in its install section (line 116):

> If FlashMLA or cuDNN Frontend is unavailable, change
> `model.backend.attn` to `sdpa`; **KDA still requires the FLA dependency.**

So even if "SM90 or later" *had* excluded sm_121, it would have said nothing about the 34-of-45 KDA
layers that the whole kernel argument was built on.

### 1.2 What the gate is in source — decisive

`nemo_automodel/components/models/common/cudnn_sparse_attention.py`, lines 74–77
([source](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/common/cudnn_sparse_attention.py)):

```python
    major, minor = torch.cuda.get_device_capability(device)
    if major < 9:
        raise RuntimeError(f"{operation} requires SM90 or later, got SM{major}{minor}.")
    return major, minor
```

GB10 reports `(12, 1)`. `12 < 9` is `False`. **sm_121 passes this gate.** The identical gate appears in
`nemo_automodel/components/models/glm_moe_dsa/kernels/cudnn_dsa.py`.

The 2026-08-31 note said *"sm_121 is not sm_90; the version numbers do not order the way you'd guess."*
NVIDIA compute capabilities are ordered lexicographically by `(major, minor)`, and NVIDIA's own code
here orders them exactly that way. The claim is wrong in the general case, and wrong in this specific
line of code.

### 1.3 What cuDNN itself requires — also permissive

NVIDIA's own cuDNN support matrix lists compute capability **12.1** as supported (CUDA ≥ 12.8, Linux
driver ≥ 570.26), alongside 12.0, 10.3, 10.0, 9.0, 8.9, 8.6, 8.0, 7.5
([cuDNN backend support matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html)).

The cuDNN Attention operations page
([docs](https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html)) names **SM121**
explicitly as "Blackwell Consumer", and states the floor for the fused SDPA op:

> "cudnn SDPA operation requires SM80 (Ampere) or newer architectures and cuda toolkit 12.x or newer."

Only the **FP8** variant carries a Hopper floor ("Requires Hopper (SM90) or newer architecture"), and the
DSA path here is not FP8.

### 1.4 The cuDNN path *is* unavailable on GB10 — for a different reason, and it doesn't matter

Having said all that: the cuDNN DSA backend genuinely will not run on a Spark. Not because of the SM90
gate, but because it hard-requires **DeepSeek FlashMLA**, whose `setup.py` emits exactly two
architectures ([source](https://github.com/deepseek-ai/FlashMLA/blob/main/setup.py), `get_arch_flags()`):

```python
    if not DISABLE_SM100:
        arch_flags.extend(["-gencode", "arch=compute_100f,code=sm_100f"])
    if not DISABLE_SM90:
        arch_flags.extend(["-gencode", "arch=compute_90a,code=sm_90a"])
```

No `sm_120`, no `sm_121`. Its `csrc/` tree is `sm90/` and `sm100/` only.

**And NeMo tells you the price of not having it**, from its own A/B on 72 H100s (doc lines 150–164):

| Backend | mean tok/s | median tok/s | peak mem | wall |
|---|---|---|---|---|
| SDPA | 8,378.15 | 8,417.33 | 57.160 / 57.68 GiB | 37:15 |
| cuDNN | 9,489.04 | 9,538.12 | 57.185 / 57.78 GiB | 33:18 |

> "The cuDNN path improved mean throughput by 13.26% and median throughput by [...]"

**A 13.26% throughput difference is not a venue decision.** The original write-up promoted an optional
13% accelerator into a hard blocker.

*(Also verified in passing from the same doc: the validated topologies are "9 nodes / 72 H100 GPUs"
(EP72/CP2) and "18 nodes / 144 H100s" (CP1 and CP8 parity runs) — the real reason the Sparks lose.)*

---

## §2 — Claim A: is there an sm_121 build today?

### 2.1 `Dao-AILab/causal-conv1d` — yes, and since 2025

**Documentation/source says:** `setup.py` on `main`
([source](https://github.com/Dao-AILab/causal-conv1d/blob/main/setup.py), lines 179–199) compiles:

```python
        cc_flag.append("arch=compute_75,code=sm_75")
        cc_flag.append("arch=compute_80,code=sm_80")
        cc_flag.append("arch=compute_87,code=sm_87")
        if bare_metal_version >= Version("11.8"):
            cc_flag.append("arch=compute_90,code=sm_90")
        if bare_metal_version >= Version("12.8"):
            cc_flag.append("arch=compute_100,code=sm_100")
            cc_flag.append("arch=compute_120,code=sm_120")
        if bare_metal_version >= Version("13.0"):
            cc_flag.append("arch=compute_103,code=sm_103")
            cc_flag.append("arch=compute_110,code=sm_110")
            cc_flag.append("arch=compute_121,code=sm_121")   # <-- sm_121
```

That block was added by commit `3d19ec779b`, **merged 2025-08-29**, from PR
[#71 "\[NVIDIA\] Add support Thor, Spark and GB300"](https://github.com/Dao-AILab/causal-conv1d/pull/71).
"Spark" in the title is the DGX Spark. **The claim was already a year out of date when it was written.**
The PR body is a single line: a link to a prebuilt wheel index,
`https://pypi.jetson-ai-lab.io/sbsa/cu130/causal-conv1d/1.5.2`.

The only caveat is real but minor: PyPI ships `causal-conv1d` as an **sdist only** (latest 1.7.0,
2026-08-20 — [PyPI JSON](https://pypi.org/pypi/causal-conv1d/json)), so a plain `pip install` compiles
from source and needs **CUDA ≥ 13.0** for the sm_121 branch to fire.

**Empirically verified (this is not doc-derived).** NVIDIA's Jetson AI Lab index at
`https://pypi.jetson-ai-lab.io/sbsa/cu130/causal-conv1d/` serves
`causal_conv1d-1.5.3-cp312-cp312-linux_aarch64.whl` (103 MB). I downloaded it, unzipped
`causal_conv1d_cuda.cpython-312-aarch64-linux-gnu.so`, walked its CUDA fatbin containers (magic
`0xBA55ED50`) and read the `smVersion` field of each entry header. Result:

```
fatbin containers: 3
entry kinds (1=PTX, 2=ELF): {2: 27}
SM versions found in fatbin entry headers:
  sm_75: 3   sm_80: 3   sm_87: 3   sm_90: 3   sm_100: 3
  sm_103: 3  sm_110: 3  sm_120: 3  sm_121: 3
```

**A prebuilt, aarch64, CUDA-13 `causal_conv1d` wheel containing `sm_121` device code exists and is
served from an NVIDIA-run index.** *(Method caveat: I parsed the fatbin entry-header layout directly
rather than using `cuobjdump`, which is not installed here. The values decode to exactly the nine
architectures `setup.py` asks for, in order, which is strong self-consistency, but it is a parse, not a
`cuobjdump --list-elf` run.)*

### 2.2 `fla-org/flash-linear-attention` — the question doesn't apply

FLA has **no compiled extension**. Its `setup.py`
([source](https://github.com/fla-org/flash-linear-attention/blob/main/setup.py)) is 33 lines: read
version, read README, `find_packages()`, `setup()`. No `CUDAExtension`, no `cc_flag`, no
`TORCH_CUDA_ARCH_LIST`. Its `pyproject.toml`
([source](https://github.com/fla-org/flash-linear-attention/blob/main/pyproject.toml)) describes it as:

> description = "Fast **Triton-based** implementations of causal linear attention"

and PyPI confirms it: every release from 0.4.0 through 0.5.2 ships exactly
`flash_linear_attention-<v>-py3-none-any.whl` plus an sdist
([PyPI JSON](https://pypi.org/pypi/flash-linear-attention/json)). Same for the newer slim
`fla-core` distribution ([PyPI](https://pypi.org/pypi/fla-core/json)).

**Triton JIT-compiles for whatever architecture the driver reports, at first call.** There is no build
matrix to be absent from. The `fla/ops/kda/` directory — the exact ops GLM-5.3-Flash needs — is nine
`.py` files and a `backends/` dir; no `.cu`, no `csrc/`
([tree](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda)).

**FLA also does not need Dao's `causal_conv1d` at all.** It has its own Triton convolution, and that is
the default. From `fla/modules/conv/causal_conv1d.py`
([source](https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/conv/causal_conv1d.py)):

```python
        backend (Optional[str]):
            Specifies the backend to use for the convolution operation. Supported values are
            `'cuda'` 、 `'triton'` and `'mix'`.  Default: `'triton'`
```

and `causal-conv1d` appears in `pyproject.toml` only as an **optional extra**:
`conv1d = ["causal-conv1d>=1.4.0"]`.

**FLA knows GB10 by name.** `fla/utils/_device.py`
([source](https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils/_device.py), lines
146–149):

```python
IS_NVIDIA_SM100 = (IS_NVIDIA and torch.cuda.get_device_capability()[0] == 10)
# NOTE: exactly 12.0 — 12.1 (GB10) is a different target that FlashQLA rejects at import time.
IS_NVIDIA_SM120 = (IS_NVIDIA and torch.cuda.get_device_capability() == (12, 0))
IS_NVIDIA_BLACKWELL = (IS_NVIDIA and torch.cuda.get_device_capability()[0] in (10, 12))
```

sm_121 **is** `IS_NVIDIA_BLACKWELL`, which triggers FLA's Blackwell-specific `global_scratch` allocator
registration (line 177–181). The one thing 12.1 is excluded from is `IS_NVIDIA_SM120`, used solely by
`fla/ops/gated_delta_rule/backends/flash_qla.py` — an **optional** TileLang backend from the Qwen team
that falls back to Triton when absent, and that only touches GDN, not KDA. That change (`in (10, 12)`)
landed in [PR #940](https://github.com/fla-org/flash-linear-attention/pull/940), merged **2026-06-06**.

### 2.3 The one real residue: an FLA 0.5.x backward bug on Blackwell, seen once on a Spark

This is the part of Claim A that deserved to survive, in a much narrower form.

[FLA issue #913](https://github.com/fla-org/flash-linear-attention/issues/913) contains a first-hand
DGX Spark report from `@NvMayMay`, 2026-06-10 — **this is empirical, on real GB10 hardware, by a third
party**:

> "Confirming this on a second Blackwell variant: same crash in `prepare_wy_repr_bwd_kernel`
> (`fla/ops/gated_delta_rule/wy_fast.py`), same `fla 0.5.0` + `torch 2.11.0` + `triton 3.6.0` +
> `python 3.12`, also training Qwen3.5. Difference from the original report: **NVIDIA GB10 / DGX Spark
> (sm_121, aarch64, unified memory)**, CUDA 13.0, and plain `transformers` 5.8.1 (no ms-swift/DeepSpeed)
> [...] Repro: training Qwen3.5-122B-A10B (hybrid; 36 of 48 layers GatedDeltaNet) through the
> transformers fla integration. **Forward completes; the first backward crashes** during Triton autotune
> of the wy-representation backward kernel. Pinning **`flash-linear-attention==0.4.2`** (everything else
> identical) **compiles and runs clean. `0.5.0` crashes.**"

Read carefully, this report is itself a refutation of "no FLA build for sm_121": someone is **training a
hybrid linear-attention model on a DGX Spark through FLA**, the forward pass completes, and one version
works. You cannot crash in a Triton kernel that does not exist for your architecture.

What it does establish: FLA **0.5.0** has a GB10 regression in the GDN backward path. Status today:

- The issue was **auto-closed as stale** on 2026-08-16 by a bot, not fixed-and-closed. No maintainer
  ever answered NvMayMay's three questions.
- The suspected fix, [PR #911](https://github.com/fla-org/flash-linear-attention/pull/911)
  ("Large-offset Pointer Arithmetic in Blackwell GPUs", int64 offset promotion in `wy_fast.py`), merged
  **2026-05-23** — i.e. it is in **0.5.1 and 0.5.2**, both released *after* NvMayMay tested 0.5.0.
  **Nobody has retested on a Spark.**
- Two more fixes of exactly that class landed after 0.5.2:
  [#1173](https://github.com/fla-org/flash-linear-attention/pull/1173) "Anchor loop-derived token offsets
  to int64 in chunk kernels" (2026-08-26) and #1188 (2026-08-28).
- It is a **GDN** kernel (`fla/ops/gated_delta_rule/wy_fast.py`). GLM-5.3-Flash is **KDA**
  (`fla/ops/kda/`). Related code, different files. Whether KDA hits the same fault on sm_121 is
  **unknown** — see §6.
- There are currently **zero open** FLA issues mentioning Blackwell, sm_121, or "misaligned".

---

## §3 — Claim 3: the `transformers` quote

**Verdict: the quote is real and verbatim. Its meaning was inverted.**

`src/transformers/integrations/hub_kernels.py`, line 160
([source](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py)):

```python
            # GB10/SM121 GDN fast path (no fla/causal_conv1d build there); dense and MoE share it.
            "Qwen3_5GatedDeltaNet": {
                Device(
                    type="cuda",
                    properties=CUDAProperties(min_capability=121, max_capability=121),
                ): LayerRepository(
                    repo_id="Atlas-Inference/gdn",
                    layer_name="Qwen3_5GatedDeltaNet",
                    revision="ef12347fc77d6ddf1cb72c0bd0af1c7d6cc69172",
                    trust_remote_code=True,
                ),
            },
```

The comment is a **parenthetical justifying why a workaround exists**, and the code around it *is the
workaround*. It never says the model refuses to run.

`docs/source/en/model_doc/qwen3_5.md` line 77, same repo, says exactly what happens instead:

> "On NVIDIA GB10 (compute capability 12.1 / SM121) neither `causal_conv1d` nor `fla` ship an SM121
> build, **so the DeltaNet path always falls back to the slow PyTorch reference.** Passing
> `use_kernels=True` [...] swaps the Gated DeltaNet conv1d and delta-rule cores for a
> compute-capability-gated Hub kernel ([`Atlas-Inference/gdn`]) [...] The kernel is numerically faithful
> to the fallback (identical greedy output) and speeds up prefill."

with a **measured GB10 table** (`Qwen/Qwen3.6-27B`, bf16, 1024-token prompt, 256-token greedy decode):

| `use_kernels` | TTFT (prefill) | Decode |
|---|---|---|
| `False` (PyTorch fallback) | 1.66 s | 4.11 tok/s |
| `True` (`Atlas-Inference/gdn`) | 1.11 s (1.49x faster) | 4.14 tok/s |

The dispatch is three-tier, documented in the same file's `use_kernel_func_from_hub_with_fallback`
docstring: *"1. Hf kernels (if requested) 2. Original package 3. Torch only path"*. **Falls back**,
never raises.

**The provenance of the "no SM121 build" phrase matters.** It comes from
[PR #46423, "Add GB10/SM121 Hub-kernel path for Qwen3.6 Gated DeltaNet"](https://github.com/huggingface/transformers/pull/46423),
merged **2026-06-19** — i.e. **ten weeks before** the 2026-08-31 finding cited it. The PR body opens:

> "On a DGX Spark (GB10, compute capability 12.1) the `fla` and `causal_conv1d` fast paths have no SM121
> build, so Qwen3.6 falls back to the slow pure-torch Gated DeltaNet (`is_fast_path_available=False`)."

and then reports **hardware verification on a Spark**:

> "**Verified on GB10 (SM121), bf16:** Layer parity vs the torch fallback through a real `DynamicCache`
> (prefill + 8 decode) [...] 27B `0.999970`, 35B-A3B `0.999967` reproduced on a second, clean box [...]
> Full-model greedy `generate()` [...] identical token IDs vs the torch fallback."

So the phrase the 2026-08-31 note treated as the ceiling was written by someone who was, in the same
breath, **shipping sm_121 kernels**. Those kernels come from Atlas Inference, who built and published
four SM121-only kernel packages
([`Atlas-Inference/gdn`](https://huggingface.co/Atlas-Inference/gdn) and three NVFP4 packages), pinned
`cuda-capabilities = ["12.1"]`, after HuggingFace added 12.1 to the kernel-builder arch matrix
([kernels#576](https://github.com/huggingface/kernels/pull/576)) specifically to unblock them
([kernels-community#866](https://github.com/huggingface/kernels-community/issues/866), May 2026).
Their own summary of why the phrase was true at all is worth recording verbatim:

> "SM121 has real silicon gaps that force algorithmic changes. So without a hardware E2M1 pack
> (`cvt.rn.satfinite.e2m1x2.f32` is missing), NVFP4 GEMM/MoE uses a software E2M1 path. ~1 tok/s without
> it, 35+ with. Also, there's no multi-CTA clusters (ClusterShape forced 1x1x1) [...] the FP4/FP8 MoE,
> GDN/SSM, and MTP-verify kernels are basically SM121-specific" — @AzeezIsh, 2026-05-18

**One honest complication that cuts the other way.** The Atlas GB10 fast path is registered only for
`Qwen3_5GatedDeltaNet` and `Qwen3_5MoeGatedDeltaNet`. Reading
`src/transformers/integrations/hub_kernels.py` end to end, **there is no SM121-gated entry for
`chunk_kda` / `fused_recurrent_kda`** — those map unconditionally to `kernels-community/fla`. And
`docs/source/en/model_doc/glm5_next.md` carries **no GB10 note at all**. So GLM-5.3-Flash specifically
would **not** get an Atlas kernel on a Spark; it would get the FLA Triton path or the torch reference.

---

## §4 — Q5: if the fallback is used, how bad is it?

**Slow. Not unusable.** Four independent numbers, none of them "cannot run":

1. **`transformers` source**, `hub_kernels.py` (the code comment guarding the new fallback warning):
   > "These torch paths are readable references, not fast kernels, so their runtimes are significantly
   > slower: **for `chunk_gated_delta_rule` the gap is more than an order of magnitude on an H100.**"

2. **Real SFT workload**, [transformers issue #48148](https://github.com/huggingface/transformers/issues/48148)
   (2026-08-20) — a regression report where the FLA path was accidentally bypassed:
   > "In our Qwen3.8-27B SFT workload this changed throughput from approximately **25-30 s/it to
   > 169-200 s/it** with the rest of the environment held fixed."

   That is a **~6-7× per-iteration** penalty, measured on a real fine-tune, torch-reference vs FLA.

3. **Real training, Blackwell**, FLA #913 (`@tommyliautaud`, on B200/B300, using the torch reference as a
   workaround):
   > "That path converges for us on B200/B300, but is **roughly 3-5x slower per training step** than the
   > FLA fused path on H200."

   Note "**converges**" — the fallback is numerically correct.

4. **GB10 inference**, HF's own table (§3): 1.49× on prefill, ~0 on decode, end-to-end on a 27B model.
   Lower than the per-kernel numbers because only the linear-attention layers are affected.

**Why it is slow, from the code.** The torch reference in
`src/transformers/models/glm5_next/modeling_glm5_next.py` (`chunk_kimi_delta_attention`, lines 483–581)
is a genuine *chunked* algorithm — `chunk_size=64`, an outer loop over `T/64` chunks and a 63-iteration
inner triangular-inverse loop — not a per-token recurrence. But it casts **everything to `torch.float32`**
and materialises a `[B, H, NT, 64, 64]` `decay_mask` plus a same-shaped `attn` in fp32 per layer. On a
GB10 that is ~273 GB/s of unified LPDDR5X, which is the worst possible place to pay a memory-traffic
penalty.

So the pure-PyTorch fallback is **~3–7× slower per step and numerically fine** — a real cost, and a
perfectly good reason to prefer datacentre GPUs on *economics*. It is not the "cannot train by any
method" the memory records.

---

## §5 — What changed since 2026-08-31

The original note warned its answer had a short half-life. In fairness: **most of the evidence that
refutes it predates 2026-08-31** (causal-conv1d sm_121: 2025-08-29; FLA `IS_NVIDIA_BLACKWELL`:
2026-06-06; the transformers GB10 Hub kernel: 2026-06-19). Only the following actually landed in the
four-day window:

| Date | What | Why it matters |
|---|---|---|
| 2026-09-02 | transformers [#48443](https://github.com/huggingface/transformers/pull/48443) — "Enable functions into kernels registry and allow non inheritance" | Reworks the kernel registry the whole dispatch depends on. |
| 2026-09-03 | transformers [#48185](https://github.com/huggingface/transformers/pull/48185) — "Warn once when a hub-kernel function falls back to its reference PyTorch path" | **Directly relevant.** Silent degradation becomes an explicit `logger.warning_once`. You will now *see* whether you are on the slow path instead of inferring it from throughput. |
| 2026-09-03 | transformers [#48221](https://github.com/huggingface/transformers/pull/48221) — "Support nested FLA kernel imports for `fla-core`" | **Directly relevant.** Fixes `fla.ops.kda.chunk_kda` (named explicitly in the PR) silently resolving to `None` and falling through to torch even when FLA *is* installed. Closes #48148 — the 25→170 s/it regression above. |
| 2026-08-25 → 2026-09-04 | FLA `main`: version bumped to **0.6.0**, ~55 commits, incl. #1173/#1188 int64 offset anchoring (the Blackwell fault class), [#1212](https://github.com/fla-org/flash-linear-attention/pull/1212) "platform-graph capture and replay for KDA chunk" (2026-09-03) | Active KDA work. **0.6.0 is not released** — GitHub latest is still v0.5.2 (2026-07-27) and PyPI latest is 0.5.2. |
| — | **NeMo Automodel: nothing.** `glm5_next` last touched 2026-08-28 (#3699), its doc 2026-08-29 (#3744). No GB10/sm_121 issues exist in that repo. | The venue conclusion there is unchanged. |
| — | **`causal-conv1d`: nothing.** `setup.py` last changed 2026-05-08. PyPI 1.7.0 (sdist) 2026-08-20. No sm_121/Spark issues. | Stable; sm_121 support unchanged. |

Net: the window changed nothing about the *substance*, and two of the three transformers PRs make the
slow-path situation **better and more observable**, not worse.

---

## §6 — What I could NOT determine from the web (needs hardware)

Everything below requires a DGX Spark. All of it is currently **unknown**, and the 2026-08-31 note
asserted through several of these gaps.

1. **Does FLA's KDA path (`fla/ops/kda/chunk.py`) run correctly on sm_121, forward and backward?**
   No report of any kind exists — the only GB10 datapoint anywhere is a **GDN** kernel. This is the
   single question the whole argument turns on and nobody on the internet has answered it.
   *Test:* `chunk_kda` fwd+bwd on a Spark against `fla/ops/kda/naive.py`, at several `(B, T, H, D)`.

2. **Is the FLA 0.5.0 GB10 backward crash fixed in 0.5.2 / `main`?** PR #911 (the likely fix) shipped in
   0.5.1, released *after* the only GB10 test. The issue was closed by a stale bot, not by a fix.
   *Test:* NvMayMay's exact repro (Qwen3.5 GDN, first backward) on 0.4.2 / 0.5.2 / `main`.

3. **Does the sm_121 `causal_conv1d` wheel actually work?** I proved `sm_121` cubins are *in the binary*.
   I did not run them. Compiled ≠ correct.
   *Test:* `pip install --extra-index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 causal-conv1d`, then
   `causal_conv1d_fn` fwd/bwd against `F.conv1d`.

4. **What does the fallback actually cost on a GB10, for training?** Every number in §4 is from H100 /
   H200 / B200 or from GB10 *inference*. Nobody has published a GB10 training-step comparison, and the
   Spark's unified 273 GB/s memory is exactly where an fp32-heavy reference implementation should hurt
   most. It could be worse than 7×.

5. **Does `kernels-community/fla` publish an aarch64 / sm_121 variant?** HuggingFace's model/kernel API
   returns 401 to unauthenticated requests from here, so I could not enumerate its build matrix. This
   matters only for the `use_kernels=True` path; the pip FLA path is unaffected either way.
   *Test:* `from kernels import get_kernel; get_kernel("kernels-community/fla")` on a Spark.

6. **Whether the `Atlas-Inference/gdn` kernel supports backward at all.** It is registered without a
   `Mode` split (so nominally both training and inference), but every published verification is
   forward-only (prefill parity, greedy `generate()`). Irrelevant to GLM-5.3-Flash — which has no Atlas
   entry — but relevant if the Sparks are ever pointed at a Qwen3.5/3.6 fine-tune.

7. **Untested, unchallenged, and still the actual reason for Nebius:** the 598.5 GiB BF16 weight figure
   and the 498 GB pooled-memory figure from #67. I did **not** re-derive these. I did re-confirm from
   `zai-org/GLM-5.3-Flash`'s own `config.json` that `num_hidden_layers: 45` with
   `layer_types` = **34 `linear_attention` + 11 `deepseek_sparse_attention`**, and `n_routed_experts: 288`
   — so the 34/45 and the EP-divisibility premises are correct.

---

## Source index

**NVIDIA / NeMo**
- NeMo Automodel GLM-5.3-Flash coverage — https://github.com/NVIDIA-NeMo/Automodel/blob/main/docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx · rendered: https://docs.nvidia.com/nemo/automodel/model-coverage/vision-language-models/thudm/glm-5-3-flash
- NeMo Automodel cuDNN sparse attention (the `major < 9` gate) — https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/models/common/cudnn_sparse_attention.py
- cuDNN backend support matrix (lists CC 12.1) — https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/support-matrix.html
- cuDNN Attention operations (SM121 = "Blackwell Consumer"; SDPA floor SM80) — https://docs.nvidia.com/deeplearning/cudnn/latest/operations/Attention.html
- Jetson AI Lab SBSA/CUDA-13 wheel index — https://pypi.jetson-ai-lab.io/sbsa/cu130/causal-conv1d/

**causal-conv1d / FlashMLA**
- setup.py arch list — https://github.com/Dao-AILab/causal-conv1d/blob/main/setup.py
- PR #71 "Add support Thor, Spark and GB300" (2025-08-29) — https://github.com/Dao-AILab/causal-conv1d/pull/71
- PyPI (sdist only) — https://pypi.org/pypi/causal-conv1d/json
- FlashMLA setup.py (`sm_90a` + `sm_100f` only) — https://github.com/deepseek-ai/FlashMLA/blob/main/setup.py

**flash-linear-attention**
- setup.py / pyproject.toml — https://github.com/fla-org/flash-linear-attention/blob/main/setup.py · https://github.com/fla-org/flash-linear-attention/blob/main/pyproject.toml
- `fla/utils/_device.py` (GB10 named; Blackwell = major in (10,12)) — https://github.com/fla-org/flash-linear-attention/blob/main/fla/utils/_device.py
- `fla/modules/conv/causal_conv1d.py` (default `backend='triton'`) — https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/conv/causal_conv1d.py
- `fla/ops/kda/` — https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/kda
- Issue #913 (the only first-hand GB10 datapoint) — https://github.com/fla-org/flash-linear-attention/issues/913
- PR #911 (Blackwell int64 pointer fix) — https://github.com/fla-org/flash-linear-attention/pull/911
- PR #940 (Blackwell = major 12) — https://github.com/fla-org/flash-linear-attention/pull/940
- PR #1173 (int64 anchoring, post-0.5.2) — https://github.com/fla-org/flash-linear-attention/pull/1173
- PyPI (`py3-none-any` only) — https://pypi.org/pypi/flash-linear-attention/json · https://pypi.org/pypi/fla-core/json

**transformers**
- `integrations/hub_kernels.py` (the GB10 comment, the KDA registry, the >10× note) — https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/hub_kernels.py
- `docs/source/en/model_doc/qwen3_5.md` (GB10 measured table) — https://github.com/huggingface/transformers/blob/main/docs/source/en/model_doc/qwen3_5.md
- `models/glm5_next/modeling_glm5_next.py` (the torch KDA reference) — https://github.com/huggingface/transformers/blob/main/src/transformers/models/glm5_next/modeling_glm5_next.py
- PR #46423 (GB10 Hub kernel, hardware-verified) — https://github.com/huggingface/transformers/pull/46423
- Issue #48148 (25→170 s/it) — https://github.com/huggingface/transformers/issues/48148
- PR #48221 / #48185 (2026-09-03, post-dating the original finding) — https://github.com/huggingface/transformers/pull/48221 · https://github.com/huggingface/transformers/pull/48185

**HuggingFace kernels**
- kernels-community#866 (Atlas SM121 kernel contribution thread) — https://github.com/huggingface/kernels-community/issues/866
- kernels#576 (adds CC 12.1 to the builder arch matrix) — https://github.com/huggingface/kernels/pull/576
- `Atlas-Inference/gdn` — https://huggingface.co/Atlas-Inference/gdn

**Model**
- `zai-org/GLM-5.3-Flash` `config.json` (45 layers = 34 linear + 11 DSA, 288 experts) — https://huggingface.co/zai-org/GLM-5.3-Flash/resolve/main/config.json
