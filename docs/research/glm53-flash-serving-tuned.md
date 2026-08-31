# Can a tuned GLM-5.3-Flash be served on our stack?

> Research for [#68](https://github.com/kreuzhofer/dgx-manager/issues/68). Part of the
> [GLM-5.3-Flash wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/65).
>
> **Sources read 2026-08-31.** vLLM was read as source, not docs, at `main` commit
> `85c1365` and at release tags `v0.25.0`, `v0.26.0`, `v0.27.0`, `v0.27.1`, `v0.28.0`.
> The GLM-5.3-Flash model implementation is **not in any vLLM release** — it lives in
> open PR [#53906](https://github.com/vllm-project/vllm/pull/53906), read at branch head
> and at its first commit `933876c`.
>
> Every claim below is labelled **read** (I can quote the line), **derived** (arithmetic
> over a config I read), or **measured** (on this fleet, from the recipe's own notes).
> The distinction matters here because the headline answer is a *code* fact, and the
> expensive part of the answer is *arithmetic* that nobody has checked against hardware.

---

## Verdict

**Route 1 — runtime LoRA against the NVFP4 base, no merge — does not work today. But it
fails for one reason, and it is not any of the four the ticket expected.**

Every kernel-level thing #68 worried about is already solved upstream, and has been since
**v0.25.0 (2026-07-11)** — three releases before the one we run:

- LoRA over a **quantized** base is quantization-scheme agnostic by construction, and
  compressed-tensors (which is what NVFP4 *is* on disk) is explicitly handled.
- LoRA on **MoE expert weights** is no longer attention-only. `FusedMoEWithLoRA` exists,
  with Triton shrink/expand kernels that run per expert.
- **`--moe-backend marlin`, the flag we pin as mandatory, is the LoRA-capable backend.**
  `NvFp4MoeBackend.MARLIN` resolves to `MarlinExperts`, which mixes in `LoRAExpertsMixin`.
  Better: the kernel oracle *filters out* any expert kernel that cannot do LoRA, so the
  FLASHINFER_CUTLASS backend that silently corrupts our output is unreachable under
  `--enable-lora`.
- **MTP does not conflict with LoRA.** The single spec-decode/LoRA incompatibility in the
  entire tree is `enable_adaptive_verification`, which we do not use.

What blocks it is the model file:

```python
# vllm/models/glm5next/nvidia/model.py:866  (PR #53906, unchanged since 933876c)
class Glm5NextForCausalLM(
    nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid
):
```

No `SupportsLoRA`. No `packed_modules_mapping`. No `lora_skip_prefixes`. The peer model —
`DeepseekV4ForCausalLM`, also MLA + huge MoE + FP4 experts + an MTP head — declares all
three and is served with LoRA today. GLM-5.3-Flash's implementation simply has not had
that work done, because it is a **five-day-old unmerged PR**.

**And the failure is not clean, which is the part worth carrying forward.** The
multimodal wrapper we actually serve inherits from GLM-4V:

```python
# model.py:962
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
```

`Glm4vForConditionalGeneration` **is** `SupportsLoRA`. So `--enable-lora` passes vLLM's
interface gate by inheritance, the server starts, and the model advertises LoRA support it
was never given — using GLM-4V's `packed_modules_mapping` (`qkv_proj: [q_proj, k_proj,
v_proj]`), naming modules that do not exist in `glm5_next`, which is NoPE MLA. The
consequence, traced through the loader in
[§6](#6-would---enable-lora-at-least-start-today), is that the fused input projection on
**all 45 layers** — MLA's `fused_qkv_a_proj` and KDA's `in_proj_qkvbfg_a` — is silently
left unwrapped, with no error and no warning. This is the same class of
silent-configuration failure the recipe already documents twice (`--moe-backend`,
`reasoning_effort`).

### So: route 2, costed

Merge-then-requantize works, and **we own exactly one machine that can execute it** —
`dgx-spark-04`, on 714 GB of free local NVMe, at a 92% fill. Nothing else we have can even
hold the base. Full derivation in [Route 2](#route-2--merge-then-re-quantise); the shape:

| step | on | cost | risk |
|---|---|---|---|
| Download `zai-org/GLM-5.3-Flash-BF16` (**not** `…/GLM-5.3-Flash`, which is FP8) | spark-04 local NVMe | **642.65 GB**, 120 shards | none |
| Merge the adapter **in place**, shard by shard | same disk, ~1 shard RAM | overwrite, no extra disk | **no first-party tool.** PEFT's offload merge is CI-tested but never at 320B; the only true streaming merger is third-party (`unsloth-zoo`) and untested on `glm5_next` |
| Re-quantise with `llm-compressor` `model_free_ptq`, deleting BF16 shards as they convert | same disk, ~16 GB/worker | writes **~190–198 GB** | needs a **dev-mode source install** — released 0.13.0 pins `transformers<=5.14.1`, and `glm5_next` first exists in **5.16.1** |
| Re-attach / preserve the MTP head | — | — | **the whole point.** See below |

Peak disk ≈ **660 GB**. Peak memory at every step is one decoder layer, **13.81 GiB at
BF16** — memory is never the constraint. Wall-clock is not estimated here: nothing in this
pipeline has been run, and this repo has a standing rule against extrapolating from
arithmetic.

**The riskiest step is not the merge — it is keeping the MTP head, and it is now a
*double* silent failure.** `transformers` drops it on load by **hardcoded layer index**
(`_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", …]`), which is
[#48](https://github.com/kreuzhofer/dgx-manager/issues/48)'s trap again; and
`llm-compressor`'s own MTP rescue is gated on config fields GLM does not have, so it never
fires and says nothing. Both the merge *and* the quantise step have to be MTP-aware here,
where for Qwen3.8 only the quantise step did.

### Does either route preserve the MTP head?

The head is worth **1.57–1.82×** decode single-stream (*measured*), and more under batch
reasoning load. The two routes fail it in different ways, and neither fails it fatally:

| | route 1 (runtime LoRA) | route 2 (merge + requantize) |
|---|---|---|
| Head present in the served artefact? | **Yes** — untouched; the adapter is a separate object | **Only if you make it so.** `transformers` drops `layers.45.*` on load; `llm-compressor`'s copy-back never fires for GLM's config field; `compressed-tensors`' default `mtp_prefix="mtp"` matches zero GLM tensors. All three are silent |
| Head *adapted*? | **No, by design.** `vllm/v1/spec_decode/` has zero LoRA references; upstream's convention is `lora_skip_prefixes = ["mtp."]` | Yes — it is part of the merged weights, if it survives |
| Speed consequence | Draft comes from the unadapted base while verify uses base+adapter, so **acceptance falls by an unmeasured amount**. Output distribution is still correct (rejection sampling) | None, if preserved |
| Precision consequence | none | MTP lands at **FP8-block, not NVFP4** — the documented microscale fallback, and what every shipped checkpoint does |

Route 1 keeps the head and loses some of its speedup. Route 2 keeps the full speedup and
risks losing the head entirely, silently, three different ways. Route 2's risk is the
manageable one *because it is checkable*: count `layers.45.*` tensors in the output index.

### The cheapest thing that changes this answer

Route 1 is **~30 lines in one upstream file**, and every hard part is already built. If
GLM-5.3-Flash matters, the highest-leverage move on this whole map is a PR to
`vllm/models/glm5next/nvidia/model.py` adding `SupportsLoRA`, a `packed_modules_mapping`
that names the real MLA fusions, and a `lora_skip_prefixes` for the positionally-named MTP
layer. That is a contribution, not a workaround, and it deletes route 2 entirely.

**Do not attempt it against our current container.** See
[The image we run is unpinnable](#appendix-a--the-image-we-run-is-unpinnable).

---

## Route 1 — runtime LoRA, in detail

### 1. Does vLLM support LoRA on a quantized base, and on NVFP4 specifically?

**Yes, and the mechanism is scheme-independent.** *(read)*

vLLM does not merge LoRA into quantized weights. It calls the base layer's own quantized
GEMM and adds an unquantised low-rank delta to the output:

```python
# vllm/lora/layers/base_linear.py:204
def _apply_sync(self, x, bias=None):
    output = self._get_quant_method().apply(self.base_layer, x, bias)
    return self._apply_lora_to_output(x, output)
```

Because the base GEMM is dispatched through `quant_method`, *any* quantization scheme that
has a working linear kernel also has working LoRA. There is no per-scheme allowlist for
linear layers, and no check in `LoRAConfig.verify_with_model_config` that inspects
quantization at all.

Compressed-tensors is explicitly anticipated. The layer's `weight` property has a branch
per storage format:

```python
# vllm/lora/layers/base_linear.py:306
if hasattr(self.base_layer, "weight"):       # unquantized
elif hasattr(self.base_layer, "weight_packed"):  # Compressed Tensor
elif hasattr(self.base_layer, "qweight"):        # GPTQ/AWQ
elif hasattr(self.base_layer, "B"):              # marlin
```

`weight_packed` is exactly the tensor name an NVFP4 compressed-tensors checkpoint uses.

There is also a dedicated FP8 LoRA kernel family (`vllm/lora/ops/triton_ops/
lora_{shrink,expand}_fp8_op.py`, `fused_moe_lora_fp8_op.py`), which matters because the
GLM-5.3-Flash *base* checkpoint is FP8, not BF16 (see [Route 2](#route-2--merge-then-re-quantise)).

**Caveat, and it is a real one:** the adapter is trained against *some* base. Serving a
BF16-trained adapter on top of NVFP4 weights is serving it against a base it never saw.
That is standard practice and usually fine, but it is a quality question this ticket does
not answer and the map has already parked as *"quality risk of tuning over a quantized
base"*.

### 2. Does vLLM support LoRA on MoE expert weights?

**Yes. The "attention only on MoE" limitation the ticket remembers is gone.** *(read)*

`vllm/lora/layers/fused_moe.py` defines `FusedMoEWithLoRA` (per-expert, "2D"/megatron
layout) and `FusedMoE3DWithLoRA` (stacked, "3D"/PEFT layout), both registered in
`_all_lora_classes`. `get_supported_lora_modules` scans the live model for
`(LinearBase, MoERunner)` instances with the comment *"In vLLM, all linear layers support
LoRA"*, so the MoE runner is discovered automatically.

The expert kernel has to opt in:

```python
# vllm/model_executor/layers/fused_moe/modular_kernel.py:594
elif moe_config.is_lora_enabled and not cls.supports_lora():
    return False, _make_reason("LoRA")
```

`supports_lora()` defaults to `False` and is flipped to `True` by mixing in
`LoRAExpertsMixin`. Three kernels do so today: `TritonExperts`, `UnfusedOAITritonExperts`,
`_TrtLlmLoRAExpertsBase` — and `MarlinExperts`.

GLM-5.3-Flash's MoE is built with vLLM's own `FusedMoEFactory` (`model.py:225`), so it is
an ordinary `MoERunner` that `FusedMoEWithLoRA` can wrap. Nothing model-specific stands in
the way here.

### 3. Does it work with `--moe-backend marlin`?

**Yes — and marlin is one of the few backends where it *does* work.** *(read)*

```python
# vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:694
class MarlinExperts(LoRAExpertsMixin, MarlinExpertsBase):
```

and the LoRA path is genuinely implemented, not inherited-and-unused — `apply()` consumes
`self._lora_context` at line 757 and threads `apply_w13_lora` / `apply_w2_lora` through
`activation_with_lora` and `moe_sum_with_lora` callbacks.

The NVFP4 oracle maps our flag to that class:

```python
# vllm/model_executor/layers/fused_moe/oracle/nvfp4.py:127
elif backend == NvFp4MoeBackend.MARLIN:
    from ...experts.marlin_moe import MarlinExperts
    return [MarlinExperts]
```

`MarlinExpertsBase.__init__` asserts on `use_nvfp4_w4a16` among its accepted schemes, so
this is the NVFP4 path, not a coincidence of naming. And our deploy log already prints
`Using 'MARLIN' NvFp4 MoE backend`, so this is the code we are on.

Two useful corollaries:

- **The corrupting backend becomes unreachable.** The generic gate at
  `modular_kernel.py:594` removes any non-LoRA kernel from candidacy when LoRA is on.
  `Nvfp4QuantizationEmulationTritonExperts` returns `supports_lora() -> False` and
  self-rejects with `"kernel does not support LoRA"`. So `--enable-lora` narrows the
  auto-select set in exactly the direction our recipe already forces by hand.
- **`--enforce-eager` is not a problem.** LoRA's cudagraph specialisation
  (`specialize_active_lora`) is an optimisation, not a requirement, and we run eager
  anyway.

### 4. Does it survive alongside `--speculative-config` with MTP?

**Nothing blocks the combination. But the drafter is not adapted, and that costs
acceptance.** *(read + derived)*

There is exactly one LoRA/spec-decode incompatibility in the tree, and it is not ours:

```python
# vllm/config/vllm.py:2651  (_validate_adaptive_verification)
if self.lora_config is not None:
    raise ValueError("Adaptive verification is not currently compatible with LoRA")
```

`vllm/v1/spec_decode/` contains **zero** occurrences of the string `lora`. The proposer
loads its own module through `get_model(...)`; LoRA wrapping happens only via
`LoRAModelRunnerMixin.load_lora_model(model)` on the *target*. For GLM-5.3-Flash the MTP
head is a separately registered model —

```python
# registry.py (PR #53906)
"Glm5NextMTPModel": ("vllm.models.glm5next", "Glm5NextMTP"),
```

`class Glm5NextMTP(nn.Module, DeepseekV2MixtureOfExperts)` — no `SupportsLoRA` either.

So under runtime LoRA the draft comes from the unadapted base and the verify comes from
base+adapter. **Correctness is safe** — rejection sampling makes the emitted distribution
the target's — but every token where the adapter has moved the distribution is a token the
drafter is now more likely to get wrong. Our measured acceptance is 45.4% single-stream /
71.7% under batch reasoning load, giving 1.57–1.82× decode. How much of that survives an
adapter is **unmeasured and unmeasurable without running it**; the direction is certain,
the magnitude is not.

Upstream has clearly decided this is the intended semantics rather than a gap:

```python
# vllm/models/deepseek_v4/nvidia/model.py:1731
# The MTP draft head is not LoRA-adapted.
lora_skip_prefixes = ["mtp."]
```

**A trap for whoever writes the glm5next patch:** copying that line verbatim would skip
nothing. GLM names its MTP head *positionally* — layer 45, one beyond
`num_hidden_layers: 45` — and the recipe already records that grepping tensor names for
`mtp`/`nextn`/`draft` returns zero hits. The skip prefix has to be written against the
real name, or an adapter that happens to carry layer-45 tensors will be loaded into the
target model.

### 5. What it costs in KV pool and throughput

**This is where route 1 stops being free even after the code lands.** *(derived, from
`zai-org/GLM-5.3-Flash` `config.json` and vLLM's LoRA weight shapes)*

vLLM allocates the stacked LoRA buffers for all `max_loras` slots at **model-load time**
(`load_lora_model` → `create_lora_manager` → each layer's `create_lora_weights`), before
the KV pool is sized — and both `profile_run` and cudagraph capture then run inside
`maybe_setup_dummy_loras(self.lora_config)`. So **LoRA memory is inside the profiled
budget and comes straight out of the KV pool.** On this model that is not a rounding
error.

Shapes, from `FusedMoEWithLoRA._create_lora_{a,b}_weights` and the `_slice_*` helpers
(`fully_sharded_loras` defaults to `False`, so `lora_A` for w13 is **not** sharded by TP):

| tensor | shape per expert | sharded at TP2? |
|---|---|---|
| w13 `lora_A` × 2 slices | `(r, 4096)` | no |
| w13 `lora_B` × 2 slices | `(2048/tp, r)` | yes |
| w2 `lora_A` | `(r, 2048/tp)` | yes |
| w2 `lora_B` | `(4096, r)` | no |

Over 288 routed experts × 42 sparse-MoE layers (`first_k_dense_replace: 3`):

| rank | all-expert MoE LoRA, per rank @ TP2 | sparse-MLA attention only | ratio |
|---:|---:|---:|---:|
| 8 | **2.77 GiB** | 7.4 MiB | 384× |
| 16 | **5.54 GiB** | 14.8 MiB | 384× |
| 32 | **11.07 GiB** | 29.6 MiB | 383× |

Set that against the pool we actually have *(measured, gmu 0.87, TP2, at ~148,131 KV
tokens/GiB)*:

- **without** MTP: 1,015,403 tokens = **6.85 GiB**
- **with** MTP: 554,535 tokens = **3.74 GiB**

So at rank 16 a single all-expert adapter is **1.5× the entire KV pool** we currently run
with MTP on, and 81% of the pool with MTP off. Even rank 8 takes 74% of the MTP-on pool.
`max_lora_rank` is a `Literal[1, 8, 16, 32, ...]`, so there is no rank-4 escape hatch.

Three ways out, in order of how much they cost elsewhere:

1. **Do not adapt every expert.** Attention-only LoRA is **384× smaller** and disappears
   into rounding. The map already lists *expert-subset tuning* as a live method; this is
   an independent argument for it.
2. **`--enable-moe-shared-loras`.** Shares w13 `lora_A` and w2 `lora_B` across experts
   (stored once, expert-dim 1, broadcast at kernel time). *Derived:* 1.12 GiB at rank 16,
   0.56 GiB at rank 8 — a ~5× reduction. But the **adapter must have been trained in that
   layout**; this is not a serving-side switch you can flip over an ordinary PEFT export.
3. **Give back the MTP head.** Dropping `--speculative-config` restores the pool from 3.74
   to 6.85 GiB — and costs the 1.57–1.82× decode. Trading the speedup for the ability to
   serve the adapter is a real trade, and it is the same trade as (1) from the other end.

**Throughput:** unquantified, deliberately. Mechanically, MoE LoRA replaces the fused
Marlin activation with `activation_with_lora`/`moe_sum_with_lora` callbacks that run a
Triton shrink and expand per expert group per layer — 42 layers × 2 extra kernel families.
`VLLM_LORA_ENABLE_DUAL_STREAM` exists to overlap them onto a second CUDA stream. Nobody
should quote a number for this without running it, least of all on GB10.

### 6. Would `--enable-lora` at least *start* today?

**Probably yes, and that is the problem.** *(read, with one inference flagged)*

Walking the path with the inherited `SupportsLoRA`:

1. `load_lora_model` checks `supports_lora(model)` — passes, by inheritance from
   `Glm4vForConditionalGeneration`.
2. `get_supported_lora_modules` scans live modules — finds `fused_qkv_a_proj`, `q_b_proj`,
   `kv_b_proj`, `o_proj`, the KDA projections, and `experts`.
3. `process_packed_modules_mapping` returns GLM-4V's mapping *verbatim* —
   `get_packed_modules_mapping` short-circuits with *"don't infer mapping if the model has
   defined it explicitly"* — then appends an `experts` entry derived from
   `get_moe_expert_mapping(model)`.
4. `_load_adapter` builds `expected_lora_modules` from (2) expanded by (3), and
   `LoRAModel.from_local_checkpoint` raises on anything outside it:
   `"While loading {dir}, expected target modules in {...} but received [...]"`.

**Two predicted failure modes, and the second one is the dangerous one.**

*Loud:* an adapter naming the MLA down-projections by their HF names (`q_a_proj`,
`kv_a_proj_with_mqa`) is rejected at *adapter load time*, not at startup, because
`fused_qkv_a_proj` has no entry in the inherited mapping to expand into those names.

*Silent:* the model's two **fused** input projections are never LoRA-wrapped at all.
`_create_lora_modules` looks each module's own suffix up in the mapping —

```python
# vllm/lora/model_manager.py:474
packed_moduled_lst = self.packed_modules_mapping.get(parts, [])
```

— and gets `[]`, because GLM-4V's mapping knows only `qkv_proj` and `gate_up_proj`. Both
layers are `MergedColumnParallelLinear` **subclasses**:

- `fused_qkv_a_proj` is `DeepSeekV2FusedQkvAProjLinear(MergedColumnParallelLinear)` — the
  Q/KV down-projection on all **11 sparse-MLA layers**
- `in_proj_qkvbfg_a` is `_Glm5NextMergedColumnParallelLinear(MergedColumnParallelLinear)`
  — the fused input projection on all **34 KDA linear-attention layers**

and every candidate wrapper for that family requires a non-empty packed list
(`MergedColumnParallelLinearWithLoRA` needs `len == 1`, the sharded variant `len == 2`),
while `ColumnParallelLinearWithLoRA` matches on `type(source_layer) is
ColumnParallelLinear` — an *exact* type check a subclass fails. So `from_layer` returns
the layer untouched. **No error, no warning: LoRA is simply absent from 45 of 45 layers'
input projections.**

A third hazard from the same direction: `q_conv1d`, `k_conv1d`, `v_conv1d` in the KDA
block are declared as plain `ColumnParallelLinear` (weight containers for the short
convolution). They are exact-type matches, so they *are* wrapped, and a PEFT config using
`target_modules="all-linear"` would happily try to adapt convolution weights.
`--target-modules` exists as a serving-side filter and should be used.

*This whole subsection is inference from reading the load path, not something I ran* — the
model is not deployed and `dgx-spark-02`/`-03` are powered down. It is cheap to falsify
once they are up.

One more: `is_3d_moe_weight` defaults to `False` on `SupportsLoRA` and neither GLM-4V nor
glm5next overrides it, while PEFT exports MoE adapters in the **3D** layout. Serving a
PEFT MoE adapter therefore additionally needs `--enable-mixed-moe-lora-format` plus
`is_3d_lora_weight: true` on each `--lora-modules` entry.

---

## Route 2 — merge, then re-quantise

### First: the map's premise needs one correction

`zai-org/GLM-5.3-Flash` — the repo everyone means when they say "the base" — **is FP8, not
BF16.** Its `config.json` carries a `quantization_config` (`fp8`, `e4m3`, block
`[128,128]`, dynamic activations) and the Hub tags it `fp8`. The BF16 weights are a
*separate* repo. *(read)*

| repo | shards | bytes | GB | GiB |
|---|---:|---:|---:|---:|
| `zai-org/GLM-5.3-Flash-BF16` | 120 | 642,652,070,880 | 642.65 | **598.52** |
| `zai-org/GLM-5.3-Flash` (FP8) | 62 | 328,337,455,672 | 328.34 | 305.79 |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` *(what we serve)* | 121 | 194,665,046,744 | 194.67 | 181.30 |
| `RedHatAI/GLM-5.3-Flash-NVFP4` | 11 | 197,843,812,476 | 197.84 | 184.26 |

The map's "~640 GB BF16" is right, but only for `-BF16`. A merge must target that repo;
merging into the FP8 release would be a merge into a quantized base, which is the
ill-defined case the ticket asks about separately.

### Can we hold it? Yes — on exactly one machine, with no margin

*(measured 2026-08-31, `df` and `free` over SSH)*

| resource | size | free |
|---|---:|---:|
| **`dgx-spark-04` local NVMe `/`** | 917 G | **714 G** |
| `agenthost` local disk `/` | 937 G | 504 G |
| `/mnt/tank` (NFS, the only *shared* store) | 3.6 T | **340 G (91% used)** |
| `dgx-spark-01` local NVMe `/` | 916 G | 80 G |
| `agenthost` RAM | 30 GiB | 22 GiB available |
| `dgx-spark-01` / `-04` unified memory | 121 GiB each | ~3 GiB (both serving) |

The naive budget — keep everything — is **BF16 base 642.65 GB + merged BF16 642.65 GB +
NVFP4 197.84 GB = 1.48 TB**, which fits nowhere we own. But neither copy has to persist:

- an in-place shard merge (§*How the merge would actually run*) overwrites the BF16 shards,
  removing the second 642.65 GB;
- `model_free_ptq` runs one shard at a time, so each BF16 shard can be deleted as its
  NVFP4 output lands.

Peak then lands at **~643 GB + a few shards in flight ≈ 660 GB**, which fits
`dgx-spark-04`'s 714 GB and nothing else. `/mnt/tank` is out (340 GB free — it cannot even
hold the base), `agenthost` is out (504 GB), `dgx-spark-01` is out (80 GB).

**So the answer to "is there hardware anywhere we have that can hold a 640 GB model to
merge into" is yes, narrowly, and it is a disk question rather than a memory question.**
Memory is not the constraint at any step: `_no_split_modules` makes the largest indivisible
unit one `Glm5NextTextDecoderLayer` — **13.81 GiB at BF16** *(derived from the config,
cross-checked against the shard headers: 288 experts × (2048×4096 ×2 + 4096×2048) =
7,247,757,312 params, plus 31 non-expert tensors)* — and `model_free_ptq` budgets 3× shard
size per worker, ~16 GB against a mean BF16 shard of 5.36 GB. Both fit a Spark with room
over, and the per-layer figure even fits the RTX 5090.

Two caveats that are easy to miss:

- **`dgx-spark-04` is currently serving**, with ~3 GiB of its 121 GiB free. This is a
  scheduling conflict, not a capacity one.
- **660 GB of 714 GB is a 92% fill on the machine's root filesystem.** That is not somewhere
  to run an unattended multi-hour job. Buying a disk is the honest answer if this becomes
  real.

### The toolchain: possible, but not with anything released

*(read, from `vllm-project/llm-compressor` @ `b7a014f` and the `transformers` tags)*

Three pins have to line up, and today they do not:

1. **`glm5_next` first appears in `transformers` v5.16.1** (2026-08-26). It is native to
   the library — the Hub repo ships **no** remote code (`modeling_glm5_next.py` 404s), so
   there is no `trust_remote_code` escape hatch.
2. **`llmcompressor` 0.13.0 (2026-08-11) pins `transformers<=5.14.1`** — six weeks before
   the architecture existed. The released version cannot load this model at all. Even
   llm-compressor HEAD in *release* mode pins `transformers==5.15.0`, which also lacks it.
   Only a **dev-mode source install** admits 5.16.1.
3. **`llmcompressor` has zero `glm5_next` awareness.** `grep -rin "glm5_next|Glm5Next"`
   over the repo returns nothing; its 21 GLM-5 hits are all GLM-5.2 (`glm_moe_dsa`), a
   different architecture. `conversion_mappings.py` has `glm4_moe`/`glm4_moe_lite` and no
   `glm5_next`.

None of that is fatal — a plain `QuantizationModifier` targeting expert `Linear`s needs no
per-architecture table, and three parties have shipped working compressed-tensors NVFP4
checkpoints of this exact model. `_no_split_modules` is read off the HF model, and
transformers already defines `["Glm5NextTextDecoderLayer", "Glm5NextVisionBlock"]`, so
sequential onloading works out of the box.

### MTP: the same trap as #48, one notch worse

[#48](https://github.com/kreuzhofer/dgx-manager/issues/48) found that anything routing
through `transformers` silently drops Qwen3.8's MTP head via
`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`. GLM does the same thing, **by hardcoded
layer index**: *(read)*

```python
# transformers/models/glm5_next/modeling_glm5_next.py:1359
_keys_to_ignore_on_load_unexpected = [r"layers\.45\.", r"layers\.\d+\.shared_head\."]
```

Present since the architecture's first commit. `num_hidden_layers = 45` builds layers
0–44, so index 45 is "unexpected" and discarded. `Glm5NextTextConfig` does not even define
`num_nextn_predict_layers` — it is a checkpoint-only key.

Our fleet's positional finding is confirmed against both real indexes: **zero** tensors in
either `zai-org/GLM-5.3-Flash` or `LibertAIDAI/GLM-5.3-Flash-NVFP4` contain the substring
`mtp` or `nextn`; the head is `model.language_model.layers.45.*` (1,760 tensors in the FP8
base, 3,481 in the NVFP4).

**And llmcompressor's own MTP rescue does not fire for GLM.** It gates on
`num_mtp_layers` / `mtp_num_hidden_layers` — GLM's field is `num_nextn_predict_layers`, so
the copy-back never runs — and even if it did, `compressed_tensors`' default
`mtp_prefix="mtp"` matches zero GLM tensors. **Both failures are silent.** That is #48's
exact failure mode with two extra layers of silence on top.

This is being fixed upstream: **llm-compressor PR #3118** (opened 2026-08-31, still open)
adds `num_nextn_predict_layers` detection, infers the prefix from the checkpoint index
including the GLM last-N-layers layout, and downloads the MTP shard on demand. It notes
that microscale MTP (NVFP4/MXFP4) **falls back to FP8-block** — which matches every shipped
checkpoint, including RedHat's, whose layer 45 lives in a shard literally named
`model_mtp.safetensors` at FP8 block `[128,128]`.

`model_free_ptq` remains the #48-style escape: it works on safetensors and never
constructs a transformers model, so layer 45 is simply visible. It supports NVFP4
*weight-only*, and its own GLM-5.2 example already carries `r"re:.*eh_proj.*"  # mtp
sensitive to quantization`. Two cautions: it **cannot** express RedHat's
`input_activations.dynamic: "local"` scheme (`validate.py:55` raises *"Model Free PTQ
cannot calibrate activations"*), and its `DEFAULT_FUSED_MAPPINGS` name MLA projections
`wq_a`/`wkv_a_with_mqa` where GLM uses `q_a_proj`/`kv_a_proj_with_mqa` — so the MLA fusion
rule silently does not match. Keep its targets to `mlp.experts.*` and that does not matter.

### Is LibertAI's recipe reproducible? No. RedHat's mostly is.

*(read)*

**`LibertAIDAI/GLM-5.3-Flash-NVFP4` is NVIDIA ModelOpt** (`quant_method: "modelopt"`), and
publishes **no recipe and no reproduction command** — the repo holds README, `config.json`,
chat template, generation config, licence and weights. Its `config.json` encodes the split
as one `Linear` group at 4 bits with `"input_activations": null` — i.e. **weight-only
NVFP4** — plus a 43-entry ignore list that names every attention and KDA projection
individually, so in effect only `layers.*.mlp.experts.*.{gate,up,down}_proj` become NVFP4
and everything else stays BF16. You could re-derive that from the config, but it is a
reconstruction, not a published recipe.

That `input_activations: null` is the useful part: **weight-only is exactly what
`model_free_ptq` can express**, and RedHat's `dynamic: "local"` is exactly what it cannot.
So the split we already serve is the one reachable by the MTP-preserving tool.

**`RedHatAI/GLM-5.3-Flash-NVFP4` does publish `recipe.yaml`** (compressed-tensors,
`llm-compressor`, `format: mixed-precision`) — a `QuantizationModifier` targeting
`re:.*mlp\.experts\..*(gate|up|down)_proj$` at NVFP4 `group_size 16`,
`strategy: tensor_group`, `dynamic: local` activations, ignoring `visual.*`, `lm_head`,
`mlp.gate` and `self_attn.indexer.*`. **It does not reproduce the shipped checkpoint:**
the recipe has one config group; the shipped `config.json` has two, the second targeting
`layers\.45\.mlp\.experts\..*` at FP8 block `[128,128]`. The layer-45 group was added
out-of-band at save time. So even the best-documented recipe on the Hub is incomplete in
exactly the place this ticket cares about.

Of ~100 GLM-5.3-Flash quantizations on the Hub, **two** publish a recipe: RedHat's and
`wtdcode/GLM-5.3-Flash-AWQ-W4A16` (which ignores `re:.*layers\.(45)\..*` and re-attaches
MTP by hand as `model-nextn-*.safetensors`, documenting it as *"NextN/MTP layer, dequantized
from the FP8 source to BF16"* — third-party corroboration that `oneshot` drops it).

### An unrelated finding that lands on our production deploy

vLLM issue [#54150](https://github.com/vllm-project/vllm/issues/54150) (open, 2026-08-28)
reports that **ModelOpt NVFP4 conversions of GLM-5.3-Flash emit invalid UTF-8 byte tokens
on SM120**, while a compressed-tensors NVFP4 of the same model is clean — 86 / 62 / 94
U+FFFD over six runs for three ModelOpt checkpoints (including **the exact
`LibertAIDAI/GLM-5.3-Flash-NVFP4` we serve**) versus **0** for RedHat's. Present at
`temperature=0`, at zero context, with MTP on or off, and **with
`moe_backend: marlin`** — the reporter's configuration is nearly ours.

A comment on 2026-08-30 identifies the cause in vLLM, not the weights: ModelOpt checkpoints
carry **separate global scales for w1 (gate) and w3 (up)**, and
`modelopt.py` picks one, mis-scaling the other shard's dequant *"up to 10× on GLM-5.3-Flash
NVFP4"*. vLLM already logs this — `"w1_weight_scale_2 must match w3_weight_scale_2.
Accuracy may be affected."` — as a `warning_once`.

**This is not part of #68's question and I have not verified it on our fleet** (`-02`/`-03`
are powered down). But it is cheap to check and it changes route 2's target: grep the next
GLM-5.3-Flash deploy log for `w13_weight_scale_2` / `w1_weight_scale_2`, and if it fires,
**re-quantise toward compressed-tensors, not toward LibertAI's ModelOpt output.** It also
puts an asterisk on our GPQA numbers, which are English-only and would not surface
multi-byte corruption.

### How the merge would actually run — and the trap in PEFT before you get there

*(read, `huggingface/peft` @ `15c0fed` (v0.20.1.dev0) and `transformers` v5.16.1)*

**PEFT has no `glm5_next` entry** — `grep -rni "glm5"` over the repo returns zero hits, and
`TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING` knows only the ancient
`"chatglm": ["query_key_value"]`. That is harmless: `_prepare_adapter_config`
(`lora/model.py:589`) raises a clean `ValueError("Please specify target_modules or
target_parameters")` rather than guessing. You just have to name them.

**The trap is that naming them the obvious way silently misses almost the whole model.**
Transformers stores the 288 routed experts as two fused 3-D `nn.Parameter`s:

```python
# transformers/models/glm5_next/modeling_glm5_next.py:106
class Glm5NextTextExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""
    self.gate_up_proj = nn.Parameter(torch.empty(num_experts, 2*intermediate, hidden))
    self.down_proj    = nn.Parameter(torch.empty(num_experts, hidden, intermediate))
```

The *checkpoint* has per-expert 2-D tensors
(`...layers.5.mlp.experts.{0..287}.{gate,up,down}_proj.weight`); transformers fuses them at
load. So `target_modules="all-linear"` — which collects only `nn.Linear`/`Conv1D`
(`tuners_utils.py:2390`) — would adapt attention, the 3 dense MLP layers, the shared expert
and the vision tower, and **silently skip ~98% of the parameters**. The first-party answer
is `LoraConfig.target_parameters` (`lora/config.py:922`), documented for exactly this case,
which here means `['mlp.experts.gate_up_proj', 'mlp.experts.down_proj']`. It carries real
restrictions: no `lora_dropout`, no DoRA, no `lora_bias`, all adapters must target the same
parameter set, and the docs themselves recommend merging because inference materialises the
LoRA contribution per expert.

*(This is trainable-surface territory rather than servability, but it is the same fact that
decides whether an expert-subset adapter can exist at all, so it belongs on the record.)*

**`merge_and_unload()` is in-place and layer-by-layer**, not a whole-model materialisation
(`tuners_utils.py:675`): for each key, `onload_layer(target)` → `target.merge()` →
`_replace_module`. `lora.Linear.merge` is `base_layer.weight.data += delta_weight`. Peak
memory is *whatever it took to hold the model*, plus one layer's delta.

PEFT does support merging an accelerate-offloaded model: `onload_layer`
(`tuners_utils.py:78`) pulls an offloaded module onto the device, merges, and writes it back
via `offload_state_dict(base_name + "-merged", …)`; `offload_dir`/`offload_folder` are
plumbed through `PeftModel.load_adapter`, and there is a CI test, `test_offload_merge`
(`tests/test_gpu_examples.py:2776`), that merges with `max_memory={0: "0.2GIB", "cpu":
"0.2GIB"}` and disk in the device map. **First-party and tested — on a tiny GPT-2. Nobody
has run it on a 320B `glm5_next` with `target_parameters`.** Rate it *supported, unproven
at this size*.

**There is no first-party safetensors-level streaming LoRA merge.** The one that exists is
third-party: `unsloth-zoo`'s `merge_and_overwrite_lora` (`saving_utils.py:2963`) mmaps and
overwrites one shard at a time, has a fused-MoE path that reconstructs per-expert slices
from a 3-D LoRA, and with `low_disk_space_usage=True` keeps peak disk at about one shard.
It is architecture-generic because it works on safetensors keys — but `glm5_next` appears
in unsloth only on the MLX (Apple Silicon) path, never on CUDA, so it is untested here.
**mergekit is the wrong tool**: `ModelReference.merged()` (`common.py:99`) just calls
`from_pretrained` + `PeftModel` + `merge_and_unload`; `--lazy-unpickle` and
`--low-cpu-memory` apply to the *later* merge-recipe graph, after the LoRA is already fully
materialised.

**And the same layer-45 trap applies here.** A `from_pretrained` → `merge_and_unload` →
`save_pretrained` round trip drops the MTP head, because transformers ignores
`layers\.45\.` on load. A shard-level merge preserves it for free. That is a second,
independent reason to prefer the streaming path over the PEFT-offload path — and it means
the *merge* step, not just the quantise step, has to be MTP-aware.

### Is "merge" even defined if the base was quantized during training?

**Into a 4-bit bnb base: defined, lossy, and PEFT warns about it. Into NVFP4 or
block-scaled FP8: not defined at all — there is no code.** *(read)*

The only true quantized merge PEFT implements is bitsandbytes 4-bit (`lora/bnb.py:355`),
and it is a full round trip:

```python
warnings.warn(
    "Merge lora module to 4-bit linear may get different generations due to rounding errors."
)
output = dequantize_bnb_weight(weight, state=weight.quant_state)
w_data = output + self.get_delta_weight(active_adapter)
... = bnb.nn.Params4bit(w_data.to("cpu"), **kwargs)
```

dequantize → add → requantize. Everything else refuses or dequantizes:

| backend | behaviour on merge |
|---|---|
| bnb 4-bit / 8-bit | dequant → merge → requant, with an explicit rounding warning |
| GPTQ | `ValueError("Cannot merge LORA layers when the model is gptq quantized")` |
| AWQ | no `merge` method at all → `BaseTunerLayer.merge` → `NotImplementedError` |
| HQQ | `layer.dequantize()` → merge → requantize |
| torchao | int8 only; otherwise `TypeError` / `NotImplementedError` |
| Intel INC | `NotImplementedError("Merging LoRA with INC layers is not yet implemented")` |
| **NVFP4 / compressed-tensors** | **no integration exists** |

`grep -rni "nvfp4"` and `grep -rni "compressed[-_]tensors"` over the whole PEFT repo both
return **zero hits**; the quantizer allow-list is `quant_methods = ["gptq", "aqlm", "awq"]`
(`lora/model.py:269`) plus the bnb/hqq/eetq/torchao/inc dispatchers. There is likewise no
block-scaled-FP8 LoRA layer, so **a LoRA cannot be merged into `zai-org/GLM-5.3-Flash`'s
FP8 weights either** — those are `w_fp8 × weight_scale_inv[128×128]` and PEFT has no code
that knows it. (PEFT *does* have fp8 awareness on the forward path — `UPCAST_DTYPES` and
`_LoraParameterProxy._low_prec_add`, *"addition in fp8 is not directly supported"* — but
`ParamWrapper.merge` does a bare `param.data += delta_weight` and does not use it.)

**Accepted practice, per PEFT's own docs and every published implementation: train the
adapter against the quantized base if you must, then merge into the full-precision base.**
PEFT's docs never bless merging into a quantized base — they say only *"if you used
quantization and merged the weights, small deviations are expected due to rounding errors"*
(`troubleshooting.md:160`) — and the LoftQ/PiSSA guidance is framed entirely as reducing the
error between the quantized base you train on and the full-precision base you merge into.
Unsloth encodes the same rule in code, refusing a 16-bit merge from an nf4/fp4 base and
preferring an existing 16-bit sibling repo over dequantizing.

**For us that resolves cleanly:** the question "is merge well-defined over a QLoRA-on-NVFP4
base" does not need answering, because the full-precision sibling is published.
`zai-org/GLM-5.3-Flash-BF16` is the merge target, whatever the adapter was trained against.
What *would* still be live is the quality question — an adapter trained against NVFP4 and
merged into BF16 is being applied to weights it never saw — and the map already holds that
one open.

---

## Appendix A — the image we run is unpinnable

Every version claim above is about **vLLM**. None of them is automatically a claim about
the container we serve from, and the gap is wider than it looks. *(read, from the Docker
Hub registry API)*

`vllm/vllm-openai:glm53-flash-arm64-cu130`, the day-0 base our overlay builds on:

```
created:  2026-08-26T02:42:34Z
labels:   ai.vllm.build.commit  = "unknown"
          ai.vllm.build.pipeline = "local"
          ai.vllm.image.tag      = "local/vllm-openai:dev"
          org.opencontainers.image.revision = "unknown"
```

It is a **local dev build with no commit hash**, published under the vLLM org. And it was
built **nine hours before PR #53906's first commit was pushed** (`933876c`,
2026-08-26T11:37:33Z), so it is not even a snapshot of a commit that exists in the PR's
history. There is no metadata path from the image to a vLLM revision.

What this does and does not undermine:

- **Does not undermine the LoRA-infrastructure findings.** Every mechanism cited in
  [Route 1](#route-1--runtime-lora-in-detail) — `FusedMoEWithLoRA`, `LoRAExpertsMixin`,
  `MarlinExperts(LoRAExpertsMixin, …)`, `lora_skip_prefixes`, `is_3d_moe_weight`,
  `enable_mixed_moe_lora_format` — is present unchanged at **v0.25.0** (2026-07-11) and
  every tag since, verified by fetching the files at each tag. Any plausible snapshot of
  main from August 2026 has all of it.
- **Does not undermine the blocker.** `Glm5NextForCausalLM`'s class declaration is
  byte-identical at the PR's first commit and at its head six days later.
- **Does undermine any attempt to patch this in place.** You cannot diff our container
  against a known tree, so a source patch against it is unverifiable in the way the
  `glm53-flash-overlay` layers are — those work only because each asserts on the exact
  source text it expects. Route 1 should be pursued **upstream in the PR**, against a
  tree you can name, not as a ninth overlay layer.

`FLASHINFER_VERSION=0.6.17` and `NCCL_VERSION=2.30.7` in the image build args also confirm
why overlay layers v3 and v4 exist, which is a useful independent check that the vendored
overlay is pinned against the image we think it is.

---

## Appendix B — confidence

| Claim | Status |
|---|---|
| `Glm5NextForCausalLM` lacks `SupportsLoRA`; the multimodal wrapper inherits it from GLM-4V | **Read**, at PR #53906 head and at `933876c`. The single load-bearing fact of this document |
| Marlin NVFP4 MoE supports LoRA; the machinery is present at v0.25.0–v0.28.0 and `main` | **Read**, files fetched at each tag |
| Only `enable_adaptive_verification` conflicts with LoRA; `vllm/v1/spec_decode/` has no LoRA plumbing | **Read** (a grep result, i.e. an absence — weaker than a positive quote, but a total absence across 17 files) |
| Repo sizes, shard counts, `config.json` fields, `transformers` line 1359, `llmcompressor` pins, `recipe.yaml` existence | **Read** — HF/PyPI/GitHub APIs and raw files, 2026-08-31 |
| Disk free, RAM, per-machine capacity | **Measured** — `df`/`free` over SSH, 2026-08-31 |
| KV-pool figures, 148,131 tok/GiB, 1.57–1.82× MTP, GPQA | **Measured** on this fleet, quoted from the recipe's own notes |
| **LoRA adapter footprint table (2.77 / 5.54 / 11.07 GiB)** | **Derived** — vLLM's tensor shapes × the config. Arithmetic only; never allocated on hardware. The shapes are read, the multiplication is mine |
| The two fused projections are silently left unwrapped; a `q_a_proj`-named adapter is rejected at load | **Derived** from reading `from_layer` / `can_replace_layer` / `_create_lora_modules`. **Not executed** — `-02`/`-03` are down. Falsifiable in one deploy |
| Throughput cost of MoE LoRA | **Unknown, deliberately.** Mechanism only |
| Wall-clock of a merge or a re-quantise pass | **Unknown.** Nothing was run. Do not extrapolate |
| PEFT offload-merge works at 320B | **Unproven.** First-party and CI-tested on a tiny GPT-2; never at this scale |
| ModelOpt UTF-8 corruption applies to *our* deployment | **Unverified.** Reported for the same checkpoint, same MoE backend, same GPU family — but not checked here |

Two of these are cheap to convert. Powering up `-02`/`-03` and running one deploy with
`--enable-lora` settles the whole "would it start" question, and grepping that same boot
log for `w1_weight_scale_2` settles the corruption question.

---

## Sources

**vLLM**

- Source at `main` @ `85c1365` (2026-08-31) and at tags `v0.25.0`, `v0.26.0`, `v0.27.0`, `v0.27.1`, `v0.28.0` — files cited inline
- PR [#53906](https://github.com/vllm-project/vllm/pull/53906) *"[Model] add GLM-5.3-Flash support"* — open, branch `ZJY0516:glm-release`, first commit `933876c`
- Issue [#54150](https://github.com/vllm-project/vllm/issues/54150) — ModelOpt NVFP4 UTF-8 corruption on SM120, and its 2026-08-30 comment identifying the w13 global-scale bug
- `docs/features/lora.md` in the vLLM tree — **the only claim taken from docs rather than source**: the 2D (megatron) vs 3D (PEFT) MoE adapter layouts
- Docker Hub registry API — image manifest and config blob for `vllm/vllm-openai:glm53-flash-arm64-cu130`

**Quantization and PEFT toolchains**

- `vllm-project/llm-compressor` @ `b7a014f` — `entrypoints/model_free/`, `modeling/moe/conversion_mappings.py`, `transformers/compression/compressed_tensors_utils.py`; PR [#3118](https://github.com/vllm-project/llm-compressor/pull/3118) (open); PyPI metadata for 0.13.0
- `huggingface/peft` @ `15c0fed` (v0.20.1.dev0) — `tuners_utils.py`, `tuners/lora/{model,layer,bnb,awq,hqq,torchao,inc}.py`, `utils/constants.py`, `tests/test_gpu_examples.py::test_offload_merge`, `docs/source/package_reference/lora.md`
- `huggingface/transformers` v5.16.1 — `models/glm5_next/modeling_glm5_next.py`, `core_model_loading.py`; tag probing showing `glm5_next` absent at v5.14.1–v5.16.0
- `arcee-ai/mergekit` @ `a6e4028` — `common.py`, `architecture/` (no `Glm5Next` support)
- `unslothai/unsloth-zoo` — `saving_utils.py` `merge_and_overwrite_lora`

**Checkpoints (HuggingFace API + raw files + safetensors headers)**

- `zai-org/GLM-5.3-Flash` (FP8), `zai-org/GLM-5.3-Flash-BF16`, `LibertAIDAI/GLM-5.3-Flash-NVFP4`, `RedHatAI/GLM-5.3-Flash-NVFP4` (incl. its `recipe.yaml`), `wtdcode/GLM-5.3-Flash-AWQ-W4A16`

**This repo and this fleet**

- `recipes/dgxrun/glm-5.3-flash-libertai-nvfp4-2x.yaml`, `scripts/build-glm53-flash-image.sh`, `scripts/glm53-flash-overlay/README.md` (branch `qwen3.8-27b-and-staging-design`) — every measured serving figure
- [#48](https://github.com/kreuzhofer/dgx-manager/issues/48) write-up `docs/research/lora-merge-to-nvfp4-path.md` (branch `research/nvfp4-merge-path`) — the `model_free_ptq` / MTP-preservation precedent
- `df` / `free` over SSH on `agenthost`, `dgx-spark-01`, `dgx-spark-04`, and `GET /api/nodes` on the manager — 2026-08-31
