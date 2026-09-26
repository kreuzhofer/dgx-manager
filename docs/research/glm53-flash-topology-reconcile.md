# One 8×H200 node, or nine H100 nodes? Reconciling #67 and #69

> Research for [#71](https://github.com/kreuzhofer/dgx-manager/issues/71). Part of the
> [GLM-5.3-Flash wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/65).
>
> **Primary sources only.** Everything below was read out of `NVIDIA-NeMo/Automodel` `main` at
> commit **`1358302c`** (2026-09-25 22:53 −0700), a full clone, plus the HuggingFace model API
> and the checkpoint's own `config.json`. Read on **2026-09-26**. No blog, no write-up, no
> secondary source. Line numbers are against `1358302c`; the GLM files have not changed since
> `cdc9147a` (2026-08-29), so they are also valid against `9228f33c`+.
>
> Every number carries a label:
> **measured** = read out of a vendor's published result or an API;
> **derived** = arithmetic over measured values, with the arithmetic shown;
> **extrapolated** = a model with an assumption in it, stated.
>
> Where I carry a fact forward from #66/#67/#69 rather than re-deriving it, I say so.

---

## VERDICT

**One node. `gpu-h200-sxm` / `8gpu-128vcpu-1600gb`, 8×H200, `eu-north1`, self-service — #69 was
right, and #67's nine nodes were never about this job.**

The two tickets never disagreed. They costed different workloads:

- **#67 reported NeMo's validated topology for a *full-parameter SFT* of the language backbone.**
  At EP72/9 nodes the 57.68 GiB/GPU peak is dominated by gradients and AdamW state over 313 B
  trainable parameters. That is the recipe NVIDIA published, and it is the right size **for that
  job**.
- **#69 costed a *frozen-base* PEFT run.** With the base frozen there are no gradients and no
  optimizer state for it — only ~112 M adapter parameters — so the whole budget collapses to
  "weights + activations".

NVIDIA says this themselves, in a shipped recipe for a structurally identical model
(`docs/model-coverage/omni/nvidia/nemotron-3-5-super-vl.mdx` L52-55, **measured**):

> "Because the base weights are frozen, there are no fp32 master weights or optimizer states for
> them, so the 121B model trains on a single node of 8 H100 (`cp_size 1`, `ep_size 8`) **instead
> of the 4 nodes that full SFT needs**."

That is exactly this ticket's question, answered by the framework's own maintainers, for a
121 B / 512-expert MoE **VLM** with a frozen vision tower, at **EP8 on one node**, at a
**measured 38.8 GiB/GPU peak** (same file, L61).

### The arithmetic, in one line

| | value | label |
|---|---:|---|
| Routed-expert params loaded (42 sparse layers × 288 × 25,165,824) | **304,405,807,104** | **derived** (exact shape arithmetic from `config.json`) |
| Everything else (45 layers, MTP dropped, incl. 0.56 B vision) | **≈ 9.45 B** | **derived**, bounded 9.42–9.52 B by the checkpoint total |
| Model NeMo actually loads (`num_nextn_predict_layers: 0`) | **≈ 313.85 B** → **584.6 GiB** BF16 | **derived** |
| Frozen weights resident **per GPU at EP8 on 8 GPUs** | **73.1 GiB** | **derived** |
| H200 usable (141 GB nameplate ≈ 140.4 GiB) | 140.4 GiB | **measured** (#69) |
| **Steady-state headroom on 8×H200** | **≈ 66 GiB/GPU** | **derived** |
| H100 usable (80 GB ≈ 79.6 GiB) → headroom | **≈ 6.5 GiB/GPU** | **derived — fatal** |

### The shape to request in [#70](https://github.com/kreuzhofer/dgx-manager/issues/70)

| | |
|---|---|
| Platform / preset | **`gpu-h200-sxm` / `8gpu-128vcpu-1600gb`**, region **`eu-north1`** *(carried from #69; not re-verified here)* |
| GPU cluster (InfiniBand) | **not needed** — single node, NVLink is intra-node |
| Shared filesystem | **≥ 1 TiB**, provisioned for read bandwidth. **Stage `zai-org/GLM-5.3-Flash` (FP8, 305.8 GiB / 328.3 GB), NOT the BF16 sibling** — see §2.3. This halves #69's 642.7 GB data-movement budget. |
| Topology | `tp_size: 1`, `pp_size: 1`, `ep_size: 8`, `cp_size: 1` at ≤ 8k; **`cp_size: 2` at 16k, `cp_size: 8` at 32k** — see §2.4 |
| Escape hatch if the probe OOMs | **2 nodes** (16 GPUs), `ep_size: 16` → 36.5 GiB/GPU of weights. Works on **H100** too. Self-service via GPU cluster + Managed Kubernetes (#69 §3). |

**8×H100 is out at 8 GPUs and this is not marginal** — 73.1 GiB of frozen weights against 79.6 GiB
leaves 6.5 GiB, and the one measured comparable job needed ~10.6 GiB of non-weight memory at a
*quarter* the sequence length. #69 called 8×H100 "a shape that can hold the model and cannot train
it"; that verdict survives contact with the framework source, for a sharper reason than #69 had.

### The one thing that could still break it

The FP8→BF16 dequantize on load allocates **rank-local FP8 destination storage alongside** the
model's BF16 weights (`nemo_automodel/components/moe/state_dict_mixin.py` L1058-1060, verbatim:
*"Quantization casts each split with `value.to(float8_e4m3fn)`, creating storage separate from the
model's grouped weights"*). At EP8 that is **73.1 + 36.5 = ~110 GiB/GPU at load peak** (**derived**)
against 140.4 GiB — 78% full, before a single activation. At EP72 the same transient is 12.2 GiB, so
NVIDIA has never exercised this path anywhere near the pressure EP8 puts on it. **This is the
single most likely place a one-node run fails, and it fails at load, not at step 1.** It is the
first thing [#72](https://github.com/kreuzhofer/dgx-manager/issues/72) should watch.

---

## §1 — Is EP8 a supported configuration?

**Yes. Not by inference from a filename — by an explicit sentence in the coverage doc, by the
model's own capability declaration, by the only two constraints in the EP plumbing, and by five
shipped single-node EP8 recipes including one VLM+PEFT+`hybridep` recipe.**

### 1.1 The coverage doc says so in as many words

`docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx` **L53-55**, verbatim:

> "An EP size must evenly divide the model's 288 routed experts and fit the available GPU
> memory; **EP72 is the published validated topology, not a model requirement.**"

That clause was added in commit **`cdc9147a`** (*"docs(models): expand GLM-5.3-Flash coverage"*,
PR [#3744](https://github.com/NVIDIA-NeMo/Automodel/pull/3744), 2026-08-29) — **one day after
`9228f33c`**, the commit #67 read. #67's "validated topologies are EP72/EP144" is a correct reading
of what existed on 2026-08-31; the framework has since said explicitly that it is not a constraint.

Two constraints are stated, both satisfied: **288 / 8 = 36** ✓, and "fit the available GPU memory" —
which is §2.

Three restrictions in the same doc are real and do **not** bite a single 8-GPU node (L168-171):

> "- TP and PP are not supported for this model. Packed contiguous CP and EP are supported with
> TP1 and PP1.
> - Full-model **single-GPU** checkpoint loading and training are not supported. Use the
> distributed checkpoint initialization path."

"Single-GPU" means `world_size == 1`. It is enforced structurally: at `world_size > 1` a custom
model takes the rank-local DCP path, and only at `world_size == 1` does
`safetensors_requires_full_cpu` become true
(`nemo_automodel/components/checkpoint/checkpointing.py` **L1065-1069**:
`… and (not is_custom_model or world_size == 1)`). An 8-GPU node is `world_size == 8`. Not affected.

### 1.2 The model declares EP support with no size attached

`nemo_automodel/components/models/glm5_next/model.py` **L223-231**:

```python
@dataclass(frozen=True)
class ModelCapabilities:
    """Parallel axes intentionally supported for the released checkpoint."""
    supports_tp: bool = False
    supports_cp: bool = True
    supports_pp: bool = False
    supports_ep: bool = True
    supports_thd: bool = True
```

`grep -n "ep_size\|ep_mesh\|expert_parallel" nemo_automodel/components/models/glm5_next/*.py`
returns **zero hits**. Expert parallelism for this model is entirely handled by the generic MoE
parallelizer; there is nothing model-specific about 72.

### 1.3 The only two EP constraints in the whole plumbing

**(a) The mesh must divide.** `nemo_automodel/components/distributed/mesh_utils.py` **L316-319**:

```python
non_pp_size = dp_size * cp_size * tp_size
if non_pp_size % ep_size != 0:
    raise ValueError(f"{non_pp_size=} must be a multiple of {ep_size=}")
ep_shard_size = non_pp_size // ep_size if ep_size < non_pp_size else 1
```

At 8 GPUs, tp1/pp1/cp1: `dp_size = 8`, `non_pp_size = 8`, `8 % 8 == 0` ✓, **`ep_shard_size = 1`**.

**(b) The experts must divide.** Twice, identically:
`nemo_automodel/components/moe/parallelizer.py` **L1216-1219** and
`nemo_automodel/components/moe/experts.py` **L491-493**:

```python
assert self.n_routed_experts % ep_size == 0, (
    f"Number of experts must be divisible by ep_size (ep_size={ep_size})")
```

288 % 8 = 0 ✓. **There is no third constraint.** No minimum EP, no minimum node count, no
`ep_size >= world_size`, no EP-size table.

**What `ep_shard_size = 1` means physically** — `parallelizer.py` **L900-921**: the experts get an
extra `fully_shard(...)` wrap **only** when `ep_shard_enabled`, and `ep_shard_enabled` is
`ep_shard_mesh is not None and ep_shard_mesh.size() > 1` (**L1254**). At EP8-on-8, `ep_shard_size == 1`,
so the experts are purely EP-sharded: **36 of 288 experts per GPU, fully resident, no all-gather.**
That is the whole of §2's arithmetic. The same is true at EP72 (4 experts/GPU) — the mechanism is
identical, only the divisor differs.

EP sharding itself is generic and dimension-agnostic — `parallelizer.py` **L295-308**:

```python
class ExpertParallel(ParallelStyle):
    def _partition_fn(self, name, module, device_mesh):
        assert device_mesh.ndim == 1
        for name, param in module.named_parameters(recurse=False):
            dist_param = nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)]))
            dist_param.requires_grad = param.requires_grad
```

Note `recurse=False` over *every* parameter on the experts module, sharded on dim 0 (the expert
axis), with `requires_grad` preserved. This is why **LoRA expert adapters are EP-sharded too**
(§4.3) — they are registered directly on the module and their dim 0 is also `n_routed_experts`.

### 1.4 Five shipped single-node EP8 recipes, one of them VLM + PEFT + `hybridep`

**measured** (read out of the repo):

| Recipe | `ep_size` | `nodes` | dispatcher | shape |
|---|---:|---:|---|---|
| `examples/vlm_finetune/nemotron_3_5_super_vl/nemotron_3_5_super_vl_cord_v2_peft.yaml` | 8 | **1** | **`hybridep`** | **VLM + LoRA + frozen tower, 121 B / 512 experts** |
| `examples/vlm_finetune/nemotron_3_5_super_vl/nemotron_3_5_super_vl_spider_peft.yaml` | 8 | **1** | **`hybridep`** | same |
| `examples/llm_benchmark/glm/glm_4.7_flash_lora.yaml` | 8 | 1 | `deepep` | GLM MoE LoRA |
| `examples/llm_finetune/qwen/qwen3_moe_30b_lora.yaml` | 8 | 1 | `deepep` | MoE LoRA |
| `examples/vlm_finetune/gemma4/gemma4_26b_a4b_moe_peft.yaml` | 8 | — | `deepep` | VLM MoE PEFT |

The first two matter most: they are `recipe: FinetuneRecipeForVLM` + a `peft:` block +
`ep_size: 8` + **`dispatcher: hybridep`** on **one node**, with `ci: nodes: 1`. That is the exact
combination the GLM recipe uses (`dispatcher: hybridep`, `experts: torch_mm`) and the exact
combination #67 flagged as unexercised. It is exercised, in CI, just not on `glm5_next`.

**Verdict on Q1: EP8 is supported. `ep_size: 8` on one 8-GPU node is a legal, plumbed,
CI-exercised configuration. Nothing in the implementation assumes EP72.**

---

## §2 — Peak memory per GPU for a frozen-base LoRA at EP8

### 2.1 The parameter budget (derived, shown)

From `zai-org/GLM-5.3-Flash/config.json` (**measured**, fetched 2026-09-26): `hidden_size 4096`,
`moe_intermediate_size 2048`, `intermediate_size 12288`, `n_routed_experts 288`,
`n_shared_experts 1`, `num_experts_per_tok 8`, `num_hidden_layers 45`, `vocab_size 154880`,
`mlp_layer_types` = 42 sparse + 3 dense, `layer_types` = 34 `linear_attention` + 11
`deepseek_sparse_attention`, `num_nextn_predict_layers 1`, `hc_mult 4`,
`linear_attn_config.num_heads 64 / head_dim 128`, `q_lora_rank 1536`, `kv_lora_rank 512`,
`index_topk 2048`. Vision: `depth 24`, `hidden_size 1024`, `intermediate_size 4096`.

HF API `?expand[]=safetensors` (**measured**): total **321,323,031,390** params; FP8 repo is
`F8_E4M3 314,396,639,232 + BF16 6,926,096,640 + F32 295,518`; BF16 sibling is
`BF16 321,322,735,872 + F32 295,518`. The 598.5 GiB / 642,652,070,880-byte figure #69 established is
confirmed and is for the **full** checkpoint including the MTP layer.

| Component | params | label |
|---|---:|---|
| One expert, one layer: `4096×2×2048 + 2048×4096` | 25,165,824 | derived (exact) |
| Routed experts, 42 sparse layers × 288 | **304,405,807,104** | derived (exact) |
| Routed experts, 43 layers (incl. MTP) | 311,653,564,416 | derived (exact) |
| ⇒ implied non-expert, incl. MTP: 321,323,031,390 − 311,653,564,416 | **9,669,466,974** | derived (exact) |
| 34 KDA attention layers | 4,682,617,088 | derived |
| 11 sparse-MLA layers incl. indexer | 1,374,036,224 | derived |
| 42 shared experts + 3 dense MLP + 42 routers | 1,559,506,752 | derived |
| `embed_tokens` + `lm_head` | 1,268,776,960 | derived |
| Vision tower + merger | ≈ 563,571,712 | derived (approximate) |
| mHC (`fn` 24×16384 × 2 per layer × 45) | 35,389,440 | derived |
| **Non-expert for the 45-layer model (MTP dropped)** | **≈ 9.45 B** | derived; bounded 9.42–9.52 B |

Cross-check: 304,405,807,104 + 9.45 B = **313.85 B**, and 311,653,564,416 + 9.67 B = 321.32 B =
the measured checkpoint total. The residual (the MTP layer's own non-expert tensors, ~0.2 B) is the
only slack. **The expert term is exact; the dense term is good to ±1%.**

**⇒ The model NeMo loads is ≈ 313.85 B params = 627.7 GB = 584.6 GiB in BF16.**
(Not 598.5 GiB — that includes the MTP layer, which `num_nextn_predict_layers: 0` drops, and which
`Glm5NextStateDictAdapter.from_hf` pops by key prefix, `state_dict_adapter.py` L232-236.)

### 2.2 Weight residency per GPU (derived)

EP shards the experts; FSDP2 shards everything else over `dp_shard × cp`.

| GPUs | EP | experts/GPU | expert GiB | dense GiB | **total GiB/GPU** | H100 (79.6) | H200 (140.4) |
|---:|---:|---:|---:|---:|---:|:---:|:---:|
| 4 (Sparks) | 4 | 72 | 141.8 | 4.40 | **146.1** | — | — |
| **8** | **8** | **36** | **70.9** | **2.20** | **73.1** | ✗ (6.5 left) | ✓ (67 left) |
| 16 | 8 or 16 | 36 / 18 | 35.4 | 1.10 | **36.5** | ✓ | ✓ |
| 24 | 24 | 12 | 23.6 | 0.73 | **24.4** | ✓ | ✓ |
| 72 | 72 | 4 | 7.9 | 0.24 | **8.1** | ✓ | ✓ |

**Cross-check against NeMo's published full-SFT peak.** At EP72/CP2, dp36: weights 8.1 GiB +
bf16 grads 8.1 + `torch.optim.AdamW` bf16 states 16.2 ≈ **32.4 GiB** of parameter memory
(**derived**), against NVIDIA's **measured 57.68 GiB peak** — leaving ≈ 25 GiB of activations,
dispatcher buffers and allocator slack at packed-2048/CP2/local-batch-1. That residual is large for
2048 tokens, which is itself informative: `hc_mult: 4` means the residual stream is
`[B, S, 4, 4096]`, four times a normal transformer's, and the DSA indexer materialises top-2048
selection state per DSA layer.

**The reconciliation, stated plainly:** at EP72, **56% of the 57.68 GiB peak is gradients and
optimizer state that a frozen base does not have.** Remove them and the same recipe's parameter
memory drops from 32.4 GiB to 8.1 GiB per GPU — which is why the identical job fits on 1/9 the GPUs
once the base is frozen, exactly as NVIDIA says for Nemotron 3.5 Super VL.

### 2.3 Can the base stay FP8? **No.**

This is the sub-question #71 flagged as changing everything. It does not change anything, because
the answer is no.

`dequantize_base_checkpoint` is a **loading-path** flag, not a residency mode.
`nemo_automodel/components/checkpoint/checkpointing.py` **L179-202**:

```python
def _should_dequantize_base_checkpoint(model, requested) -> bool:
    """Return whether this load requires checkpoint dequantization.
    ``requested`` permits dequantization unless it is explicitly ``False``.
    The conversion is needed only when the source model config declares a
    quantization method …"""
```

and `nemo_automodel/components/models/glm5_next/state_dict_adapter.py` **L207-215**:

```python
def _dequantize(self, state_dict):
    for key, value in list(state_dict.items()):
        scale_key = key + "_scale_inv"
        if key.endswith(".weight") and scale_key in state_dict:
            state_dict[key] = dequantize_block_fp8(value, state_dict[scale_key], dtype=self.dtype)
```

`self.dtype` is the model dtype, **bfloat16** (`state_dict_adapter.py` L188, set from the recipe's
`torch_dtype: bfloat16`). The model's parameters are BF16 `nn.Linear`/`nn.Parameter` throughout
(`glm5_next/layers.py`). The coverage doc states the same from the other side (L23, L47-48):
*"Training Precision: **BF16 after FP8 checkpoint dequantization**"*, *"The supported base-checkpoint
initialization path uses distributed checkpoint loading and **dequantizes the released FP8 weights
for BF16 training**"*.

**There is no FP8-resident training path.** NeMo's FP8 option (`apply_fp8_to_model`, torchao
float8) is *compute* quantization over BF16 master weights; it does not reduce weight residency.
NVFP4 exists in the tree only as a KV-**cache** format for `deepseek_v41`
(`components/models/deepseek_v41/quantization.py` L93-108); there is no NVFP4 base-weight training
path anywhere. So **the 305.8 GiB and 181.3 GiB figures are irrelevant to residency** — 584.6 GiB
BF16 is what sits in HBM.

**But they are relevant to data movement, and this is a useful new fact for #70.** The recipe's
`pretrained_model_name_or_path` is `zai-org/GLM-5.3-Flash` — the **FP8** repo — and NeMo
dequantizes it itself. **Do not download the 642.7 GB BF16 sibling.** Stage the 328.3 GB FP8 one.
That halves #69's largest fixed cost.

**The cost of that choice is the load-time transient.** Because the destinations must be FP8, they
cannot be views into the model's BF16 storage — `components/moe/state_dict_mixin.py` **L1058-1060**:

> "Quantization casts each split with `value.to(float8_e4m3fn)`, creating storage separate from the
> model's grouped weights. DCP must not treat that cast as model weight memory … Quantized loads
> therefore rebuild the grouped expert tensor after the read."

and `checkpoint_load_destination` returns `view.contiguous()` whenever `quantization` is true
(L1080-1088). The whole rank-local destination dict is built in one call before `dcp.load`
(`checkpointing.py` L1174-1186). **Derived** load peak:

| | BF16 model | + FP8 destinations | **load peak/GPU** |
|---|---:|---:|---:|
| 8 GPUs, EP8 | 73.1 GiB | 36.5 GiB | **109.6 GiB** |
| 16 GPUs, EP16 | 36.5 GiB | 18.3 GiB | 54.8 GiB |
| 72 GPUs, EP72 (NVIDIA's) | 8.1 GiB | 4.1 GiB | 12.2 GiB |

109.6 of 140.4 GiB fits; 109.6 of 79.6 does not. This is a second, independent reason 8×H100 is out,
and it is the most likely single-node failure mode.

### 2.4 Peak memory at 16k and 32k — the honest answer

**The weight term does not move with sequence length: 73.1 GiB/GPU at EP8, at any length.** The
question is entirely the non-weight term, and **I cannot derive it. It has to be measured.**

What I *can* offer is one measured anchor of the right shape and one exact lever.

**The anchor** (`docs/model-coverage/omni/nvidia/nemotron-3-5-super-vl.mdx` L52-63, **measured** by
NVIDIA): Nemotron 3.5 Super VL, 121 B / 512 routed experts / MoE VLM, LoRA rank 64 with the base,
vision tower, projector and `lm_head` frozen, 177 M trainable (0.15%), `ep_size 8`, `cp_size 1`,
8×H100, `max_length 4096`, `local_batch_size 1`, activation checkpointing on, TE `FusedAdam` with
fp32 master weights → **peak 38.8 GiB/GPU**, 400 steps in 13 min.

121 B BF16 / 8 GPUs = 28.2 GiB/GPU of weights (**derived**), so the **non-weight term there was
≈ 10.6 GiB/GPU at 4096 tokens** (**derived** from a measured peak).

**The lever.** CP is numerically validated for `glm5_next` — the coverage doc's
"Packed CP1 / CP8 Training Parity" table (L136-147, **measured**: 18 nodes / 144 H100, EP144, peaks
38.89 vs 41.04 GiB, final losses 1.2344 vs 1.2328, mean |Δ| 0.001879 across 100 matched steps) —
and CP shards the *sequence*, not the weights. On 8 GPUs every one of these meshes is legal by
§1.3's arithmetic:

| target seq | `cp_size` | `dp_size` | `ep_size` | `non_pp = dp·cp·tp` | tokens/rank | weights/GPU |
|---:|---:|---:|---:|---:|---:|---:|
| 8k | 1 | 8 | 8 | 8 ✓ | 8,192 | 73.1 GiB |
| **16k** | **2** | 4 | 8 | 8 ✓ | **8,192** | 73.1 GiB |
| **32k** | **8** | 1 | 8 | 8 ✓ | **4,096** | 73.1 GiB |

NeMo uses exactly this construction elsewhere —
`examples/llm_finetune/deepseek_v41/deepseek_v41_flash_tulu3_packed_cp8_32k.yaml` header:
*"packed Tulu3, **32768 tokens** … **CP8 splits each pack into contiguous 4096-token shards**"*.

**So the 32k/CP8 configuration puts the same 4,096 tokens per rank as the measured 38.8 GiB
Nemotron job.** If GLM's per-token activation cost were comparable, 32k would land near
73.1 + ~11 ≈ 84 GiB/GPU, inside 140.4. **It is not comparable, and I will not pretend it is:**
GLM carries `hc_mult: 4` residual streams (4× the activation-checkpoint boundary state) and a
`index_topk: 2048` DSA indexer that Nemotron's Mamba-2/attention hybrid does not have, and CP adds
its own state-carry buffers.

The honest bracket, **extrapolated, ±2×, from one datapoint on a different architecture**:

| config | weights | non-weight (extrapolated) | total | verdict |
|---|---:|---:|---:|---|
| 8k, CP1 | 73.1 | 20–45 GiB | 93–118 | probably fits, 140.4 available |
| 16k, CP2 | 73.1 | 20–45 GiB | 93–118 | probably fits |
| 16k, CP1 | 73.1 | 40–90 GiB | 113–163 | **coin flip** |
| 32k, CP8 | 73.1 | 12–25 GiB | 85–98 | probably fits |
| 32k, CP1 | 73.1 | 80–180 GiB | — | **no** |

**This is the number #72 exists to measure.** Do not provision on the extrapolated column;
provision on the weight column (which is solid) plus the biggest headroom available, and let the
probe find the sequence-length ceiling. The good news is that the lever (CP) costs nothing in
weights and is already validated for this model.

**Additional escape hatches, in order of preference, all shipped:**
`distributed.cpu_offload: true` (used by NeMo's own 32k CP8 recipe, L108);
`moe.reshard_after_forward: true`; a second node at EP16 (halves weights to 36.5 GiB/GPU).

---

## §3 — If one node cannot do it: the smallest topology that can

**The next step is two nodes, not nine.** Legal EP sizes are the divisors of 288 that also divide
`dp·cp·tp`; with 8 GPUs per node that means EP ∈ {8, 16, 24, 32, 48, 72, 96, 144, 288} (and EP8 with
`ep_shard > 1` is equivalent to a larger EP for memory purposes — §1.3).

| Topology | weights/GPU | load peak/GPU | fits H100 (79.6)? | fits H200 (140.4)? | provisioning friction |
|---|---:|---:|:---:|:---:|---|
| **1 node, 8×H200, EP8** | **73.1** | **109.6** | — | **✓** | **none** — SSH key, one VM, `torchrun --nproc_per_node=8` |
| 1 node, 8×H100, EP8 | 73.1 | 109.6 | ✗ | — | n/a |
| **2 nodes, 16×H200, EP16** | 36.5 | 54.8 | — | ✓✓ | GPU cluster + MK8s node group |
| **2 nodes, 16×H100, EP16** | 36.5 | 54.8 | **✓** | — | GPU cluster + MK8s node group |
| 9 nodes, 72×H100, EP72 | 8.1 | 12.2 | ✓ | — | NeMo's *full-SFT* recipe. Massively over-provisioned for a LoRA. |

**Provisioning friction for two nodes** *(carried from #69 §3; I did not re-verify Nebius docs)*: a
GPU cluster (InfiniBand fabric) and a Managed Kubernetes GPU node group are **both self-service** —
#69 quotes `docs.nebius.com/compute/clusters/gpu` verbatim: *"If you use the web console, you don't
need to complete any prerequisites."* Only **managed Slurm/Soperator** is capacity-block-gated.
Default `eu-north1` quota is 32 H200 and 32 H100, so both one-node and two-node shapes are inside
quota. The concrete delta versus one node:

1. `nebius compute gpu-cluster create --infiniband-fabric fabric-7` (H200 in `eu-north1`).
2. Create both VMs *with* `--gpu-cluster-id` — **a GPU cluster cannot be attached to a running VM**
   (#69, quoting the docs: *"You can assign a GPU cluster only when creating a VM"*). Decide before
   you launch.
3. Multi-node `torchrun`/`automodel` rendezvous, NCCL IB env, and a shared filesystem mounted on
   both nodes (the checkpoint must be visible to every rank).

That is roughly a half-day of setup versus zero, and it reinstates most of the porting surface #69
deleted by choosing one node. **Given that 73.1 GiB of 140.4 is a 52% fill with a validated CP lever
for the sequence-length term, one node is the right first attempt and two nodes is a cheap, fully
self-service fallback.** Nine nodes is not in the running for this job.

### 3.1 The 4×DGX-Spark column — closed on scale and framework, per #71's correction

I do **not** rely on the kernel argument; it was disproved on 2026-09-06
(`docs/research/sm121-fla-blocker-recheck.md`, branch `research/sm121-fla-blocker-recheck`).

**Scale (derived, decisive).** 313.85 B params BF16 = 627.7 GB across 4 GB10s = **156.9 GB per
node** against ~124.5 GB of unified memory. **Over by ~32 GB per node** before optimizer state,
activations, adapters, or the operating system — and unified memory is also the host RAM the
checkpoint load runs in. The split does not matter: EP4 gives 72 experts/GPU, EP2 gives 144 with
`ep_shard 2`, EP1 gives FSDP-over-4 — all three land on total/4. There is no arrangement of four
GB10s that holds this model in BF16.

**Framework.** NeMo states *"TP and PP are not supported for this model"* (coverage doc L168) and
`supports_tp: False` / `supports_pp: False` in source (`model.py` L227, L229), so the only axes are
EP, CP and FSDP — all of which shard, none of which compress.

**QLoRA over a quantized base — the honest re-cost, on framework grounds.**
The old dismissal rested on a dtype-independent kernel veto that is now known to be wrong, so this
deserved a fresh look. It closes anyway, for a different and verifiable reason:

1. **NVFP4 is not a training format in NeMo Automodel at all.** `grep -rn "nvfp4"` over
   `nemo_automodel/` returns only `components/models/deepseek_v41/quantization.py` and
   `attention.py` — **KV-cache** quantization for a different model. There is no NVFP4 base-weight
   path. The 181.3 GiB NVFP4 checkpoint would fit 4 pooled GB10s; nothing can train it.
2. **The only quantized-base path NeMo has is BitsAndBytes NF4, and requesting it *disables* the
   `glm5_next` implementation.** `nemo_automodel/_transformers/model_init.py` **L1289-1300**:

   ```python
   model_cls = _resolve_custom_model_cls_for_config(hf_config)
   if model_cls is not None:
       if quantization_config is not None:
           # BnB quantization is tightly integrated with HF's from_pretrained weight
           # loading pipeline.  Custom model constructors only create the architecture
           # (no weight loading, no quantization), so we must fall through to the HF
           # path which handles load + quantize atomically.
           logger.info("BnB quantization requested; using HuggingFace model loader for %s "
                       "(custom implementations do not support BnB quantization natively).", ...)
   ```

   The HF fallback then needs `transformers` to know `glm5_next`. NeMo pins
   **`transformers==5.15.1`** (`pyproject.toml` L89; coverage doc L44), and `glm5_next` first
   appears in **5.16.1** (#67, verified). So the fallback cannot construct the class.
3. **The Spark-oriented streaming BnB loader is explicitly unavailable for this class.**
   `model_init.py` **L934-960**, `_streaming_bnb_supported`: *"Automodel's custom implementations
   fuse projections … Detected via the `HFCheckpointingMixin` marker"* →
   `if issubclass(model_cls, HFCheckpointingMixin): return False`. And
   `Glm5NextForConditionalGeneration(HFCheckpointingMixin, nn.Module, MoEFSDPSyncMixin)`
   (`model.py` L208) is one.

   Even if the pin were raised, taking the HF path forfeits EP, `GroupedExpertsDeepEP`, the
   boundary-correct packing, and the state-dict adapter, landing on the bare `transformers`+PEFT
   route whose experts are 3-D `nn.Parameter`s (#66) — which is #67's §4 in full, on 4 GB10s.

4. **And the kernel cost is still real, just not fatal**: the re-check measured the torch reference
   at **3–5× to ~7× per training step** versus FLA (transformers issue #48148: *"25-30 s/it to
   169-200 s/it"*; FLA #913 on B200/B300: *"roughly 3-5x slower per training step"*), across 34 of
   45 layers. Applied on top of a ~14× GPU-count deficit, that is not a venue.

**Conclusion unchanged, grounds corrected: the Sparks are out on scale and framework. Not on
kernels.**

---

## §4 — LoRA target selection on the NeMo tree

#67 called composing the VLM recipe with a `peft:` block "a YAML change". **That is true of the
recipe machinery and false of the target list.** The `peft:` plumbing is fully wired for VLM; the
`glm_5.2_lora.yaml` target globs, copied verbatim, are wrong in three specific ways on this tree.

### 4.1 The `peft:` machinery is wired, and the base freeze is enforced twice

- `nemo_automodel/recipes/vlm/finetune.py` **L521-523** reads `cfg.peft` and **L551-552** threads it
  into `build_model` alongside `freeze_config`.
- `nemo_automodel/_transformers/infrastructure.py` **L161** applies it:
  `apply_lora_to_linear_modules(model, peft_config, quantization_config=..., skip_freeze=True)`.
- `nemo_automodel/components/_peft/lora.py` **L597-601** is the ordinary freeze
  (`for w in model.parameters(): w.requires_grad_(False)`), here deferred.
- `infrastructure.py` **L495-505**, `_apply_trainability_policy`, is the authoritative one, and it
  runs **three times** — before parallelization (strict), after TP/EP/AC surgery, and after
  checkpoint load:

```python
if peft_enabled:
    for name, param in model.named_parameters(remove_duplicate=False):
        param.requires_grad_("lora_" in name)
if freeze_config is not None:
    apply_parameter_freezing(model, freeze_config, strict=strict)
```

**Order is load-bearing and it is the right order:** the PEFT baseline freezes everything that is
not a `lora_*` parameter, and then `freeze_config` runs and can only freeze *more*.

### 4.2 The vision freeze holds — by module path, in code, after adapter injection

`components/utils/model_utils.py` **L645**:

```python
(freeze_vision_tower, "vision_tower", ("vision", "visual", "image_encoder")),
```

feeding **L314-322**:

```python
def _freeze_module_by_attribute_and_patterns(model, attribute_name, name_patterns):
    if attribute_name is not None and hasattr(model, attribute_name):
        getattr(model, attribute_name).requires_grad_(False)
    for name, module in model.named_modules():
        if any(pattern in name.lower() for pattern in name_patterns):
            module.requires_grad_(False)
```

The NeMo module tree (`glm5_next/model.py` L195-201, L269-274) is
`Glm5NextForConditionalGeneration.model.{visual, language_model}`, so every vision module path
contains the substring `visual` and the **entire `model.visual.*` subtree is set
`requires_grad=False`, including any LoRA adapter injected into it**.

**So #66's trap (b) cannot produce a *trained* vision tower on the NeMo tree.** That is a
meaningfully stronger guarantee than the `transformers`+PEFT route offers, and it is asserted by
module path, in code, exactly as the map demanded — not by leaf name and not by assumption.

**It can still produce *dead* adapters, and you should not ship those.** They consume memory, land
in `adapter_model.safetensors`, and quietly misreport the trainable-parameter count.

### 4.3 Where the `glm_5.2_lora.yaml` globs actually land (executed, not reasoned)

`apply_lora_to_linear_modules` calls `matcher.match(module, name)` with `name` from
`model.named_modules()` — i.e. the **full dotted path**, with `prefix=None`
(`lora.py` L620, L622, L663; `module_matcher.py` L130-146). So the `name == pattern` branch is an
exact *full-path* comparison and never fires; only the wildcard branch matters.
`_compile_wildcard_pattern` (`module_matcher.py` **L34-38**) turns `*.mlp.gate_proj` into the regex
`^(.*).mlp.gate_proj$`.

I ran that exact code against the real module paths:

| module path | matched by `glm_5.2_lora.yaml` globs? |
|---|---|
| `model.language_model.layers.0.mlp.gate_proj` (dense MLP) | ✓ `*.mlp.gate_proj` |
| `model.language_model.layers.5.mlp.shared_experts.gate_proj` | ✓ `*.mlp.shared_experts.gate_proj` |
| `model.language_model.layers.5.mlp.experts` | ✓ `*.mlp.experts` |
| `model.language_model.layers.3.self_attn.q_a_proj` (MLA) | ✓ `*.self_attn.q_a_proj` |
| `model.language_model.layers.5.self_attn.q_proj` (**KDA**) | **✗ — no pattern reaches it** |
| **`model.visual.blocks.7.mlp.gate_proj`** | **✓ `*.mlp.gate_proj` — the trap fires** |
| `model.visual.merger.gate_proj` (VL projector) | ✗ (path has no `mlp` segment) |
| `model.visual.blocks.7.attn.qkv` | ✗ (vision uses `attn.qkv`, not `self_attn`) |

Three findings:

- **#66's trap (a) reproduces and is worse than "34 of 45".** `glm_5.2_lora.yaml` targets
  `q_a_proj/q_b_proj/kv_a_proj_with_mqa/kv_b_proj`, which exist only on the **11** sparse-MLA
  layers. The **34 KDA layers** use `q_proj/k_proj/v_proj/f_a_proj/f_b_proj/b_proj/g_a_proj/
  g_b_proj/o_proj` (`glm5_next/layers.py` L297-310) and would receive **only `o_proj`** (which the
  5.2 glob does cover). Copied verbatim, the adapter reaches **1 of 9 projections on 76% of the
  layers**.
- **#66's trap (b) reproduces**, on `model.visual.blocks.*.mlp.{gate,up,down}_proj` — 24 vision MLP
  blocks × 3 projections get dead adapters. Frozen, but allocated and serialized.
- **#66's projector half does *not* reproduce on the NeMo tree.** NeMo names the merger's
  projections `model.visual.merger.{proj,gate_proj,up_proj,down_proj}`
  (`glm5_next/vision.py` L182-194) with no `mlp` segment, so `*.mlp.*` globs miss it. This is a
  NeMo-tree-specific correction to #66, which measured the `transformers` tree.

### 4.4 `exclude_modules` cannot be used to fix this

`module_matcher.py` **L110-113**:

```python
if self.target_modules and self.exclude_modules:
    raise ValueError(
        "target_modules and exclude_modules are mutually exclusive. Please provide only one of them.")
```

So the Nemotron recipe's style (`match_all_linear: false` + `exclude_modules: ["*vision_tower*", …]`)
and an explicit allowlist are **alternatives, not composable**. Given that `all-linear`-style
matching is fatal here (#69: 288 experts → ~14 GiB of adapters; #66: it also sweeps the frozen
tower), **the allowlist must be the mechanism, and every pattern in it must be anchored on the
language-model path prefix.**

### 4.5 The `GroupedExpertsLoRA` binding — it exists, and the recipe's backend is the supported one

`lora.py` **L620-621** matches `GroupedExperts | GroupedExpertsDeepEP | GroupedExpertsTE |
GroupedExpertsMoK` and dispatches to `patch_moe_module` (**L523-566**), which:

- **raises** `NotImplementedError` for `GroupedExpertsMoK` (L543-544) and **`GroupedExpertsTE`**
  (L545-546: *"LoRA is not supported for Transformer Engine (TE) expert modules."*);
- returns `GroupedExpertsDeepEPLoRA` for `GroupedExpertsDeepEP`;
- returns `GroupedExpertsLoRA` for `GroupedExperts`.

The GLM recipe sets `experts: torch_mm` + `dispatcher: hybridep`, which selects
`GroupedExpertsDeepEP` at `world_size > 1` (`components/moe/layers.py` L791-800) — **the supported
branch**. (Had the recipe used `experts: te`, expert LoRA would raise.) The adapters are
`lora_gate_and_up_A/B`, `lora_down_A/B` with leading dim `n_routed_experts`
(`_peft/lora_experts.py` L112-125), so `ExpertParallel._partition_fn`'s `Shard(0)` EP-shards them
(§1.3), and `components/moe/state_dict_mixin.py` L33 (`_LORA_EXPERT_SUFFIXES`) round-trips them to
the PEFT `ParamWrapper` layout on save. The binding is real, generic, and checkpoint-complete.

**But `"*.mlp.experts"` appears in exactly one recipe repo-wide** — `glm_5.2_lora.yaml` — so #67's
"untested" label stands, and it is untested on *any* VLM tree.

**Cost of turning it on** (**derived**, 42 sparse layers × 288 experts, EP-sharded over 8; AdamW
states in param dtype):

| rank | trainable params | per GPU: weights + grads + AdamW |
|---:|---:|---:|
| 8 | 1.39 B | **1.29 GiB** |
| 16 | 2.77 B | **2.58 GiB** |
| 32 (the 5.2 recipe's `dim`) | 5.55 B | **5.17 GiB** |

Note `moe_rank_scaling: true` (`lora.py` L629-646) divides the expert rank by
`n_activated_experts` (8 here), so `dim: 32` + `moe_rank_scaling` gives expert rank 4 while keeping
rank 32 on the dense projections — the Kimi recipe uses exactly that.

**Recommendation: start without expert LoRA.** Attention + shared experts + dense MLP at r=32 is
**112 M trainable params, 0.036% of the model, 0.03 GiB/GPU** — the same order as the 177 M / 0.15%
NVIDIA shipped for Nemotron 3.5 Super VL, which reached 8/20 exact match on its task. Adding expert
LoRA is a *quality* lever that costs 5 GiB/GPU and lights up an untested code path; decide it after
the probe, not before.

### 4.6 The config that provably excludes the vision tower

Every pattern anchored on `*.language_model.layers.*`. Verified by executing
`_compile_wildcard_pattern` + `ModuleMatcher.match` against the real paths — **zero** matches under
`model.visual.*`, and full coverage of all 45 LM layers.

```yaml
# Compose onto examples/vlm_finetune/glm5_next/glm5_3_flash_medpix_packed2k_ep72_cp2_100steps.yaml.
# Every glob is anchored on the language-model prefix: nothing under model.visual.* can match.
peft:
  _target_: nemo_automodel.components._peft.lora.PeftConfig
  target_modules:
    # --- KDA linear attention: 34 of 45 layers. The glm_5.2 recipe reaches NONE of these. ---
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

    # --- 288 routed experts: OFF for the first probe. Costs 5.17 GiB/GPU at dim 32,
    #     and GroupedExpertsDeepEPLoRA has never been exercised on a VLM tree.
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
  # NOTE: do NOT add `freeze_embeddings` — FreezeConfig has no such field and
  # parse_freeze_config raises on unknown keys. #67 reported the published recipe
  # shipping it; it was removed in a9bc22f9 and a unit test now asserts its absence
  # (tests/unit_tests/recipes/test_glm5_next_medpix_recipes.py L39).
```

**Deliberately excluded, and why** (all verified in source):

| excluded | reason |
|---|---|
| `model.visual.*` (all of it) | frozen by decision; unreachable by these globs; also frozen by `freeze_vision_tower` |
| `*.self_attn.indexer.*` (`wq_b`, `wk`, `weights_proj`) | the KPool indexer; NeMo runs it under `@torch.no_grad()` in places (`layers.py` L454, L482, L540, L564) — adapters there would be dead |
| `*.mlp.gate` (the MoE router) | an `nn.Parameter`, not `nn.Linear` (`moe/layers.py` L281-289) — unmatched anyway, and you do not want to perturb routing in a small tune |
| `lm_head` | `dim`-sized adapter over a 154,880-row head; breaks downstream serving assumptions |
| `*.attn_hc` / `*.ffn_hc` (mHC) | `nn.Parameter`, partly fp32 (`_keep_in_fp32_modules_strict`, `model.py` L216-220) |

**Assert the freeze, do not trust it.** `print_trainable_parameters`
(`components/utils/model_utils.py`) reports only aggregate counts, so add a per-prefix assertion to
the probe:

```python
from collections import defaultdict
buckets = defaultdict(lambda: [0, 0])          # prefix -> [trainable, total]
for name, p in model.named_parameters(remove_duplicate=False):
    key = ("visual" if ".visual." in name or name.startswith("model.visual")
           else "language_model" if "language_model" in name
           else "other")
    buckets[key][1] += p.numel()
    if p.requires_grad:
        buckets[key][0] += p.numel()
for k, (t, n) in buckets.items():
    print(f"{k:16s} trainable {t:>14,} / {n:>15,}")
assert buckets["visual"][0] == 0, "vision tower is trainable — the freeze did not hold"
assert buckets["other"][0] == 0, "lm_head/embeddings trainable — target globs are too broad"
```

Run it **after** `apply_model_infrastructure` returns, not before — the policy is re-resolved three
times and only the last pass reflects post-shard reality.

---

## §5 — Corrections to the record

| Claim on the map | Status |
|---|---|
| #67: *"NeMo's validated topologies are EP72/9 nodes and EP144/18 nodes"* | **Correct as of 2026-08-31, now superseded by the doc itself**: *"EP72 is the published validated topology, **not a model requirement**"* (coverage doc L55, added `cdc9147a`, 2026-08-29). |
| #67: *"57.68 GiB peak per GPU"* | **Correct and measured — but for full-parameter SFT.** ≈56% of it is gradients + AdamW state that a frozen base does not have. |
| #67: *"`transformers==5.12.1`"* | **Now `transformers==5.15.1`** (`pyproject.toml` L89; coverage doc L44, fixed in `e2f4fbb0`, PR #3939). Still below the 5.16.1 floor — NeMo owns the implementation. |
| #67: *"the recipe ships `freeze_embeddings: true`"* | **No longer true, and it was a bug.** Removed in `a9bc22f9`; `FreezeConfig` has no such field and `parse_freeze_config` raises on unknown keys. A unit test now asserts its absence. |
| #67/#69: *"BF16 weights are 598.5 GiB"* | **True for the checkpoint. The model NeMo loads is 584.6 GiB** — `num_nextn_predict_layers: 0` drops the MTP layer (7.25 B expert params + ~0.2 B). |
| #69: *"8×H200 leaves ~64 GiB/GPU"* | **Confirmed independently: 67 GiB steady-state, from 73.1 GiB of weights.** #69 got the right number from a coarser model (naive 8-way shard of 598.5 GiB = 74.8 GiB). Two errors cancelled: including the MTP layer, and ignoring that EP + FSDP shard different things. |
| #69: *"pull the 642.7 GB BF16 checkpoint"* | **Wrong checkpoint.** NeMo's recipe loads `zai-org/GLM-5.3-Flash` (FP8, 328.3 GB) and dequantizes on the fly. Halves the largest fixed cost in #69's wall-clock model. |
| #66: *"the VL projector shares `gate_proj`/`up_proj`/`down_proj` with the LM"* | **True on the `transformers` tree, false on the NeMo tree** — NeMo names it `model.visual.merger.*`, which `*.mlp.*` globs miss. The **vision MLP** half of the trap does reproduce. |
| Map: *"the Sparks are out, decisively on kernels"* | **Grounds replaced** per #71's own correction. Out on **scale** (156.9 GB/node needed vs ~124.5 available) and **framework** (no TP/PP; no NVFP4 training path; BnB routes away from the custom class into a `transformers` pin that predates `glm5_next`). |

---

## §6 — What I could NOT determine without hardware

Listed in the order [#72](https://github.com/kreuzhofer/dgx-manager/issues/72) should measure them.

1. **Does the base load at EP8 without OOM?** The FP8→BF16 dequantize allocates rank-local FP8
   destinations alongside BF16 model storage (§2.3). **Derived** peak is ~110 GiB/GPU; NVIDIA has
   only ever run this at ~12 GiB/GPU. Whether all destinations are live simultaneously depends on
   DCP's planner, which I cannot settle by reading. **Measure: `torch.cuda.max_memory_allocated()`
   immediately after `load_base_model` returns, before step 1.** If this fails, `ep_size: 16` on
   2 nodes is the fix and it halves both terms.

2. **The non-weight term at 16k and 32k.** The only anchor is 10.6 GiB/GPU at 4096 tokens on a
   *different* architecture (**derived from NVIDIA's measured 38.8 GiB**). GLM's `hc_mult: 4` and
   its `index_topk: 2048` DSA indexer both inflate it by unknown factors. **Measure: peak at
   (8k, CP1), (16k, CP2), (32k, CP8), all `ep_size: 8`.** The three-point curve is the answer, not
   any single point.

3. **s/step, and therefore every wall-clock figure on this map.** NeMo publishes mean TPS 9,489
   (cuDNN) / 8,378 (SDPA) with a 33:18 / 37:15 training loop for 100 steps at global batch
   144 × 2048-token packs on 72 H100. **Those three numbers are not mutually consistent under any
   reading I could find, and the doc does not define whether TPS is global or per-GPU** — 144 packs
   × 2048 tokens / 19.98 s per step implies ~14,800 tok/s globally, not 9,489. **I therefore refuse
   to compute an MFU from them, and I will not extrapolate a wall-clock.** What I can say is that
   #69's **20% MoE MFU is an assumption with no support anywhere in NeMo's published results**, and
   the only published GLM-5.3-Flash throughput is at a deliberately small-batch parity operating
   point that is the least favourable configuration they could have chosen. Treat every wall-clock
   estimate on this map as unsettled in the *pessimistic* direction. **Measure: tokens/s and s/step
   at a real sequence length, with the optimizer step included.**

4. **Whether `GroupedExpertsDeepEPLoRA` binds on the VLM tree.** Every link in the chain is verified
   by reading (§4.5), and `"*.mlp.experts"` exists in exactly one recipe, on an LLM. Only running it
   settles it. This is optional for the first probe — leave expert LoRA off.

5. **Whether `hybridep` performs well at EP8 intra-node.** Two shipped single-node EP8 VLM PEFT
   recipes use it (§1.4), so it *works*; whether the hybrid intra/inter-node dispatcher is the right
   choice when there is no inter-node hop is a throughput question, not a correctness one.
   `deepep` is what every other single-node EP8 recipe uses. **Measure both if time permits.**

6. **Whether `backend.attn: cudnn` is worth the FlashMLA build.** NVIDIA measured **+13.26% mean
   throughput** for cuDNN over SDPA at EP72/CP2 (coverage doc L155-161). It needs FlashMLA at
   `b7643bd54521f563b839b98289b5cd048c062ba2` built from source with submodules (L94-103). On H200
   (SM90+) the gate passes. **Start with `sdpa`, add cuDNN once it steps.**

7. **Everything Nebius.** I did not re-verify a single Nebius fact; §3's friction table and the
   recommended VM shape are carried from #69 and inherit its open items — in particular whether
   `gpu-h200-sxm` / `8gpu-128vcpu-1600gb` is creatable on the employee tenant's default quota
   (#69's §7 item 1, still the load-bearing unconfirmed claim), and the HuggingFace→Nebius download
   rate. The FP8-not-BF16 correction (§2.3) halves that download and should be folded into #70's
   measurement plan.

8. **Image-bearing examples end to end.** NeMo's own scope note (coverage doc L167): *"Image
   training is supported; video training is not."* The VLM dataset builders and collators exist, but
   the map's own `lib/dataset.py` still has no image path
   ([#55](https://github.com/kreuzhofer/dgx-manager/issues/55)). Out of scope here; it remains the
   critical path for an actual run in either venue.

---

## Reproducing these checks

```bash
git clone https://github.com/NVIDIA-NeMo/Automodel.git && cd Automodel   # HEAD 1358302c, 2026-09-25

# Q1 — EP8 is legal; EP72 is not a requirement
sed -n '40,56p;165,176p' docs/model-coverage/vlm/thudm/glm-5-3-flash.mdx
sed -n '223,231p' nemo_automodel/components/models/glm5_next/model.py
sed -n '314,321p'   nemo_automodel/components/distributed/mesh_utils.py
sed -n '295,309p;898,922p;1214,1256p' nemo_automodel/components/moe/parallelizer.py
grep -rn "ep_size\|ep_mesh" nemo_automodel/components/models/glm5_next/    # zero hits
grep -rln "^peft:" examples/vlm_finetune | xargs grep -l "ep_size: 8"      # single-node VLM EP8 precedents

# Q2 — the base is dequantized to BF16; the frozen-base collapse is NVIDIA's own claim
sed -n '179,203p;1035,1070p' nemo_automodel/components/checkpoint/checkpointing.py
sed -n '182,216p' nemo_automodel/components/models/glm5_next/state_dict_adapter.py
sed -n '1056,1090p' nemo_automodel/components/moe/state_dict_mixin.py
sed -n '50,64p'   docs/model-coverage/omni/nvidia/nemotron-3-5-super-vl.mdx   # 38.8 GiB, one node
sed -n '16,30p'   examples/vlm_finetune/nemotron_3_5_super_vl/nemotron_3_5_super_vl_cord_v2_peft.yaml

# Q3 — Sparks: no TP/PP, no NVFP4 training, BnB routes away from the custom class
grep -rn "nvfp4" --include=*.py nemo_automodel/ | grep -v deepseek_v41   # empty
sed -n '934,960p;1288,1302p' nemo_automodel/_transformers/model_init.py

# Q4 — matcher semantics; run the real code on the real paths
sed -n '30,60p;108,153p' nemo_automodel/components/_peft/module_matcher.py
sed -n '523,566p;595,624p' nemo_automodel/components/_peft/lora.py
sed -n '493,506p'  nemo_automodel/_transformers/infrastructure.py
sed -n '312,324p;640,655p' nemo_automodel/components/utils/model_utils.py
python3 -c "
import re
c=lambda p: re.compile('^'+re.sub(r'\.\*','(.*)',re.sub(r'(?<!\.)\*','.*',p))+'$')
print(bool(c('*.mlp.gate_proj').match('model.visual.blocks.7.mlp.gate_proj')))             # True  <- trap
print(bool(c('*.language_model.layers.*.mlp.gate_proj').match('model.visual.blocks.7.mlp.gate_proj')))  # False
"

# Checkpoint sizes (HuggingFace API)
for r in zai-org/GLM-5.3-Flash zai-org/GLM-5.3-Flash-BF16; do
  curl -s "https://huggingface.co/api/models/$r?expand[]=safetensors" | python3 -m json.tool; done
curl -sL https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/config.json
```

---

## Confidence summary

| Claim | Label |
|---|---|
| Coverage doc says EP72 is "not a model requirement"; TP/PP unsupported; single-GPU unsupported | **measured** — verbatim, `glm-5-3-flash.mdx` L53-55, L168-171 |
| `supports_ep: True`; only two EP constraints in the plumbing; `ep_shard_size = 1` at EP8-on-8 | **measured** — source, line-cited |
| Five shipped single-node EP8 recipes, two of them VLM+PEFT+`hybridep` with `ci: nodes: 1` | **measured** — repo files |
| Checkpoint totals (321,323,031,390 params; FP8/BF16 dtype split) and `config.json` dimensions | **measured** — HF API + checkpoint, 2026-09-26 |
| Routed-expert params = 304,405,807,104 (42 layers) | **derived** — exact shape arithmetic |
| Model NeMo loads ≈ 313.85 B = 584.6 GiB BF16 | **derived**, dense term ±1%, cross-checked against the measured total |
| **73.1 GiB/GPU of frozen weights at EP8 on 8 GPUs** | **derived** — the load-bearing number of this document |
| Nemotron 3.5 Super VL: 38.8 GiB/GPU peak, frozen-base LoRA, EP8, 1 node of 8×H100, 4096 tokens | **measured** — NVIDIA's published coverage doc |
| Non-weight term ≈ 10.6 GiB/GPU at 4096 tokens | **derived** from that measured peak |
| Non-weight term at 16k / 32k on `glm5_next` | **extrapolated, ±2×, from one datapoint on a different architecture** |
| FP8→BF16 load transient ≈ 110 GiB/GPU at EP8 | **derived** from the source's own comment; simultaneity of the destinations **unverified** |
| The base cannot stay FP8; no NVFP4 training path; BnB disables the custom class | **measured** — source, line-cited |
| `*.mlp.gate_proj` matches `model.visual.blocks.*.mlp.gate_proj`; anchored globs do not | **measured** — NeMo's own matcher code, executed |
| `freeze_vision_tower` freezes `model.visual.*` by path substring, after PEFT | **measured** — source, line-cited |
| Sparks need 156.9 GB/node vs ~124.5 available | **derived** |
| Wall-clock / s/step / MFU | **unknown.** NeMo's three published throughput numbers are mutually inconsistent; #69's 20% MFU has no support. Settled only by #72. |
| Nebius shapes, quotas, regions, access friction | **carried from #69, not re-verified here** |
