# From a trained LoRA adapter to a served NVFP4 model on the RTX 5090

**Research date:** 2026-08-30. Resolves [#48](https://github.com/kreuzhofer/dgx-manager/issues/48).

**Sources.** Primary throughout. The strongest evidence here is not a document at all —
it is the `unsloth/Qwen3.8-27B-NVFP4` checkpoint, already on `/mnt/tank/models/hub`,
whose `config.json` and safetensors headers were read directly, byte by byte.
Everything else is `llm-compressor` / `compressed-tensors` / `transformers` / `vllm` /
`Model-Optimizer` source at a named ref, HuggingFace repo files, or PyPI release
metadata. No claim rests on a blog post or a summary.

---

## Verdict

**A proven path exists. It needs one piece of build work — about 40 lines — and that
work is not optional.**

Four findings, in descending order of how much they change the picture.

**1. The blocker in `docs/ROADMAP.md` is stale.** The `llmcompressor` ↔ `transformers`
pin conflict that killed the offline FP8 plumbing (#43) was fixed upstream in **June
2026**. `llmcompressor` 0.13.0 pins `transformers<=5.14.1,>=5.9.0` — it now *requires*
transformers 5.x, the very thing `qwen3_5` needs. The conflict was real against 0.10.0
and is not real now. **This unblocks offline FP8 as well as NVFP4**, so the roadmap line
slating that plumbing for removal should be re-read before anyone rips it out.

**2. The recipe is canonical, not reverse-engineered.** Three independent parties —
unsloth, RedHatAI (who *maintain* llm-compressor), and NVIDIA (in `Model-Optimizer`'s
shipped `qwen3_5` recipe) — converge module-for-module on the same split: **NVFP4 on MLP
projections, FP8 on all attention, BF16 on the vision tower and the MTP head.** Two
different toolchains, same answer. §2 and §6 lay the three side by side.

**3. That "FP8 on attention" is load-bearing, not incidental.** Qwen3.8-27B has
`attn_output_gate: true` — `q_proj` emits a fused Q half *and* a gate half. There is
evidence that putting NVFP4 on that gate degrades this architecture badly (perplexity
2–6× worse). **The obvious simple recipe — uniform `scheme="NVFP4"` over all Linear
layers — is the wrong one here**, even though a popular public checkpoint of this exact
model uses it. §7.

**4. MTP survives only because someone puts it back.** `transformers` *silently
discards* every `mtp.*` tensor on load, by an explicit regex, and llm-compressor's own
first-party example says so out loud. A naive quantize therefore yields a checkpoint
that loads, serves, looks healthy, and has no MTP head — costing exactly the 2.76
tokens/forward the gate is measuring, with no error anywhere. Every published
MTP-preserving NVFP4 of this model re-attaches the head manually. §3.

**Recommended path: Recipe B in §6** — `model_free_ptq`, which needs no calibration
data, no new quantization code, preserves MTP automatically because it never loads the
model through transformers, and reproduces NVIDIA's shipped module split. Recipe A
bit-matches the checkpoint already measured at 124.9 tok/s and is the fallback.

---

## 1. The pin conflict, resolved

`docs/ROADMAP.md` records the offline FP8 quantize plumbing as dead, "driven by an
unresolvable llmcompressor↔transformers pin conflict (llmcompressor 0.10 pins
`transformers<=4.57.6`; qwen3_5 needs `>=5.0`)". Accurate for 0.10.0; obsolete for
everything after it. From PyPI metadata (`https://pypi.org/pypi/llmcompressor/<v>/json`):

| `llmcompressor` | uploaded | `transformers` pin | `compressed-tensors` |
|---|---|---|---|
| 0.10.0 | 2026-02-28 | `<=4.57.6,>=4.56.1` | `==0.14.0` |
| 0.12.1a20260716 | 2026-07-17 | `>=5.9.0` | `>=0.17.2a2` |
| **0.13.0** (latest stable) | **2026-08-11** | **`<=5.14.1,>=5.9.0`** | **`==0.18.0`** |

The window inverted: `transformers<=4.57.6` became `transformers>=5.9.0` in the
0.11–0.12 line, and 0.13.0 caps at exactly `5.14.1`.

That cap matters. **`5.14.1` is the `transformers_version` stamped into
`unsloth/Qwen3.8-27B-NVFP4`'s `config.json`**, and its `compressed-tensors` version is
`0.17.2.a20260716` — matching the `0.12.1a20260716` alpha *uploaded the same day*. We
are not guessing at the toolchain; we can read it off the artifact.

The competing toolchain has no conflict either. NVIDIA `Model-Optimizer` (the repo was
renamed from `TensorRT-Model-Optimizer`) pins `transformers>=4.57,<5.15` — 5.x is
*permitted, not required*, so you supply the floor yourself.

**Consequence for #43.** The FP8 path was abandoned for a constraint that no longer
exists. `scripts/quantize_fp8.py` may well work today against `llmcompressor 0.13.0` —
but it has two independent bugs regardless (§8), so "un-abandon it" is a decision, not a
no-op.

---

## 2. What the served checkpoint actually is

Read from `/mnt/tank/models/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/`.

**It is not modelopt.** There is no `hf_quant_config.json`. `config.json` carries:

```json
"quantization_config": {
  "quant_method": "compressed-tensors",
  "format": "mixed-precision",
  "quantization_status": "compressed",
  "version": "0.17.2.a20260716"
}
```

`"mixed-precision"` is not a scheme — it is the marker `compressed-tensors` writes when
a model is compressed by **two different compressors**. The two groups:

| group | format | scheme | targets |
|---|---|---|---|
| `group_0` | `float-quantized` | FP8 W8A8 — weights per-**channel** static, activations per-**token** dynamic | `self_attn.(q\|k\|v\|o)_proj`, `linear_attn.(in_proj_qkv\|in_proj_z\|out_proj)`, `lm_head`, **and `layers.(56–63).mlp.(gate\|up\|down)_proj`** |
| `group_1` | `nvfp4-pack-quantized` | NVFP4 W4A4 — group_size 16, `scale_dtype float8_e4m3`, `strategy tensor_group`, weights `actorder: static`, input activations `dynamic: "local"` with a calibrated `static_minmax` global scale | all other `mlp.(gate\|up\|down)_proj` |

Plus `kv_cache_scheme`: FP8, per-tensor, static — the `k_scale`/`v_scale` tensors are
present on the 16 full-attention layers (11, 15, …). The shipped serving recipe does
*not* pass `--kv-cache-dtype fp8`, so those scales are currently carried and unused.

`ignore` (303 entries) holds three distinct things:
- the **entire vision tower** (`model.visual.blocks.0–26` attn/mlp + `merger`),
- the GatedDeltaNet **non-projection** parts of every linear-attention layer
  (`linear_attn` itself, `.norm`, `.in_proj_a`, `.in_proj_b`) — note `in_proj_qkv`,
  `in_proj_z`, `out_proj` are *not* ignored, they are FP8 in group_0,
- and, last in the list, **`"re:^mtp.*"`**.

Two editorial choices are worth copying rather than rediscovering. **The last 8 layers
(56–63) are deliberately held at FP8 rather than NVFP4** — a standard accuracy carve-out
for the layers nearest the output. And `lm_head` is FP8 rather than BF16.

Verified against the actual tensor headers:

```
lm_head.weight                                       F8_E4M3  (248320, 5120)
lm_head.weight_scale                                 BF16     (248320, 1)
…layers.3.mlp.gate_proj.weight_packed                U8       (17408, 2560)   <- NVFP4
…layers.3.mlp.gate_proj.weight_scale                 F8_E4M3  (17408, 320)
…layers.3.mlp.gate_proj.weight_global_scale          F32      (1,)
…layers.3.mlp.gate_proj.input_global_scale           F32      (1,)            <- calibrated
…layers.3.self_attn.q_proj.weight                    F8_E4M3  (12288, 5120)   <- FP8
…layers.60.mlp.gate_proj.weight                      F8_E4M3  (17408, 5120)   <- FP8 carve-out
```

### The same recipe, from the tool's maintainers

**`RedHatAI/Qwen3.8-27B-NVFP4`** (created 2026-08-17, 13,942 downloads; tags
`llm-compressor`, `nvfp4`, `compressed-tensors`) has an **identical** quantization
config: same `compressed-tensors` build `0.17.2.a20260716`, same `transformers_version`
5.14.1, same two groups with the same target regexes, same `re:^mtp.*` exclusion, same
`kv_cache_scheme`, and the same file layout **including a separate
`model_mtp.safetensors`**. Both main files are byte-identical in size
(22,568,192,096 and 849,400,392).

RedHatAI is the org that maintains `llm-compressor`. So the recipe decoded above is
theirs, applied to this exact model.

**This also resolves sglang#34895.** That issue reports a *dropped* `lm_head weight_scale`
causing degenerate repetition. The scale **is present** — BF16, `(248320, 1)`. The
RedHatAI model card says why, at the very top:

> This quant ONLY works in vLLM and **NOT SGLang**. The `lm_head` is quantized to FP8 and
> SGLang can't load it

It is an SGLang loader limitation — reportedly fixed on sglang `main` by PR #35228
(2026-08-19), not yet in a release — not a defect in the weights. That matches the
recipe's own measured finding that the degeneration never materialised under vLLM. The
`⚠ KNOWN RISK` block in `qwen3.8-27b-nvfp4-rtx.yaml` can be downgraded to a note.

**One forward-looking consequence:** any checkpoint we produce with `lm_head` in the FP8
group is **vLLM-only**. If SGLang portability ever matters, leave `lm_head` in BF16 —
which is what NVIDIA's and several community recipes do, and what unsloth declined to do
on size grounds (HF discussion #22: *"making it even bigger will cause no person with a
5090 be able to run this"*).

---

## 3. Does MTP survive? Yes — because it is copied, not quantized

The sub-question that matters most, with a sharper answer than expected.

### The landmine

`transformers` refuses to load the MTP weights. From
[`modeling_qwen3_5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
on `main`:

```python
class Qwen3_5PreTrainedModel(PreTrainedModel):          # line 873
    ...
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*"]    # line 881

class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):          # line 1652
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]    # line 1658
```

`Qwen3_5ForConditionalGeneration` (line 1749) subclasses `Qwen3_5PreTrainedModel` and
does not override this, so it inherits `[r"^mtp.*"]`. There is **no MTP module anywhere
in the transformers implementation** — a full grep for `mtp` in `modeling_qwen3_5.py`
returns those two lines and nothing else.

So anything shaped `from_pretrained(...)` → quantize → `save_pretrained(...)` — which is
exactly `llmcompressor.oneshot`, and exactly what `scripts/quantize_fp8.py` does today —
**drops all 15 MTP tensors, silently, by design**. No warning, because suppressing the
warning is the entire purpose of `_keys_to_ignore_on_load_unexpected`.

llm-compressor's own first-party example states this as a known property. From
[`examples/quantization_w4a4_fp4/qwen3_5_example.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_5_example.py):

```python
# No need to ignore mtp layers as they are not loaded
# through Qwen3_5MoeForConditionalGeneration
```

The resulting model serves fine. It just has no MTP head, so
`--speculative-config '{"method":"mtp",...}'` has nothing to load. **This is precisely
the silent-failure shape the issue worried about, and it is real.**

### Why the fix is trivial

unsloth hit this and solved it by copying. Their MTP weights live in a separate
**`model_mtp.safetensors`** (0.79 GiB) rather than inline, with all 15 keys registered in
`model.safetensors.index.json`:

```
files: Counter({'model.safetensors': 1953, 'model_mtp.safetensors': 15})
```

All 15 are **BF16 and unquantized** — `mtp.fc.weight` BF16 `[5120, 10240]`,
`mtp.layers.0.mlp.down_proj.weight` BF16 `[5120, 17408]`, and so on. The `"re:^mtp.*"` in
`ignore` is the config-side record of that.

And they are **byte-identical to the base checkpoint**. SHA-256 over the raw tensor bytes
in `Qwen/Qwen3.8-27B`'s `model-00018-of-00018.safetensors` (which holds all 15 MTP keys)
versus unsloth's `model_mtp.safetensors`:

```
mtp.norm.weight                       BF16 (5120,)         414fb74a971b168f == 414fb74a971b168f
mtp.pre_fc_norm_hidden.weight         BF16 (5120,)         f2e1aa414ce84d89 == f2e1aa414ce84d89
mtp.layers.0.self_attn.k_proj.weight  BF16 (1024, 5120)    0ab4e67930451a2f == 0ab4e67930451a2f
mtp.fc.weight                         BF16 (5120, 10240)   4eee377b67ec2122 == 4eee377b67ec2122
mtp.layers.0.mlp.down_proj.weight     BF16 (5120, 17408)   2fdda9751e3b7c14 == 2fdda9751e3b7c14
```

The MTP head was extracted from the base and re-attached verbatim. Nothing was re-derived.

### This is universal practice, not a hack

Every party that ships an MTP-preserving NVFP4 of this architecture does the same thing:

- **unsloth / RedHatAI** — `model_mtp.safetensors`, 15 BF16 tensors, `ignore: re:^mtp.*`.
- **`sakamakismile/Qwen3.8-27B-MTP-NVFP4`** (2026-08-14, 93,425 downloads) — ships its
  `recipe.yaml` publicly with `ignore: [lm_head, 're:.*visual.*', 're:.*conv1d.*',
  're:.*mtp.*']`, and re-attaches as `model-mtp-bf16.safetensors`. Its index confirms
  the mechanics exactly: `Counter({'model.safetensors': 2672,
  'model-mtp-bf16.safetensors': 15})`, `total_size: 20559282592` (the MTP bytes are
  counted in).
- **NVIDIA** — the shipped `qwen3_5` modelopt recipe disables `'*mtp*'` quantizers
  outright, and PR #1868 for their Qwen3.5-397B NVFP4 records *"MTP block left in BF16."*

Note the FP8 sibling `unsloth/Qwen3.8-27B-FP8` *does* quantize its MTP (22 keys) — so
BF16-MTP is an NVFP4-pipeline convention, not a property of the model.

### And for a *merged* model, they are still byte-identical

`recipes/qwen3.6-27b-base-lora-attn-mlp/train.py` **freezes the MTP head during
training** — the suffix matcher would otherwise hit `mtp.layers.0.self_attn.{q,k,v,o}_proj`:

```python
for name, p in model.named_parameters():
    if ".gate.weight" in name or "router" in name or "mtp." in name:
        if p.requires_grad:
            p.requires_grad = False
```

So the adapter carries no MTP deltas, the merge writes none, and the merged model's MTP
tensors equal the base's. **The MTP head can be copied straight out of
`Qwen/Qwen3.8-27B` at any point in the pipeline.** It never needs to survive anything —
it needs to be re-attached.

> Two caveats. If a future recipe ever *unfreezes* the MTP head, this shortcut breaks and
> the tensors must come from the merged model instead — the merge script preserves them
> (§4), so that path stays open. And because the MTP head is never fine-tuned, a merged
> model's drafter stays calibrated to the *base* output distribution; if your fine-tune
> shifts that distribution much, expect acceptance to drop below the measured 58.8%.
> That is a speed effect, not a correctness one.

---

## 4. The merge step — already proven for this architecture

`scripts/merge_qwen3moe.py` in `dgx-manager-fine-tune-recipes` applies unchanged, and
this is not an inference: **`recipes/qwen3.6-27b-base-lora-attn-mlp/recipe.yaml` already
declares it** (`merge: scripts/merge_qwen3moe.py`) for Qwen3.6-27B — which
`docs/qwen3.8-model-survey.md` establishes is field-for-field identical in config to
Qwen3.8-27B. Same `Qwen3_5ForConditionalGeneration`, 64 layers,
`full_attention_interval: 4`, `mtp_num_hidden_layers: 1`.

Three properties matter:

- **It solves the wrapper problem the right way.** It never calls PEFT's
  `merge_and_unload()`; it adds LoRA deltas into the base safetensors and rewrites them
  with identical layout, keys and config. The output is byte-compatible with the base's
  serving path — no `model_type=qwen3_5_text` leaf config for vLLM to choke on.
- **The MoE machinery is inert on a dense model.** The name is historical. `compute_delta`
  Case 1 (`base_tensor.ndim == 2`) handles dense 2D LoRA on q/k/v/o/gate/up/down_proj;
  the 3D expert path never matches. `--num-experts` is meaningless here.
- **It preserves MTP through the merge.** It processes one shard at a time, loading *all*
  tensors, applying deltas only where a target matches, writing every tensor back. The 15
  `mtp.*` keys in `model-00018-of-00018.safetensors` pass through unchanged, and
  `model.safetensors.index.json` is copied verbatim with the other non-safetensors files.

One caveat: it hard-exits on a single-file base (`"Single-file safetensors not supported
by this script"`). `Qwen/Qwen3.8-27B` is 18 shards, so this does not bite — but it is why
you cannot point the merge at an already-quantized single-file checkpoint. **Merge first,
quantize second.** That order is forced anyway: no library currently supports LoRA
training against NVFP4 weights (modelopt #1294, open and unanswered since 2026-04:
*"No training library supports NVFP4 right now… PEFT, Unsloth, Axolotl, TRL all need an
FP4 dequant-matmul kernel with PyTorch-tracked gradients, which hasn't shipped yet."*).

**No fine-tune recipe for Qwen3.8-27B exists yet** — the recipes repo has
`qwen3.6-27b-base-lora*` and nothing for 3.8. Creating one is a copy-and-retarget of the
3.6 attn-MLP recipe, not new design work, but it has to happen and is outside this issue.

---

## 5. Where each step runs — the 51.7 GiB problem

The merged BF16 model is 51.7 GiB; the 5090 holds 31.8 GiB usable. Real constraint,
**not** a blocker, and it does **not** force a Spark.

`llmcompressor` quantizes big models with **sequential onloading**, which is
[enabled by default](https://github.com/vllm-project/llm-compressor/blob/main/examples/big_models_with_sequential_onloading/README.md):

> Instead of loading the entire model into memory—which can easily require hundreds of
> gigabytes—this method loads and compresses one layer at a time. […] Sequential
> onloading is enabled by default within LLM Compressor.

The worked example is Llama-3.3-70B — "larger than 80 GB, surpassing the size of 1 A100
… can still be quantized seamlessly using a single GPU". The model sits on **CPU**
(`device_map=None`) and one decoder layer at a time is onloaded.

So the binding constraint is **host RAM, not VRAM**:

| host | RAM | VRAM | verdict |
|---|---|---|---|
| aihost01 (RTX 5090) | 62 GB | 32.6 GB | 51.7 GiB + calibration activations in 62 GB is **tight but plausible** |
| a Spark (GB10) | 121.6 GB unified | (same pool) | **comfortable** — CPU and GPU memory are one pool |

**Recommendation: quantize on a Spark, serve on the 5090.** Not because the 5090 cannot
— it probably can — but because the Spark removes the one variable that turns a
40-minute job into a debugging session, and leaves the 5090 free to serve. Recipe B
reduces the pressure further by never loading the model as a model at all.

If you do use the 5090 and reach for modelopt instead, **use `--offload_folder` disk
offload, not `--low_memory_mode`**: modelopt #2160 (open, unrebutted) reports
`--low_memory_mode` quantizing *meta* tensors before real weights load, producing a
silently broken NVFP4 export (dequant cosine 0.756 against BF16, vLLM aborts on a
dimension mismatch) while printing a success message. The supported single-GPU path for
oversized models is PR #2008 (merged 2026-08-05, in 0.46.0), which materializes one
decoder layer at a time and added a meta-tensor guard that raises instead of corrupting.

**Serving BF16 on the 5090 is not an option** and should not be discussed as one: 51.7
GiB against 31.8 GiB usable is not a tuning gap. An unquantized comparison point has to
be a Spark, which is exactly what `qwen3.8-27b-bf16.yaml` is for. That recipe measured
9–12 tok/s against this one's 124.9 — and, as its own header warns, that ratio is ~6.5×
memory bandwidth × ~2.4× fewer weight bytes, not a model result.

---

## 6. The recipes

All pin `llmcompressor==0.13.0` (2026-08-11) with `compressed-tensors==0.18.0` and
`transformers` in `[5.9.0, 5.14.1]`. Serving is the existing
`recipes/dgxrun/qwen3.8-27b-nvfp4-rtx.yaml` with `model:` pointed at the output
directory — **nothing in that recipe needs to change**.

### The module split everyone agrees on

Before the recipes, the thing they encode. Two unrelated toolchains, three publishers:

| module | NVIDIA modelopt `qwen3_5` recipe | unsloth / RedHatAI (llm-compressor) |
|---|---|---|
| `mlp.(gate\|up\|down)_proj` | **NVFP4** | **NVFP4** |
| `self_attn.*` | **FP8** (weight + input) | **FP8** W8A8 |
| `linear_attn.(in_proj_qkv\|in_proj_z\|out_proj)` | **FP8** | **FP8** |
| `linear_attn.(in_proj_a\|in_proj_b)`, `conv1d` | disabled → BF16 | ignored → BF16 |
| `visual` / `vision_tower` | disabled → BF16 | ignored → BF16 |
| **`mtp`** | **disabled → BF16** | **ignored → BF16** |
| KV cache | FP8 cast | FP8 static |
| `lm_head` | NVFP4 | FP8 |

From `modelopt_recipes/huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast.yaml`,
whose own description reads: *"NVFP4 for MLP projection weights and lm_head; FP8 for
self-attention and the large linear-attention projections; FP8 KV cache with constant
amax."* The only disagreement in the whole table is `lm_head`.

Note also that NVIDIA's is **W4A16** — NVFP4 weights, BF16 activations — while
unsloth's is W4A4. Both ship. That is the licence for Recipe B.

### Recipe B — `model_free_ptq` (RECOMMENDED; no new quantization code)

`model_free_ptq` [works directly on the safetensors](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/README.md):

> `model_free_ptq` works directly with the safetensors in the checkpoint to which
> observers are applied, thereby removing the requirement for a model definition or
> transformers.

That single sentence eliminates the MTP landmine. transformers is never involved, so
`_keys_to_ignore_on_load_unexpected` never fires. And
[`process.py`](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/model_free/process.py)
loads every tensor in a shard, quantizes only those matching a scheme and not in
`ignore`, then `save_file(tensors, save_path)` writes **all** of them back — ignored
tensors pass through verbatim into the rebuilt weight map. **No `attach_mtp.py` needed.**

There is a first-party example for this family:
[`examples/model_free_ptq/qwen3.5_int8.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/model_free_ptq/qwen3.5_int8.py)
already ignores `re:.*visual.*`, `re:.*conv1d.*`, `re:.*norm.*`, `re:.*embed_tokens.*`.

```python
# llmcompressor==0.13.0, compressed-tensors==0.18.0, transformers 5.9.0–5.14.1
from compressed_tensors.quantization import QuantizationConfig, QuantizationScheme
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4A16
from llmcompressor import model_free_ptq

IGNORE = [
    "re:^mtp.*",                        # MTP head stays BF16 and INTACT
    "lm_head",                          # BF16 -> also keeps SGLang able to load it
    "re:.*visual.*",                    # whole vision tower
    "re:.*conv1d.*",                    # GatedDeltaNet depthwise conv
    "re:.*linear_attn\\.in_proj_(a|b)$",
    "re:.*norm.*",
    "re:.*embed_tokens.*",
]

model_free_ptq(
    model_stub="/mnt/tank/outputs/<jobId>/merged-clean",
    save_directory="/mnt/tank/outputs/<jobId>/merged-nvfp4",
    config=QuantizationConfig(
        config_groups={
            # FP8 on ALL attention -- see §7, this is not optional on this architecture
            "group_0": QuantizationScheme(**FP8_DYNAMIC, targets=[
                r"re:.*self_attn\.(q|k|v|o)_proj$",
                r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
            ]),
            # NVFP4 weight-only on the MLPs -- NVIDIA's shipped W4A16 split
            "group_1": QuantizationScheme(**NVFP4A16, targets=[
                r"re:.*mlp\.(gate|up|down)_proj$",
            ]),
        },
        ignore=IGNORE,
    ),
    max_workers=15,
    device="cuda:0",
)
```

**Why W4A16 and not W4A4 here.** Full `NVFP4` (W4A4) is rejected by the model-free path,
deliberately and with a clear error. From
[`validate.py`](https://github.com/vllm-project/llm-compressor/blob/main/src/llmcompressor/entrypoints/model_free/validate.py):

```python
input_dynamic = getattr_chain(scheme, "input_activations.dynamic", True)
output_dynamic = getattr_chain(scheme, "output_activations.dynamic", True)
if input_dynamic is not True or output_dynamic is not True:
    raise ValueError(
        "Model Free PTQ cannot calibrate activations. Please use `oneshot` instead."
    )
```

`NVFP4` sets `input_activations.dynamic = DynamicType.LOCAL`, so it trips this.
`NVFP4A16` (weight-only) and `FP8_DYNAMIC` (`dynamic=True`) both pass. That constraint is
not a compromise so much as an alignment: it lands you on exactly the split NVIDIA ships.

vLLM serves it. From
[`compressed_tensors.py`](https://github.com/vllm-project/vllm/blob/v0.28.0/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py)
at v0.28.0:

```python
if self._is_nvfp4_format(weight_quant):
    if input_quant is None:
        return CompressedTensorsW4A4Fp4(use_a16=True)
    ...
    return CompressedTensorsW4A4Fp4()
```

Same scheme class, `use_a16=True`, and `CompressedTensorsW4A4Fp4.get_min_capability()`
returns **75** — no Blackwell gate, sm_120 nowhere near the floor.

**Weight bytes are unchanged** (4-bit either way), so the ~21.8 GiB footprint and the
bandwidth-bound decode should both hold. What changes is the kernel: W4A16 dequantizes to
BF16 for the GEMM rather than using sm_120's native FP4 tensor cores. Decode should cost
little; prefill will cost more. **This is the one number here that is predicted rather
than measured** — see §9.

### Recipe A — `oneshot` + MTP re-attach (bit-matches what we serve today)

Use this if Recipe B's kernel path measures materially slower. It reproduces the config
decoded in §2 exactly, inheriting that recipe's measured 124.9 tok/s, 58.8% MTP
acceptance, and passed coherence check.

The shape is
[`examples/quantization_non_uniform/quantization_nvfp4_fp8.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_non_uniform/quantization_nvfp4_fp8.py),
whose own closing comment confirms the artifact:

> The model produced is compressed using two different compressors with two different
> formats: nvfp4-pack-quantized and float-quantized. The presence of multiple
> compressors is indicated by the `mixed-precision` format in the model's config.json.

with multimodal loading from
[`examples/quantization_w4a4_fp4/qwen3_5_example.py`](https://github.com/vllm-project/llm-compressor/blob/main/examples/quantization_w4a4_fp4/qwen3_5_example.py)
— which is for the MoE sibling, so drop its `mlp.gate` / `shared_expert_gate` ignores and
`moe_calibrate_all_experts`, and swap the model class. `sequential_targets` is
**`Qwen3_5DecoderLayer`** (`modeling_qwen3_5.py` line 817).

```python
# llmcompressor==0.13.0.  NOTE: requires transformers >= v5
from compressed_tensors.quantization.quant_scheme import FP8_DYNAMIC, NVFP4
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.utils import load_context

MERGED = "/mnt/tank/outputs/<jobId>/merged-clean"
with load_context(Qwen3_5ForConditionalGeneration):
    model = Qwen3_5ForConditionalGeneration.from_pretrained(MERGED, device_map=None)
processor = AutoProcessor.from_pretrained(MERGED)

g0 = dict(FP8_DYNAMIC); g0["targets"] = [
    r"re:.*self_attn\.(q|k|v|o)_proj$",
    r"re:.*linear_attn\.(in_proj_qkv|in_proj_z|out_proj)$",
    r"re:.*lm_head",
    r"re:.*layers\.(56|57|58|59|60|61|62|63)\.mlp\.(gate|up|down)_proj$",
]
g1 = dict(NVFP4);       g1["targets"] = [r"re:.*mlp\.(gate|up|down)_proj$"]

oneshot(
    model=model,
    dataset=ds,                       # ultrachat_200k train_sft, 512 samples, seq 2048
    recipe=QuantizationModifier(
        config_groups={"group_0": g0, "group_1": g1},
        ignore=["re:.*visual.*", r"re:.*linear_attn\.(norm|in_proj_a|in_proj_b)$"],
    ),
    sequential_targets=["Qwen3_5DecoderLayer"],
    data_collator=data_collator,
    max_seq_length=2048,
    num_calibration_samples=512,
)
model.save_pretrained(OUT, save_compressed=True)
processor.save_pretrained(OUT)
```

Calibration data: unsloth's public runner defaults to `HuggingFaceH4/ultrachat_200k`
`train_sft`, 512 samples, `max_seq_length` 2048, `shuffle(seed=42)`; llm-compressor's
own qwen3_5 example uses 256 samples at 4096. Their card notes their shipped mix also
includes unpublished coding/tool-calling data — so a bit-identical reproduction is not
possible, but that affects scale *values*, not structure. **Calibrating on your own
fine-tune's data distribution is arguably more correct anyway.**

**Then re-attach the MTP head. This is the build work.** ~40 lines:

1. Read the 15 `mtp.*` tensors from `Qwen/Qwen3.8-27B`'s
   `model-00018-of-00018.safetensors` (all 15 live in that one shard).
2. `save_file(...)` them to `model_mtp.safetensors` in the output directory.
3. Add the 15 entries to the output's `model.safetensors.index.json` `weight_map`
   pointing at that file, and add their bytes to `metadata.total_size`.

That reproduces unsloth's, RedHatAI's and sakamakismile's layout. It belongs next to
`merge_qwen3moe.py` in the recipes repo — call it `attach_mtp.py` — and should be
**unconditional, not an option**, because the failure it prevents is silent.

### Recipe C — uniform NVFP4. Documented so nobody reaches for it

`sakamakismile/Qwen3.8-27B-MTP-NVFP4` (93k downloads) ships a much simpler recipe:

```yaml
default_stage:
  default_modifiers:
    QuantizationModifier:
      targets: [Linear]
      ignore: [lm_head, 're:.*visual.*', 're:.*conv1d.*', 're:.*mtp.*']
      scheme: NVFP4
```

It is tempting: four lines, W4A4 everywhere, MTP handled. **Do not use it here.** `targets:
[Linear]` puts NVFP4 on `self_attn.q_proj`, which on this architecture carries a fused
attention-output gate — see §7. Neither NVIDIA nor RedHatAI nor unsloth does this, and
there is perplexity evidence that it is actively harmful. Its popularity is not
validation; nobody has published a quality comparison against it.

### unsloth's one-liner — the shortest path, and its two traps

unsloth ships a public API that does merge-then-quantize in one call
(`unsloth/save.py`, `_compressed_quantize.py`, in `unsloth==2026.8.22`):

```python
save_pretrained_merged(dir, tokenizer, save_method="nvfp4",
                       calibration_dataset=..., num_calibration_samples=512)
```

> "FP8 / FP4 compressed export for vLLM via llm-compressor… The LoRA is merged to 16bit
> at `save_directory`, then a quantized checkpoint is written to
> `save_directory + "-<fmt>"`."

Genuinely convenient, and it confirms the whole pipeline shape is a supported use case.
But two things make it unsuitable as-is:

1. **It builds a single uniform scheme** — `QuantizationModifier(targets="Linear",
   scheme=args.scheme, ignore=ignore)` — i.e. Recipe C, with the §7 problem.
2. **It drops MTP silently.** The code has a warning for this, but it is gated on
   `_has_mtp()`, which checks top-level `num_nextn_predict_layers` / `num_mtp_layers` /
   `mtp_num_layers` and a `model_type` containing `qwen3_next`/`mtp`. Qwen3.8 carries
   `text_config.mtp_num_hidden_layers` and `model_type: "qwen3_5"` — **none match, so the
   warning never fires**.

Worth knowing it exists; not worth using without the two fixes.

---

## 7. The gated-attention hazard — why FP8 on attention is not a style choice

Qwen3.8-27B's config carries **`attn_output_gate: true`** (verified in both the BF16 and
NVFP4 configs). On this architecture `q_proj` emits two concatenated roles: the Q half
feeds attention scores, and the gate half feeds a sigmoid that multiplies the attention
output. A quantizer sees one fused tensor and treats both halves alike.

modelopt issue #2091 argues NVFP4 on that gate half is destructive, with teacher-forced
perplexity on a matched corpus:

| model | attention | quantizer | PPL |
|---|---|---|---|
| Qwen3-14B-NVFP4 | dense | modelopt | **10.03** ✅ |
| Qwen3.6-27B-Text-NVFP4-MTP | hybrid + gate | modelopt | **65.13** ❌ |
| Qwen3.6-35B-A3B-NVFP4 | hybrid + gate | llm-compressor | 13.65 (4-bit twin: 6.55) ❌ |
| Ornith-1.0-35B-NVFP4 | hybrid + gate | llm-compressor | 16.16 (4-bit twin: 6.50) ❌ |

Localised per-layer against a 4-bit GGUF twin (both arms 4-bit, so not a
4-bit-vs-BF16 artifact): gated `full_attention` blocks add **+0.0156** divergence per
block; `linear_attention` blocks add **−0.0017**; a dense NVFP4 model measured the same
way adds **+0.0000**.

**Handle this evidence carefully.** The issue was **closed, withdrawn by its author**,
who had not verified the forward direction — that excluding the gate *repairs* an export.
No maintainer rebutted it either. So treat it as a strong hypothesis with suggestive
data, not a settled result.

What makes it actionable is that it does not stand alone. **NVIDIA, RedHatAI and unsloth
all independently keep the whole of `self_attn` at FP8 on this architecture**, across two
unrelated toolchains. Whatever the mechanism, the recipes that ship agree, and the
recipes that put NVFP4 on gated attention are the ones with bad perplexity numbers
attached. Recipes A and B both follow the shipped convention. Recipe C does not.

The Qwen3.6-27B row is the closest published analogue to what we would be building — same
hybrid architecture, with MTP, modelopt-produced — which is reason enough to run a
perplexity check against the BF16 source before trusting any output (§8).

---

## 8. On-load quantization, and the state of `quantize_fp8.py`

### On-load NVFP4: no, and it would not help

`vllm serve --quantization fp8` converts BF16 weights at load with no offline step; that
is how `recipes/qwen3.6-27b-base-lora-attn-mlp/inference-fp8.yaml` already serves merged
fine-tunes, and what the roadmap means by "FP8 deploys use vLLM's on-load
`--quantization fp8`".

**There is no NVFP4 equivalent, and there structurally cannot be a useful one here.**
NVFP4 in `compressed-tensors` is a *checkpoint* format: `weight_packed` (U8, two values
per byte), `weight_scale` (FP8 per group of 16), `weight_global_scale` (F32), and for
W4A4 a calibrated `input_global_scale`. vLLM's loader dispatches on `quantization_config`
already being present — `_is_nvfp4_format(weight_quant)` *reads* the config, it does not
synthesize one. Producing those tensors requires the observer pass llm-compressor
performs; W4A4 additionally requires calibration data a serving process does not have.

**And it is moot on this card anyway.** On-load quantization must load BF16 first — 51.7
GiB into 31.8 GiB. Even if on-load NVFP4 existed it could not run here. On-load FP8 is
out for two further independent reasons: FP8 is 28.7 GiB, leaving under 3 GiB for KV and
activations, and `vllm#51884` (blockwise FP8 dying in `process_weights_after_loading`,
"Unknown SF transformation") is filed against capability family 12 — this exact card
class.

**Offline quantization is not a workaround here. It is the only mechanism that fits.**

### Two bugs in `scripts/quantize_fp8.py`

Independent of the pin question, the existing script would not work on this model:

1. It calls `AutoModelForCausalLM.from_pretrained(...)`, which on a
   `Qwen3_5ForConditionalGeneration` checkpoint resolves to `Qwen3_5ForCausalLM` — the
   class that drops **both** `^mtp.*` **and** `^model.visual.*` (line 1658). The vision
   tower goes with the MTP head.
2. It passes `device_map="auto"`, which fights sequential onloading rather than using it.

Whoever revisits #43 should treat it as a rewrite, not a dependency bump.

---

## 9. Quality cost of quantizing after merge

No first-party guidance says quantize-after-merge is worse than the alternative, and the
question is largely academic: the alternative (serve merged BF16) does not fit the card.

More usefully, **quantize-after-merge for this exact model is already common practice.**
Several published NVFP4 checkpoints are quantizations of merged/fine-tuned Qwen3.8-27B
derivatives — `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` (ships its `recipe.yaml`),
`sakamakismile/Huihui-Qwen3.8-27B-abliterated-NVFP4`,
`Blackfrost-AI/Qwen3.8-27B-ABLITERATED-NVFP4`,
`esatapedico/Qwen3.8-27B-Cold-Fusion-GAIN-V1.1-NVFP4-*`. The order of operations is not
novel.

The honest framing is that **the quantization tax and the fine-tune delta are confounded**
unless measured apart. Two cheap controls make them separable:

1. **Quantize the *unmerged* base with your own recipe** and compare against
   `unsloth/Qwen3.8-27B-NVFP4`. Same weights, same target, only your toolchain differs —
   zero fine-tune signal. This is also the natural first end-to-end test, since it needs
   no adapter at all.
2. **Evaluate merged BF16 on a Spark** (where it fits) against merged NVFP4 on the 5090.
   The difference is the quantization tax on *your* weights.

Given §7, a **perplexity check against the BF16 source** is worth the hour on any output.
The failure mode there is not a crash; it is a 2–6× perplexity regression that a
smoke-test conversation will not reveal.

Also worth a sanity check on every output, from modelopt's issue history but applicable
generally: **output size**. A BF16-sized "quantized" directory is the signature of a
silent quantization no-op.

Finally, a measurement trap that has already bitten twice on this model and will
invalidate any quality comparison that ignores it: **pin
`chat_template_kwargs={"reasoning_effort":"low"}`**. Both Qwen3.8 recipes record that the
default effort can think past the token cap and return a completely empty response
(`finish_reason="length"`, both `content` and `reasoning_content` empty). An empty reply
is indistinguishable from quantization damage.

---

## 10. What is *not* proven

Stated plainly, because the rest is strong enough that the gaps should not get lost.

| Claim | Status |
|---|---|
| llmcompressor 0.13.0 accepts transformers 5.x | **Proven** — PyPI metadata, verbatim pin |
| The served checkpoint is compressed-tensors; recipe recoverable | **Proven** — read from the local checkpoint |
| RedHatAI ships an identical config for this model | **Proven** — fetched and compared |
| transformers silently drops `mtp.*` | **Proven** — source, two named lines, plus llm-compressor's own example comment |
| MTP tensors are a byte-identical copy of the base | **Proven** — SHA-256 of raw tensor bytes, 5 of 15 sampled |
| MTP re-attach is standard practice | **Proven** — three published checkpoints do it; index layouts inspected |
| The merge script handles this architecture | **Proven** — in production for Qwen3.6-27B, identical config |
| NVIDIA/RedHatAI/unsloth agree on the module split | **Proven** — recipe files compared side by side |
| Quantize-after-merge is normal for this model | **Proven** — multiple published fine-tune NVFP4s |
| A 51.7 GiB model can be quantized on one GPU | **Documented** (70B on one A100); **not run here** |
| `oneshot` completes on `Qwen3_5ForConditionalGeneration` | **Not run.** Multimodal + hybrid-attention + MTP is an unusual combination |
| `model_free_ptq` output serves on sm_120 | **Path confirmed in vLLM source** (min capability 75); **not run** |
| Recipe B decode ≈ Recipe A decode | **Predicted, not measured.** W4A16 dequantizes to BF16 rather than using sm_120 FP4 tensor cores. **Measure before choosing.** |
| NVFP4 on gated `q_proj` hurts quality | **Suggestive, not settled.** The issue was withdrawn by its author and never rebutted. Convergent recipe evidence is what makes it actionable |
| No quality cliff from quantize-after-merge | **Assumed.** Needs the §9 controls |

Two things could still turn this into real build work: `oneshot`/`model_free_ptq` failing
on the hybrid+multimodal architecture in a way the ignore list cannot express, or Recipe
B's W4A16 kernel measuring badly enough on sm_120 to force Recipe A. Neither is likely;
both are cheap to find out.

**Suggested order.** Run Recipe B against the *unmerged* base first. It needs no new
code, no calibration data, no adapter, and no fine-tune — and it answers both open
questions (does the toolchain work on this architecture, and how fast is W4A16 on
sm_120) in one job, against a checkpoint whose correct behaviour is already measured on
this exact hardware. Only then point it at a merged model.

---

## 11. Sources

**Local checkpoints** (read directly — the strongest evidence here)
- `/mnt/tank/models/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/` — `config.json`,
  `model.safetensors.index.json`, safetensors headers of `model.safetensors` and
  `model_mtp.safetensors`
- `/mnt/tank/models/hub/models--Qwen--Qwen3.8-27B/snapshots/*/` — index and
  `model-00018-of-00018.safetensors` header + tensor bytes

**HuggingFace**
- [`unsloth/Qwen3.8-27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4) ·
  [`RedHatAI/Qwen3.8-27B-NVFP4`](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) ·
  [`sakamakismile/Qwen3.8-27B-MTP-NVFP4`](https://huggingface.co/sakamakismile/Qwen3.8-27B-MTP-NVFP4)
  (`recipe.yaml`, `model.safetensors.index.json`)

**PyPI release metadata**
- `https://pypi.org/pypi/llmcompressor/{0.10.0,0.12.1a20260716,0.13.0}/json`,
  `https://pypi.org/pypi/compressed-tensors/json`

**llm-compressor** (`vllm-project/llm-compressor`, `main`)
- `examples/quantization_non_uniform/quantization_nvfp4_fp8.py`
- `examples/quantization_w4a4_fp4/qwen3_5_example.py`
- `examples/model_free_ptq/{README.md,qwen3.5_int8.py,deepseek_r1_nvfp4_fp8_block.py}`
- `examples/multimodal_vision/qwen3_vl_example.py`
- `examples/big_models_with_sequential_onloading/{README.md,llama3.3_70b.py}`
- `src/llmcompressor/entrypoints/model_free/{validate,process,lifecycle}.py`

**compressed-tensors** (`vllm-project/compressed-tensors`, `main`)
- `src/compressed_tensors/quantization/quant_scheme.py` — `NVFP4`, `NVFP4A16`, `PRESET_SCHEMES`

**transformers** (`huggingface/transformers`, `main`)
- `src/transformers/models/qwen3_5/modeling_qwen3_5.py` — lines 817, 873–881, 1652–1658, 1749

**vLLM** (`vllm-project/vllm`, tag `v0.28.0`)
- `.../compressed_tensors/compressed_tensors.py` — NVFP4 dispatch
- `.../compressed_tensors/schemes/compressed_tensors_w4a4_nvfp4.py` — `get_min_capability() -> 75`

**NVIDIA Model-Optimizer** (`NVIDIA/Model-Optimizer`, formerly `TensorRT-Model-Optimizer`)
- `modelopt_recipes/huggingface/qwen3_5/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast{,.quant_cfg}.yaml`
- PR #2008 (single-GPU disk-offload PTQ, merged 2026-08-05) · issue #2160
  (`--low_memory_mode` silently broken, open) · issue #2091 (gated-attention NVFP4
  perplexity, closed/withdrawn) · PR #860, PR #1868, issue #750 (MTP left BF16) ·
  issue #1778 (transformers pin) · issue #1294 (no LoRA-on-NVFP4 training)

**unsloth** (`unslothai/unsloth`, `main`)
- `unsloth/_compressed_quantize.py`, `unsloth/save.py` (`save_pretrained_merged`,
  `save_method="nvfp4"`), `unsloth/models/mapper.py`

**Other**
- sglang issue #34895 (FP8 `lm_head` unloadable; fixed on `main` by PR #35228)

**This repo**
- `recipes/dgxrun/qwen3.8-27b-nvfp4-rtx.yaml`, `recipes/dgxrun/qwen3.8-27b-bf16.yaml`,
  `docs/qwen3.8-model-survey.md` (all on branch `qwen3.8-27b-and-staging-design`) ·
  `docs/ROADMAP.md`

**`dgx-manager-fine-tune-recipes`** (`/mnt/tank/src/github/dgx-manager-fine-tune-recipes`)
- `scripts/merge_qwen3moe.py`, `scripts/quantize_fp8.py`
- `recipes/qwen3.6-27b-base-lora-attn-mlp/{recipe.yaml,train.py,inference-fp8.yaml}`
