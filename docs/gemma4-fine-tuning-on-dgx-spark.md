# Fine-Tuning Gemma 4 on NVIDIA DGX Spark: A Practitioner's Guide

> What we learned building a complete LoRA fine-tuning → merge → deploy pipeline for Google's Gemma 4 models on DGX Spark hardware, and every bug we hit along the way.

---

## TL;DR

We fine-tuned Gemma 4 E2B, E4B, and 26B-A4B models on NVIDIA DGX Spark (GB10, 128GB unified memory) using LoRA with DeepSpeed ZeRO. The 26B model requires multi-node training across 2 DGX Spark nodes. After merging the LoRA adapter back into the base model, we serve it with vLLM. On a SQL generation benchmark (`b-mc2/sql-create-context`, 100 held-out examples, exact-match accuracy), the E4B model jumped from 23% to **90%** after one epoch and the E2B went from 4% to 22%. The 26B-A4B-it run *looked* healthy during training (monotonically decreasing eval loss) but came in **worse than base** at inference — root-caused to a MoE-targeting bug in PEFT's `target_modules` walk. With the fix (PEFT `target_parameters` reaching the fused expert tensors), the 26B-A4B-base hits 38% → **46%** in 50 steps, and surprisingly *peaks* there — see Results.

This post documents every workaround, patch, and architectural decision we made — because the official tooling doesn't cover any of this.

---

## The Hardware: DGX Spark and its Quirks

The NVIDIA DGX Spark uses the GB10 chip with **128GB unified memory** shared between CPU and GPU. This creates several unique challenges:

### No Separate VRAM and RAM

Unlike traditional GPU setups where you have 80GB VRAM + 256GB RAM, the DGX Spark shares all 128GB. This means:
- **CPU offloading is pointless** — DeepSpeed ZeRO's `offload_param` and `offload_optimizer` just shuffle data within the same memory pool
- **VRAM reporting is broken** — `nvidia-smi` reports `[N/A]` for many fields, and `pynvml.nvmlDeviceGetMemoryInfo` raises `NVMLError_NotSupported`
- **Memory planning requires /proc/meminfo** instead of GPU memory queries

### ARM Architecture (aarch64)

The GB10 is ARM-based, which means:
- Most ML Docker images are x86_64 only
- NVIDIA's `nvcr.io/nvidia/pytorch:25.11-py3` is the go-to base image (ARM-compatible)
- Tools like Triton need ARM-specific builds (relevant for Unsloth)

### InfiniBand for Multi-Node

DGX Spark clusters connect via InfiniBand (RDMA). For multi-node training, you need to explicitly set:
```bash
NCCL_SOCKET_IFNAME=enp1s0f0np0
GLOO_SOCKET_IFNAME=enp1s0f0np0
NCCL_IB_HCA=rocep1s0f0
NCCL_IB_DISABLE=0
```
Without these, NCCL falls back to TCP over Ethernet — dramatically slower for parameter synchronization.

---

## The Model: Gemma 4 and Why It's Different

Google developed Gemma 4 in JAX and released PyTorch-compatible weights via HuggingFace. But the PyTorch integration has rough edges that Google didn't polish because they don't use PyTorch internally.

### Gemma4ClippableLinear: The Layer That Breaks Everything

Gemma 4's vision and audio towers use a custom `Gemma4ClippableLinear` layer that wraps `nn.Linear` with optional input/output clamping for numerical stability:

```python
class Gemma4ClippableLinear(nn.Module):  # NOT nn.Linear!
    def __init__(self, config, in_features, out_features):
        self.linear = nn.Linear(in_features, out_features, bias=False)
        # ... clipping buffers ...
```

This causes three cascading problems:

**Problem 1: PEFT doesn't recognize it.** PEFT's LoRA implementation checks `isinstance(target, nn.Linear)` — `Gemma4ClippableLinear` inherits from `nn.Module`, so it fails with "Target module not supported."

**Problem 2: Weight key names.** Because the linear layer is nested (`self.linear`), weights are stored as `module.linear.weight` instead of `module.weight`. After merging and saving, `save_pretrained()` may flatten these keys, creating a mismatch with what vLLM expects.

**Problem 3: Multi-format saves.** Some keys need `.linear.weight` (ClippableLinear modules in towers), others need plain `.weight` (language model), and some are missing entirely (clipping buffers, k_eq_v shared weights).

### Our Solution: Patch PEFT, Fix Keys After Save

Instead of modifying the model architecture (which breaks weight names), we patch PEFT's dispatch to recognize ClippableLinear:

```python
from peft.tuners.lora import model as lora_model
from peft.tuners.lora.layer import Linear as LoraLinear

_orig_dispatch = lora_model.dispatch_default

def _patched_dispatch(target, adapter_name, lora_config, **kwargs):
    if (hasattr(target, 'linear') and isinstance(target.linear, nn.Linear)
            and not isinstance(target, nn.Linear)):
        kwargs.update(lora_config.loftq_config)
        return LoraLinear(target.linear, adapter_name, **kwargs)
    return _orig_dispatch(target, adapter_name, lora_config, **kwargs)

lora_model.dispatch_default = _patched_dispatch
```

After merging, we remap weight keys by comparing against the original model's checkpoint:

```python
# Load original model's key names
orig_tensors = load_original_safetensors(base_model)

# Fix flattened keys and copy missing ones
for key, tensor in saved_tensors.items():
    linear_key = key.replace(".weight", ".linear.weight")
    if key not in orig_keys and linear_key in orig_keys:
        remapped[linear_key] = tensor  # Rename
    else:
        remapped[key] = tensor

# Copy ALL missing keys from original (clipping buffers, shared weights, etc.)
for orig_key in orig_keys:
    if orig_key not in remapped:
        remapped[orig_key] = orig_tensors[orig_key]
```

### The use_cache Bug

Setting `use_cache=False` (which `gradient_checkpointing_enable()` does by default) **corrupts Gemma 4's attention computation**, producing garbage logits. You must force it back:

```python
model.gradient_checkpointing_enable(
    gradient_checkpointing_kwargs={"use_reentrant": False})
if hasattr(model.config, "use_cache"):
    model.config.use_cache = True  # Critical for Gemma 4!
```

### Missing Chat Template

The Gemma 4 tokenizer ships without a `chat_template`. You need to provide one:

```python
tokenizer.chat_template = (
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}<start_of_turn>user\n{{ message['content'] }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}<start_of_turn>model\n{{ message['content'] }}<end_of_turn>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)
```

### Multimodal Token Type IDs

Even for text-only fine-tuning, Gemma 4 requires `mm_token_type_ids` in training inputs. Without it:
```
ValueError: `mm_token_type_ids` is required as a model input when training
```

The fix: add zeros for all text tokens, and use a custom data collator that preserves these fields (SFTTrainer's default collator strips them):

```python
tokens["mm_token_type_ids"] = [0] * len(tokens["input_ids"])
tokens["token_type_ids"] = [0] * len(tokens["input_ids"])

# Plus: remove_unused_columns=False in SFTConfig
# Plus: custom Gemma4DataCollator that preserves mm_token_type_ids
```

---

## Training Architecture

### Single-Node (E2B, E4B)

For smaller models (2-8B effective params), single-node training with DeepSpeed ZeRO-2 works:
- **E2B** (2B effective): ~24GB for training, comfortable on one node
- **E4B** (4.5B effective): ~32GB for training, fits on one node

### Multi-Node (26B-A4B)

The 26B model (MoE, ~26B total / 4B effective) needs ~100GB+ for LoRA training:
- Model weights: ~47GB in bf16
- Optimizer states: ~47GB (AdamW 2x)
- Gradients + activations: ~20GB

This exceeds 128GB on one DGX Spark node. We use **2-node training** with DeepSpeed ZeRO-3:
- Each node holds half the model parameters
- torchrun with static rendezvous coordinates the nodes
- Head node orchestrates workers via SSH

**Critical ZeRO-3 gotcha:** All ranks must call `trainer.save_model()`, not just rank 0. With ZeRO-3, parameters are distributed — rank 0 needs rank 1 to participate in the all-gather for saving. If only rank 0 calls save, it deadlocks waiting for parameters that rank 1 has already discarded.

### DGX Spark-Specific Patches

We maintain a shared Python library (`lib/`) with all DGX Spark workarounds:
- `patch_pynvml()` — monkey-patches `nvmlDeviceGetMemoryInfo` to return system RAM
- `patch_safetensors_cache()` — flushes NFS page cache after each safetensors shard load (prevents 60GB cache buildup)
- `patch_peft_for_clippable_linear()` — teaches PEFT to handle ClippableLinear
- `fix_clippable_linear_keys()` — remaps weight keys after merge for vLLM compatibility
- `flush_page_cache()` — drops /proc page cache (needs root + writable /proc)
- `Tee` logger — writes stdout to both pipe and `train.log` file for log persistence

---

## Production Pitfalls (Lessons from a 44h Training Run)

The "happy path" for multi-day training runs has several sharp edges. Here's what we hit and fixed.

### The Eval Collator Crash at the Finish Line

After 44 hours of successful training, our first full 26B run crashed during the final end-of-epoch evaluation:

```
ValueError: Unable to create tensor... Perhaps your features
(`labels` in this case) have excessive nesting
```

The root cause: our `format_example` function was pre-computing `tokens["labels"] = tokens["input_ids"].copy()` during tokenization. Training worked fine because `per_device_train_batch_size=1` — each batch has one sequence, no stacking needed. Evaluation uses `per_device_eval_batch_size=8` by default, and `tokenizer.pad()` doesn't pad the `labels` field, so stacking sequences of different lengths into a tensor blows up.

**Fix:** don't pre-compute labels. `DataCollatorForLanguageModeling(mlm=False)` derives labels from `input_ids` *after* padding, which is the correct order.

### The Invisible HF Datasets Cache

After removing the pre-computed labels, the bug still reproduced. Files on disk showed the fix, but the tokenized dataset was being pulled from a stale HF datasets cache dated *before* the fix. HuggingFace's dataset fingerprint hashing misses changes behind `lambda` in `.map()` calls.

**Fix:** pass `load_from_cache_file=False` to `raw.map()`. Costs ~1-2 min of re-tokenization per run; saves you from debugging ghost bugs.

### DeepSpeed Resume is Not "Just Works"

`save_only_model=True` in `SFTConfig` keeps checkpoints small (~60MB LoRA adapter only). Fast saves, great for frequent checkpointing. But attempting to resume with `trainer.train(resume_from_checkpoint=True)` explodes:

```
ValueError: Can't find a valid checkpoint at .../checkpoint-500
```

DeepSpeed's resume path expects `zero_pp_rank_*` optimizer state files, which `save_only_model=True` skipped. You need `save_only_model=False` if you want resumability. With LoRA + ZeRO-3, saves are still small (~430MB per checkpoint) because DeepSpeed only tracks the trainable params' optimizer state, not the frozen base model.

### Silent Multi-Node Hangs

When rank 1 (worker) crashes during a collective operation, rank 0 (head) doesn't crash — it hangs waiting forever inside NCCL. We set `TORCH_NCCL_ASYNC_ERROR_HANDLING=1` to disable NCCL's 10-min timeout (needed for 26B ZeRO-3 all-gather), which also disables rank-failure detection.

Symptoms: training log stops updating, GPU stays at 96% utilization on head, container never exits, agent never fires its "training exited" event. Job stuck in "running" forever, GPU pinned, can't start new work until something intervenes.

**Fix (in our agent):** a 30-second watchdog that polls each worker container's state via SSH. If a worker container transitions out of `running` while the head is still alive, force-clean the head container and mark the job failed. Turns a silent 44h zombie into a clear error within 30 seconds.

### Mixed Rank Output in One Log File

Python's `sys.stdout = Tee(file)` captures Python `print()` calls but misses C/C++ writes from PyTorch/DeepSpeed loaders (they write directly to fd 1/2). On top of that, if every rank's Python Tee points at the same shared NFS `train.log`, output interleaves unpredictably.

**Fix:** shell-level `tee` in the launch script, per-rank:

```bash
if [ "$NODE_RANK" -eq 0 ]; then
    RANK_LOG="$OUTPUT_DIR/train.log"
else
    RANK_LOG="$OUTPUT_DIR/train-rank${NODE_RANK}.log"
fi
exec > >(tee -a "$RANK_LOG") 2>&1
```

Captures *everything* (Python, C extensions, NCCL debug), keeps ranks separate, survives resumes via append mode.

### Metric Deduplication

Each training step emits multiple log patterns that the agent parses: `[TRAIN] step=N/M loss=X`, `{'loss': '13.4', ...}` dict, and tqdm bars. Our server was naïvely `prisma.trainingMetric.create()`-ing on every one of them, so the loss curve showed 3× the data points for each step. Upsert on `(jobId, step)` with a unique constraint makes this a non-issue.

---

## Serving with vLLM

After merging the LoRA adapter, we generate a vLLM recipe and deploy:

### Container Selection

Gemma 4 requires `transformers >= 5.x` (model type `gemma4` not in 4.x). The standard `vllm-node` container has transformers 4.57. We use `vllm-node-tf5` which includes transformers 5.

The training recipe specifies which container to use for deployment:
```yaml
deploy:
  container: vllm-node-tf5
  gpu_memory_utilization: 0.3
  max_model_len: 4096
```

### GPU Memory Utilization

vLLM's `gpu_memory_utilization` is a **reservation**, not actual usage. It pre-allocates that percentage of total GPU memory at startup for KV cache. For a 15GB E4B model, 0.85 (102GB) is overkill. We use 0.3 (36GB) for small models, 0.5 for the 26B.

---

## Results

### SQL Generation Benchmark

Dataset: [b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context) (78K examples)

**Gemma 4 E2B** (1 full epoch, single-node, 50 examples):
| | Accuracy |
|---|---|
| Base model | 4% |
| Fine-tuned | 22% |
| **Improvement** | **+18pp (5.5×)** |

**Gemma 4 E4B** (1 full epoch, single-node, 100 examples):
| | Accuracy |
|---|---|
| Base model | 23% |
| Fine-tuned | **90%** |
| **Improvement** | **+67pp (3.9×)** |

E4B is the sweet spot for this dataset: the base model already has meaningful SQL awareness (23%), and one epoch of LoRA on `b-mc2/sql-create-context` pushes it to near-perfect exact-match accuracy. E2B's lift is proportionally larger (5.5×) but its absolute ceiling is much lower.

Key lesson: the eval prompt format must exactly match the training format. Adding a system prompt or "Schema:"/"Question:" prefixes that weren't in the training data destroys accuracy.

**Gemma 4 26B-A4B-it** (1 full epoch, 2 DGX Spark nodes, ~46h): **SQL eval came in worse than base**, and the reason wasn't the training run — it was the LoRA config.

Eval loss decreased monotonically throughout training (step 500 = 2.14 → step 1000 = 2.02 → step 2000 = 1.96 → step 3500 = 1.93), so from the training dashboard everything looked healthy. The regression only showed up at inference time.

**Gemma 4 26B-A4B-base** (with the MoE-targeting fix from the next section, 100 examples). Each row is a separate run, identical config except `max_steps`:

| Steps | Base | Tuned | Improvement |
|---:|---:|---:|---:|
| 50 | 38% | **46%** | **+8pp** |
| 50 (rerun) | 38% | 46% | +8pp |
| 500 | 38% | 42% | +4pp |
| 1 epoch (~9.3K steps) | 38% | 40% | +2pp |

The peak is at 50 steps, then accuracy *decays* with more training. The loss curves are still monotonically decreasing — eval loss at the full epoch is the lowest we've seen — so the model is genuinely learning the data distribution; it's just learning past the point where exact-match SQL phrasing matches the test set's specific style. We didn't expect to see this with LoRA (parameters are frozen, only adapters update), so the takeaway is: train loss / eval loss alone aren't a sufficient signal — task-specific eval needs to run at multiple checkpoints, not just at the end.

Net result: the `target_parameters` fix takes 26B-A4B-base from "broken" (worse than base, IT model) to "working" (+8pp at 50 steps). 50 steps is the right operating point for this task on this model.

### The bug: `target_modules` never hit the MoE path

Diagnosis was fully static — we inspected the saved adapter's safetensors header and compared the wrapped module paths against the base model's `model.safetensors.index.json`. Summary:

| | Count |
|---|---|
| Modules wrapped by the adapter | 394 |
| ├─ Language layers × 7 projections (`q/k/v/o/gate/up/down_proj`) | 30 × 7 = 210 |
| └─ Vision-tower layers × 7 projections | 27 × 7 = 184 |
| MoE expert / router modules wrapped | **0** |

Gemma 4 26B-A4B uses a *shared-expert* pattern: each of the 30 language layers has **both** a dense MLP (`mlp.{gate,up,down}_proj`) and an MoE block next to it:

```
model.language_model.layers.N.mlp.{gate,up,down}_proj.weight   # dense shared expert
model.language_model.layers.N.experts.gate_up_proj             # fused 3D: [num_experts, 2*inter, hidden]
model.language_model.layers.N.experts.down_proj                # fused 3D
model.language_model.layers.N.router.proj.weight               # router gate
model.language_model.layers.N.router.{scale, per_expert_scale}
```

PEFT's LoRA target matcher is `module_name.endswith(f".{target}")` plus "is this an `nn.Linear`?". That rejects every MoE module:

- `experts.gate_up_proj` / `experts.down_proj` are **fused 3D tensors**, not `nn.Linear`, and the name `gate_up_proj` doesn't end with `.gate_proj` or `.up_proj` anyway.
- `router.proj.weight` ends with `.proj`, not in our target list.

So the MoE path — which carries the bulk of the 26B's parameters and most of the active compute on GB10 — saw **zero gradient updates**. We only adapted attention, the dense shared-expert MLP, and (wastefully, for a text task) the entire 27-layer vision tower.

### Why that makes results *worse than base*

If the MoE experts had simply stayed untouched, you'd expect accuracy roughly equal to base. But the dense path *did* shift toward SQL output style, which changes the activations flowing into the router. A router that was calibrated on the original distribution now selects different experts — experts whose weights were trained on a different input regime. The inputs into the experts and the outputs coming out no longer match what the rest of the model expects. The downstream mismatch compounds across 30 layers and you get answers that are *less* coherent than the base model's.

On E2B/E4B this failure mode doesn't exist: those are fully dense, LoRA covers everything, no routing to mis-align.

### Sanity checks that ruled out easier explanations

- **Prompt format parity.** Training (`lib/dataset.py:67-88`) and eval (`scripts/evaluate.py:66-72,180`) both use the same single-user-turn `{context}\n\n{question}` via `apply_chat_template`. Identical.
- **Merge key remap.** `fix_clippable_linear_keys()` handles ClippableLinear's flattened `.linear.weight` keys correctly, and the "copy any missing key from base" fallback did not fire on MoE keys for this run (because we never saved MoE deltas to begin with). The merge isn't the culprit — but the fallback is fragile enough that we're going to add per-overwrite logging so a silent expert-key mismatch can't hide in a future run.

### What full fine-tuning would cost (and why we're not doing it next)

For the record, full FT of 26B-A4B in the obvious configuration (bf16 weights + grads, AdamW fp32 states, fp32 master weights for mixed precision):

| Component | Size |
|---|---|
| Weights (bf16) | ~52 GB |
| Gradients (bf16) | ~52 GB |
| AdamW states (fp32, 2 moments × 4 B × 26 B) | ~208 GB |
| FP32 master weights | ~104 GB |
| **Subtotal** | **~416 GB** |
| Activations (seq 2048, bs 1, grad checkpointing) | ~15–30 GB per rank |

"A4B" (active 4B) is a compute optimization — the memory class is still 26B, since you hold and update every parameter. Sharding across DGX Sparks at ~100 GB usable per node (after OS / NFS page cache / container overhead) needs **4–5 Sparks** with ZeRO-3, or **3–4 Sparks** with 8-bit AdamW. The current 2-Spark LoRA setup cannot hold full FT.

The cheaper, more targeted fix is a proper MoE-aware LoRA: custom PEFT dispatch that wraps `experts.gate_up_proj` and `experts.down_proj` as per-expert low-rank deltas on the fused 3D tensors, while leaving the router frozen. This is the next step; full FT is the fallback if MoE-LoRA also disappoints.

### The fix: PEFT `target_parameters` (no custom dispatch needed)

PEFT 0.16+ added `target_parameters=` to `LoraConfig`, exactly for this case. It bypasses the `nn.Linear`-or-bust `target_modules` walk and applies LoRA to fused `nn.Parameter` tensors directly. Two-line change to the recipe:

```python
attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]      # dense path (unchanged)
moe_param_targets = ["experts.gate_up_proj",              # NEW — fused 3D
                     "experts.down_proj"]                  # NEW — fused 3D

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16,
    target_modules=attn_targets,
    target_parameters=moe_param_targets,                  # ← reaches the experts
    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))

# Freeze the router — fine-tuning routing weights destabilizes which experts
# see which tokens, creating a feedback loop with the LoRA updates.
for name, p in model.named_parameters():
    if ".gate.weight" in name or "router" in name:
        p.requires_grad = False
```

Trainable jumps from ~30M (0.12%) to ~505M (1.9%) — the experts are now in the LoRA. Combined with `target_modules` covering the attention + dense MLP, this gives full coverage of the model's compute path, the router stays frozen so token→expert routing remains stable, and the IT-vs-base mismatch we hit earlier is sidestepped by training against the BASE model directly. Per-checkpoint SQL eval results are in the Results section above.

The bug was not in PEFT or in our code — it's a hidden requirement of the `target_modules` API: it silently does nothing for `nn.Parameter` tensors. PEFT only flags this when *all* targets miss; partial misses (attention hits, MoE misses) look successful from the trainable-parameter count alone unless you compare against the model's expected total. We added a hard sanity check at training start: if `trainable% < 0.5%` we abort with an explanation. Cheap insurance.

---

## What Google Could Have Done Better

1. **Make ClippableLinear inherit from nn.Linear** — would fix PEFT, weight key naming, and save/load compatibility in one change
2. **Include chat_template in the tokenizer** — basic oversight
3. **Don't break use_cache=False** — this is a standard PyTorch training pattern
4. **Document mm_token_type_ids requirement** — not mentioned in the model card
5. **Provide PyTorch fine-tuning examples** — they only provide JAX code, leaving the community to figure out PyTorch
6. **Test save_pretrained round-trip** — the weight key flattening issue would have been caught immediately

---

## Tools & Infrastructure

All code is open source:
- **DGX Manager**: Full-stack cluster management with fine-tuning UI, real-time loss curves, merge & deploy pipeline
- **Training Recipes**: Recipe-based system with shared `lib/` for all DGX Spark patches
- **Evaluation**: SQL exact-match comparison script with chart generation

The training UI shows:
- Phase-aware progress (container → downloading → loading → tokenizing → training → eval → saving)
- Live loss curve visualization persisted to database
- Real-time log streaming via SSE
- Merge and deploy buttons on completed jobs

---

## Appendix: Model-Specific Notes

### Gemma 4 E2B (google/gemma-4-e2b)
- 5B total params, ~2B effective
- Single-node LoRA with DeepSpeed ZeRO-2
- ~24GB VRAM for training
- Fast: 3 steps in ~5 min (including load)

### Gemma 4 E4B (google/gemma-4-e4b)
- 8B total params, ~4.5B effective
- Single-node LoRA with DeepSpeed ZeRO-2
- ~32GB VRAM for training
- 3 steps in ~8 min

### Gemma 4 26B-A4B-it (google/gemma-4-26B-A4B-it)
- 26B total params, ~4B effective (MoE)
- Multi-node (2x DGX Spark) with DeepSpeed ZeRO-3
- ~50-60GB per node for training
- 3 steps in ~8 min (ZeRO-3 overhead)
- Weight save requires all ranks (ZeRO-3 all-gather)

### Unsloth on DGX Spark
- Official `dgxspark` Docker image exists but ships with transformers 4.57 (no Gemma 4 support)
- Works for Llama 3.1 8B (tested and verified)
- Gemma 4 needs transformers 5.x which is incompatible with Unsloth 2026.1.4
- FlashAttention head dimension limit (>256) blocks Gemma 4 even with upgraded transformers

---

*Written April 2026. Based on hands-on experience fine-tuning on a 3-node DGX Spark cluster.*
*Training recipes and infrastructure: [dgx-manager](https://github.com/kreuzhofer/dgx-manager) + [dgx-manager-fine-tune-recipes](https://github.com/kreuzhofer/dgx-manager-fine-tune-recipes)*
