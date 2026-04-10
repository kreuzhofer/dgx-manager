# Fine-Tuning Gemma 4 on NVIDIA DGX Spark: A Practitioner's Guide

> What we learned building a complete LoRA fine-tuning → merge → deploy pipeline for Google's Gemma 4 models on DGX Spark hardware, and every bug we hit along the way.

---

## TL;DR

We fine-tuned Gemma 4 E2B, E4B, and 26B-A4B models on NVIDIA DGX Spark (GB10, 128GB unified memory) using LoRA with DeepSpeed ZeRO. The 26B model requires multi-node training across 2 DGX Spark nodes. After merging the LoRA adapter back into the base model, we serve it with vLLM. On a SQL generation benchmark, the E2B model improved from 4% to 22% exact-match accuracy after one epoch of fine-tuning.

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

**Gemma 4 E2B** (1 full epoch, single-node):
| | Accuracy (50 examples) |
|---|---|
| Base model | 4% |
| Fine-tuned | 22% |
| **Improvement** | **+18pp (5.5x)** |

Key lesson: the eval prompt format must exactly match the training format. Adding a system prompt or "Schema:"/"Question:" prefixes that weren't in the training data destroys accuracy.

**Gemma 4 26B-A4B-it** (training in progress):
- Multi-node across 2 DGX Spark nodes
- ZeRO-3 parameter partitioning
- Expected ~23 hours for 1 full epoch

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
