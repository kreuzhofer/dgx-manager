# VLM Alignment Drift Under a Text-Heavy LoRA SFT Run

> How much does a text-dominant LoRA SFT run degrade `Qwen/Qwen3.8-27B`'s vision–language alignment, and what proportion of multimodal examples prevents it?

Research for [issue #47](https://github.com/kreuzhofer/dgx-manager/issues/47). Feeds the LoRA-config and dataset-composition tickets.

Companion reading: [Fine-Tuning Qwen 3.6-27B on NVIDIA DGX Spark](../qwen3.6-27b-fine-tuning-on-dgx-spark.md) — **note that this document corrects one claim in it**, see [The premise is wrong](#the-premise-is-wrong-the-vit-was-never-being-adapted).

---

## TL;DR — the two decisions

**(a) Multimodal replay ratio: aim for a floor of ~10% of training examples containing at least one image. Do not spend effort tuning the exact value.**

Every published dose-response curve is *flat* between roughly 10% and 50% and *falls off a cliff at zero*. The only direct measurement of a vision-poor instruction mix ([MLAN, Table 5](#mlan--the-only-direct-dose-response-curve)) puts the vision average at **56.2 with 12.5% image data and 49.5 with 0%** — a 6.7 pp cliff that opens entirely between "a little" and "none". Anything in 12.5%–87.5% scored 54.8–57.0, i.e. indistinguishable.

So: **measure the natural image-bearing fraction of the chat3d trajectories first.** If it is already above ~10% of examples, ship it — no upweighting is justified by the evidence. If it is near zero, upweight to ~10%. Confidence: **medium** — this is an extrapolation across a setup gap, and I flag exactly where below.

**(b) Vision-tower LoRA targeting: keep the ViT frozen. Do not "fix" the recipe to reach it.**

This turns out not to be a judgement call. **On `Qwen3.8-27B` the recipe's `target_modules=["q_proj","k_proj","v_proj","o_proj"]` cannot match the vision tower at all** — its attention is fused as `attn.qkv` / `attn.proj`. The ViT has been frozen the whole time, and the 3.6 recipe comment claiming otherwise is factually wrong. That is the *correct* configuration, so the action is to make the freeze explicit and assert on it, not to extend the target list. Confidence: **high** — verified from the checkpoint's own tensor index and confirmed by exact parameter arithmetic.

**One thing worth more attention than either of the above:** the measured evidence says **learning rate and trainable-parameter count dominate the data mix by an order of magnitude**. At identical target-task accuracy, one paper moves out-of-distribution damage from **−33.64 pp to −1.51 pp with a single 10× LR cut**, and LoRA r=8 @ 1e-4 lands at **−2.97 pp**. Getting the LR right matters more than getting the ratio right.

---

## The premise is wrong: the ViT was never being adapted

The issue asks whether attaching LoRA to the vision tower's `q_proj/k_proj/v_proj/o_proj` is the right call for a mixed mix. It is not the right call, but not for the reason expected — **those modules do not exist on this model.**

I pulled the checkpoint's own tensor index. All 333 `model.visual.*` tensors, grouped by shape of name:

```
27  model.visual.blocks.N.attn.qkv.{weight,bias}      <- FUSED qkv, not q_proj/k_proj/v_proj
27  model.visual.blocks.N.attn.proj.{weight,bias}     <- named proj, not o_proj
27  model.visual.blocks.N.mlp.linear_fc{1,2}.{weight,bias}
27  model.visual.blocks.N.norm{1,2}.{weight,bias}
 1  model.visual.merger.{norm,linear_fc1,linear_fc2}.{weight,bias}
 1  model.visual.patch_embed.proj.{weight,bias}
 1  model.visual.pos_embed.weight
```

Source: [`Qwen/Qwen3.8-27B` `model.safetensors.index.json`](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/model.safetensors.index.json) (333 `model.visual.*` entries, matching the count in the issue exactly). Corroborated in the transformers source: [`modeling_qwen3_5.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py) defines `Qwen3_5VisionAttention` with `self.qkv = nn.Linear(dim, dim*3)` and `self.proj = nn.Linear(dim, dim)`.

PEFT matches `target_modules` by **name suffix**, so `q_proj` / `k_proj` / `v_proj` / `o_proj` match nothing under `model.visual.*`.

**Confirmed independently by arithmetic.** The 3.6 write-up reports `10,485,760` trainable parameters at `lora_r=16`. From [`config.json`](https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json): `hidden_size=5120`, `num_attention_heads=24`, `head_dim=256`, `num_key_value_heads=4`, and `attn_output_gate=true` (which doubles the `q_proj` output — the modeling code chunks it into query and gate). Per full-attention layer, at `r=16`:

| module | in → out | LoRA params `r·(in+out)` |
|---|---|---:|
| `q_proj` | 5120 → 12288 | 278,528 |
| `k_proj` | 5120 → 1024 | 98,304 |
| `v_proj` | 5120 → 1024 | 98,304 |
| `o_proj` | 6144 → 5120 | 180,224 |
| | **per layer** | **655,360** |

`655,360 × 16 full-attention layers = 10,485,760` — **exactly** the reported figure, with nothing left over for the vision tower or the MTP head. The tensor index confirms `q_proj` appears on exactly 17 modules: LM layers `[3, 7, 11, …, 63]` plus `mtp.layers.0` (which the recipe's `frozen` loop catches).

**Consequence.** Sub-question 3 is largely moot as a live risk, and becomes a documentation-and-guardrail task instead:

1. Correct the comment in `recipes/qwen3.6-27b-base-lora/train.py:39` and the corresponding paragraph in [`docs/qwen3.6-27b-fine-tuning-on-dgx-spark.md`](../qwen3.6-27b-fine-tuning-on-dgx-spark.md) — the vision tower is **not** "kept multimodal-capable"; it is untouched, and no LoRA capacity is being wasted on it.
2. Add an explicit assertion that zero adapters land under `model.visual.*`. The live hazard is a future switch to `target_modules="all-linear"`, which **would** attach to `attn.qkv`, `attn.proj`, `mlp.linear_fc1/2` and the merger — all `nn.Linear`. Frameworks guard against this by default; a hand-rolled PEFT config does not.
3. Keep the trainable-parameter count check. It is already the tripwire that would have caught this.

---

## The mental model: confirmed, then sharpened

The issue proposes: the ViT does not drift (it is not in the forward pass, and a zero-init LoRA `B` contributes nothing); what drifts is the **language model moving away from the visual embedding space it learned to read**.

**Both halves of the mechanism are confirmed from primary source.** The conclusion is right, but the literature sharpens *what* the drift actually is, and that changes the mitigation.

### The vision tower provably receives zero gradient on a text-only batch

From [`modeling_qwen3_5.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py), `Qwen3_5Model.forward`:

```python
if inputs_embeds is None:
    inputs_embeds = self.get_input_embeddings()(input_ids)

if pixel_values is not None:
    image_outputs = self.get_image_features(pixel_values, image_grid_thw, ...)
    image_embeds = image_outputs.pooler_output
    ...
    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
```

`self.visual` is reachable **only** through `get_image_features` / `get_video_features`, both inside those guards. With `pixel_values=None` the tower is not in the graph and cannot receive gradient — regardless of what adapters are attached.

### A zero-init LoRA adapter is an exact no-op

From [PEFT `src/peft/tuners/lora/layer.py`](https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py), `reset_lora_parameters`:

```python
# initialize A the same way as the default for nn.Linear and B to zero
nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
...
nn.init.zeros_(self.lora_B[adapter_name].weight)
```

`B = 0` makes `B·A = 0` exactly, so an attached-but-never-updated adapter changes nothing at forward time and merges to a zero delta. Note this also survives weight decay: decay can only shrink `A`, and `B·A` stays exactly zero either way.

### The alignment surface on this model is unusually small — DeepStack is off

Qwen3-VL injects multi-level ViT features additively into the **first three LLM layers** ("DeepStack"), which would make those layers alignment-critical. That does not apply here:

- `Qwen3.8-27B` `config.json` has **`vision_config.deepstack_visual_indexes: []`** — empty.
- `modeling_qwen3_5.py` contains **zero** references to deepstack. (For contrast, [`modeling_qwen3_vl.py`](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py) has 29, and [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/raw/main/config.json) sets `deepstack_visual_indexes: [8, 16, 24]`.)

So the **only** vision→LLM coupling in `Qwen3.8-27B` is a single `masked_scatter` of merger output into the input embedding sequence. One interface, at layer 0, not four. Advice inherited from Qwen3-VL about protecting early LLM layers does not carry over.

*(Aside, verified: `Qwen3.6-27B` and `Qwen3.8-27B` `config.json` differ in exactly one field, `transformers_version`. The issue's "field-for-field identical" claim holds.)*

### What actually drifts: mostly output format, not visual knowledge

This is the refinement that matters. Three independent measurements say the dominant component of measured "forgetting" is **instruction/format drift**, not loss of visual capability:

- **[CoIN](https://arxiv.org/abs/2403.08350)** (LLaVA-1.5-7B, LoRA, ViT frozen) separates the two and quantifies them: Instruction-Following backward transfer **−32.62** vs General-Knowledge backward transfer **−11.75**. Their conclusion: *"the failure in intention alignment assumes the main responsibility, instead of the knowledge forgetting."* The canonical failure is answering "Two apples" where the ground truth is "Two".
- **[VFA](https://arxiv.org/abs/2608.26155)**'s catastrophic case is a *binary* task. Qwen2.5-VL-7B after text-only SFT scores **0.00 on MaRVL**, down from 53.50. MaRVL is a true/false judgement over an image pair ([Liu et al. 2021](https://arxiv.org/abs/2109.13238): *"discriminating whether each grounded statement is true or false"*), so chance is ~50%. **Scoring 0.00 is far below chance — that is a format or refusal collapse, not a graded loss of visual skill.** (My inference from the task definition, clearly labelled as such; the paper does not diagnose it.)
- **[Fine-tuning MLLMs Without Forgetting Is Easier Than You Think](https://arxiv.org/abs/2603.14493)** finds damage appears **only** in the out-of-distribution-*text* quadrant, not the OOD-*image* quadrant, and names it *"task-specific overfitting"* — the model memorises the fine-tuning prompt template and stops attending to the prompt at inference.

**Practical consequence.** The mitigation is not only "put images in the mix" — it is **task and output-format diversity**. A run whose every example is an agent trajectory in one rigid format is exposed even if a healthy fraction of them contain screenshots. This reframes the ticket: prompt/format variety is a first-class lever alongside the image ratio.

---

## Q1 — How much does it actually degrade? (magnitude and detectors)

### The direct evidence is thin, and bimodal

Only a handful of papers do the actual experiment (take an aligned VLM → SFT on text → measure vision). The results split sharply.

**[VFA — Vision-Free Adaptation](https://arxiv.org/abs/2608.26155)** is the strongest single case. 100K text-only multilingual instructions, **full FT of the LLM backbone only** (ViT + projector frozen), 1 epoch, LR 1e-5. Table 2, "Direct Text SFT" vs base:

| model | MaXM | xGQA | xMMMU | XM100 | MaRVL | M3Exam | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen2.5-VL-7B** | −0.69 | **−16.61** | +1.35 | −1.94 | **−53.50** | −0.11 | **−11.92** |
| **LLaVA-Next-8B** | +12.17 | +0.41 | −2.23 | +12.29 | **−39.83** | −3.24 | −3.40 |
| Idefics3-8B | +0.16 | −3.78 | −6.38 | +0.26 | **+11.50** | +16.86 | **+3.10** |
| LLaVA-OV-1.5-8B | −1.05 | −1.12 | +1.09 | −1.44 | −0.34 | +0.22 | **−0.44** |
| LLaVA-OV-1.5-4B | −0.05 | −4.32 | −0.39 | −0.54 | −0.17 | −0.04 | **−0.92** |

**This variance is the headline, and it is under-reported.** Only 2 of 5 models collapsed. The two **LLaVA-OneVision-1.5** models — the most recent, most heavily multimodally-post-trained of the set, both on Qwen3 backbones — were essentially **unaffected** (−0.44 and −0.92 average, MaRVL −0.34 and −0.17). Idefics3 *improved*. So "text-only SFT destroys a VLM" is not a law; it is a property of some checkpoints.

`Qwen3.8-27B` is described by Qwen as *"Early fusion training on trillions of multimodal tokens"* with *"near-100% multimodal training efficiency compared to text-only training"* ([QwenLM/Qwen3.8](https://github.com/QwenLM/Qwen3.8)). That places it, on maturity of alignment, much closer to the LLaVA-OV-1.5 end of that table than to Qwen2.5-VL-7B. **This is a plausibility argument, not evidence** — nobody has run the experiment on this model — but it is the right prior.

**[VOLD](https://arxiv.org/abs/2510.23497)** (Qwen2.5-VL-3B, **LoRA**, text-only distillation data) is the only text-only *LoRA* datapoint I found: MMStar aggregate **55.9 → 55.2 (−0.7)**, but the split is **perception −4.0 / reasoning +2.4**. The authors argue this is a reasoning-tuning artifact rather than text-only-specific, since a model trained on images shows the same pattern — treat the attribution as contested, the number as real.

### Adjacent evidence, for calibration on the severe tail

These are narrow *multimodal* fine-tunes, not text-only, so they bound the worst case rather than describe ours:

- **[EMT](https://arxiv.org/abs/2309.10313)** — LLaVA, ViT frozen throughout. Adapter-only FT on miniImagenet: CIFAR-10 56.71 → 38.99. **LoRA on adapter + LLM** on CIFAR-10: MNIST 56.96 → **2.80**. Touching the LLM turns a −18 pp problem into a near-total collapse.
- **[CoIN](https://arxiv.org/abs/2403.08350)** — 8-task LoRA sequence: ImageNet 96.05 → 10.25, ScienceQA 82.45 → 21.26, BWT **−32.62**.
- **[MDGD](https://arxiv.org/abs/2502.11740)** — LLaVA-1.5 FT on Flickr30K: six-benchmark pre-trained average **61.94 → 47.97** (−14.0 pp). Attributes forgetting to instruction tuning being *"text-driven with limited direct visual supervision"*, measured as a drop in the **effective rank** of visual features.

Note that EMT, CoIN and MDGD all observe severe degradation **with the vision encoder frozen for the entire run**. That is strong corroboration for the issue's mechanism claim: this is not encoder drift.

### Which benchmarks detect it — and which are decoys

The most actionable finding in this whole review: **the standard English leaderboard did not notice a catastrophe.** In VFA's Qwen2.5-VL-7B run, while MaRVL went to zero and xGQA fell 16.6 pp, the general benchmarks moved (Table 3): **OCRBench +0.20, MMBench −0.60, MMMU +0.22, MathVista +4.00.**

Cross-referencing that against [MMStar](https://arxiv.org/abs/2403.20330), which measures how much of each benchmark an LLM answers **with no image at all**:

| benchmark | answerable without the image | verdict as a drift tripwire |
|---|---:|---|
| MMBench | 18.4% | High visual dependency, but **empirically inert** — moved −0.6 pp during a −53.5 pp failure |
| SEED | 35.5% | Medium; no measured text-only-SFT delta found |
| MathVista | 25.6% | Medium; *rose* +4.0 during the failure |
| MMMU | 42.9% | **Decoy** — largely LLM knowledge |
| AI2D | 59.2% | **Decoy** |
| ScienceQA | 68.9% | **Decoy** |

**Recommended tripwires**, in priority order:

1. **Open-ended grounded VQA scored with strict matching** (GQA / xGQA style). Fired first and hardest in every case; catches format collapse and grounding loss together.
2. **Free-form OCR/document generation** (TextVQA, DocVQA, or this model's own OmniDocBench). Goes to zero in severe cases.
3. **MMStar — read the perception subscore separately.** Perception moved 4 pp where the aggregate moved 0.7 pp.
4. **An out-of-format prompt probe.** [2603.14493](https://arxiv.org/abs/2603.14493) amplified a −10.4 pp signal to −37.4 pp simply by adding a class-name distractor. This is cheap and it is the most sensitive instrument available.

**Do not use MMMU, ScienceQA-IMG or AI2D as tripwires.** A rising MMMU after text-heavy SFT is not evidence that alignment survived.

---

## Q2 — What multimodal ratio prevents it?

### Honest answer: the exact measurement you want has not been published

The literature is overwhelmingly about the **opposite** direction — how much *text-only* data to add to a *multimodal* mix to protect language ability. Almost every frontier VLM report has a number for that. **Nobody has published a sweep that takes an already-aligned VLM, fine-tunes it on a text-heavy domain mix, and walks the multimodal fraction down through 20 / 10 / 5 / 1 / 0% while measuring vision.** Every close analogue changes the alignment stage, the model scale, or the benchmark suite at the same time.

So the ratio below is an **extrapolation across a setup gap**, and should be paired with a measured hold-out rather than trusted on its own.

### MLAN — the only direct dose-response curve

[MLAN](https://arxiv.org/abs/2411.10557) fixes an 80K-instance budget and sweeps the language-only share from 0% to 100%, filling the rest with vision-language data. Table 5, Llama-2 (the `% Language Data` column is the *text-only* share, so vision share = 100 − x):

| % language | vision share | **Vision Avg** | GQA | RealWorldQA | AI2D | POPE (F1) | Lang Avg |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0% | 100% | 53.8 | 44.0 | 46.5 | 48.8 | 75.8 | 66.2 |
| 12.5% | 87.5% | 55.5 | 44.7 | 44.8 | 50.2 | 82.4 | 71.0 |
| 25% | 75% | 56.2 | 44.6 | 47.8 | 54.8 | 77.5 | 72.2 |
| 37.5% | 62.5% | **57.0** | 44.8 | 48.5 | 54.6 | 80.3 | 74.4 |
| 50% | 50% | 54.8 | 43.6 | 46.4 | 54.5 | 74.7 | 73.9 |
| 62.5% | 37.5% | 55.7 | 44.8 | 48.9 | 52.3 | 76.6 | 74.6 |
| 75% | 25% | 56.6 | 44.3 | 46.7 | 54.0 | 81.3 | 73.3 |
| **87.5%** | **12.5%** | **56.2** | 42.5 | 45.0 | 54.7 | 82.8 | 74.3 |
| **100%** | **0%** | **49.5** | **33.8** | 41.8 | 52.7 | **69.6** | 74.4 |

**Read the last two rows.** Going from 12.5% vision data to 0% costs **6.7 pp of vision average** — GQA −8.7, POPE −13.2, RealWorldQA −3.2, AI2D −2.0. Everything from 12.5% to 87.5% vision sits in a 54.8–57.0 band, i.e. flat within noise.

**The cliff is at zero, not at low fractions.** That is the single most decision-relevant shape in the literature.

Also note *which* benchmarks moved: **GQA and POPE** took the damage; **AI2D barely moved** (−2.0) — consistent with the decoy analysis above.

**Caveats, stated plainly.** MLAN starts from a stage-1 (projector-aligned, not instruction-tuned) checkpoint, uses **full fine-tuning** of the LLM at LR 2e-5 for 1 epoch, and its absolute vision numbers are modest. It measures "does a language-heavy instruction mix unlock vision" more than "does a language-heavy SFT break an aligned VLM". It is the closest published thing to our question, not the same question.

### Converging anchors from the other direction

The direction-(a) ablations are much stronger evidence and land in the same place — **~10% is where the curve goes flat**:

| source | finding |
|---|---|
| [MM1](https://arxiv.org/abs/2403.09611) | Sweeping image:text-only, TextCore 52.2 (0% text) → 54.0 (9% text) → 54.6 (33% text). **~9% buys most of the benefit**; 33% costs 1.2 pp of 8-shot. Final mix 45:45:**10**. |
| [MM1.5](https://arxiv.org/html/2409.20566v1) | Sweeps SFT text weight 0→0.2, *"minor effects"* across the range, picks **10%**. |
| [Molmo](https://arxiv.org/html/2409.17146v2) | Tulu 3 down-sampled to **10%** beat the full set on **both** axes (MMLU 65.4 vs 64.9, multimodal avg 77.1 vs 76.9). |
| [VILA](https://arxiv.org/abs/2312.07533) | Blending text-only into visual SFT recovered MMLU (−5.8% → −0.3%) **and improved vision** (OKVQA 69.0 → 72.3). Mixing is not a tax. |
| [Should VLMs be Pre-trained with Image Data?](https://arxiv.org/abs/2503.07603) | *"from 10% to 20% of tokens should be visual. Going above or below this ratio results in worse downstream performance."* (Pre-training, 1B scale; the paper says the ratio moves with scale.) |
| [2603.14493](https://arxiv.org/abs/2603.14493) | 50% LLaVA-665K hybrid keeps target-task accuracy within ~1 pp while fixing the OOD-text failure; **70% adds nothing further**. |

For context, the *stated* text-only share across frontier models spans **4.92% ([InternVL 2.5](https://arxiv.org/html/2412.05271v4)) to 23.8% ([Cambrian-1](https://arxiv.org/abs/2406.16860)) to 50% ([Qwen2.5-VL](https://arxiv.org/html/2502.13923v1)) to 62% (mPLUG-Owl)** ([survey table, MLAN Table 8](https://arxiv.org/html/2411.10557v3)). **There is no consensus number** — only a consensus that it is non-zero. Same conclusion applies, mirrored, to our direction.

Qwen's own practice, for what it is worth: [Qwen3-VL](https://arxiv.org/abs/2511.21631) SFT cold start is *"one-third … text-only entries and the remaining two-thirds … image-text and video-text pairs"*, and the long-CoT cold start maintains *"an approximate 1:1 ratio between vision-language and text-only samples."*

### The trap: instances vs tokens

**This is the part most likely to bite in practice.** Every ratio above is quoted in **instances**, but our situation is asymmetric in a way the papers' are not: an agent trajectory is a very long text document with an occasional screenshot in it. A mix where **30% of examples contain an image** could still be **well under 1% visual tokens** — the loss is computed over text tokens either way, and the gradient signal reaching the vision-reading pathway scales with how much of the sequence is visual.

**Recommendation:** instrument both. Report (i) the fraction of examples containing ≥1 image and (ii) the fraction of total training tokens that are visual tokens. Use ~10% of *examples* as the floor to act on, and treat a near-zero *token* share as a warning sign worth a second look — the token-share literature ([2503.07603](https://arxiv.org/abs/2503.07603)'s 10–20% of tokens) is about pre-training and does not transfer cleanly, so I am not proposing a token target, only that you look at the number before assuming the mix is safe.

### What the GUI-agent literature says: nothing

I checked, because agent trajectories with screenshots are the closest published analogue. [UI-TARS](https://arxiv.org/abs/2501.12326) contains **zero** occurrences of "forgetting", "general data", "data mixture", "replay" or "rehearsal". The GUI-agent papers (UI-TARS, OS-Atlas, Aguvis) train on data that is overwhelmingly screenshot-bearing, so the question never arises for them. **This is a negative result worth recording** — do not expect to find the answer there.

---

## Q3 — Vision-tower targeting

Given the [premise correction](#the-premise-is-wrong-the-vit-was-never-being-adapted), the question becomes: *should* the ViT be adapted? The answer is a clear no, from four independent directions.

**1. Every framework default freezes it.**

| framework | setting | default |
|---|---|---|
| [ms-swift](https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/source_en/Instruction/Command-line-parameters.md) | `freeze_vit` | **True** |
| ms-swift | `freeze_aligner` | **True** |
| [LlamaFactory](https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/src/llamafactory/hparams/finetuning_args.py) | `freeze_vision_tower` | **True** |
| LlamaFactory | `freeze_multi_modal_projector` | **True** |

ms-swift documents the intent explicitly: *"For multimodal LLMs, tuners are by default only attached to the LLM component."*

**2. Qwen's own fine-tuning scripts freeze it.** [`qwen-vl-finetune`](https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/qwen-vl-finetune/README.md) uses `--tune_mm_llm True --tune_mm_vision False --tune_mm_mlp False`, with the note *"If trained with both image and video data, tune_mm_vision should be False."* The Qwen3-VL LoRA script `sft_30a3b_lora.sh` sets `tune_mm_vision False`.

There is also a **merged, official ms-swift example doing almost exactly our run**: [PR #9920](https://github.com/modelscope/ms-swift/pull/9920) (merged 2026-08-18) adds `qwen3_8_27b_lora_fsdp2.sh` — LoRA rank 8 / alpha 32 on `--model_type qwen3_5`, trained on **pure-text datasets**, LR 1e-4, with no `freeze_vit` override, so the `True` defaults apply and LoRA lands on the LLM only.

**3. Tuning the ViT is measurably harmful at a normal LR.** [2603.14493](https://arxiv.org/abs/2603.14493) Table 1:

| trainable part | setting | val acc | Δ OODT–IDI | Δ OODT–OODI |
|---|---|---:|---:|---:|
| LLM backbone | Full, 1e-5 | 91.56 | **−16.56** | **−33.64** |
| LLM backbone | **LoRA, 1e-4** | 91.08 | **+0.46** | **−2.97** |
| LLM backbone | Full, 1e-6 | 91.37 | +1.06 | −1.51 |
| **Vision encoder** | Full, 1e-6 | 90.96 | −1.36 | +0.49 |
| **Vision encoder** | **Full, 1e-5** | 91.08 | **−9.90** | −2.76 |
| Projector | Full, 1e-6 | 86.86 | +0.26 | +0.05 |
| Projector | Full, 1e-5 | 89.68 | −0.64 | −0.26 |

The vision encoder is fine at 1e-6 and costs **−9.90 pp at 1e-5** — it needs its own, much lower LR to be safe. (This is why Qwen's scripts carry separate `--vision_tower_lr 1e-6` / `--mm_projector_lr 1e-5` knobs.) Also note the projector-only row: it is the *worst* option for target-task accuracy (−4.70 pp), so "adapt the projector instead" is not a better idea.

**4. In a text-heavy mix specifically, a ViT adapter would train on a small, biased sample.** It receives gradient only on the image-bearing minority of batches. That is the worst case for a component every downstream representation depends on: few updates, drawn from a narrow slice (render screenshots), applied to a general-purpose encoder. A frozen ViT has zero such risk, and the tower is only ~460M of 27B params (~1.7%) — there is nothing to gain by unfreezing it.

**Verdict:** freeze it, explicitly and assertively. If a future vision-aware fine-tune genuinely needs the tower, it needs a separate, much lower LR — not membership in the same `target_modules` list.

---

## Q4 — Do rank and learning rate change the risk?

**Learning rate: decisively yes, and it dominates everything else.** The 2603.14493 table above is the cleanest statement in the literature: **identical target-task accuracy (91.56 vs 91.37), and OOD damage moves from −33.64 pp to −1.51 pp on a single 10× LR cut.** LoRA r=8 at 1e-4 sits at −2.97 pp, i.e. in the same safe regime as a 10×-reduced full fine-tune.

**LoRA vs full fine-tuning: LoRA forgets less, measurably.** [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673) (Biderman et al., TMLR): *"LoRA substantially underperforms full finetuning [on the target domain]. Nevertheless, LoRA better maintains the base model's performance on tasks outside the target domain … LoRA mitigates forgetting more than common regularization techniques such as weight decay and dropout"*, and *"full finetuning learns perturbations with a rank that is 10-100X greater than typical LoRA configurations."* The rank gap is the proposed mechanism for the forgetting gap.

But note the asymmetry in our case: LoRA's regularisation benefit is measured on **target-domain underperformance vs out-of-domain retention**. Our out-of-domain capability is vision, which is exactly what we are protecting — so this cuts in our favour.

**Rank: no direct measurement found for this case.** I did not find a paper that sweeps LoRA rank and measures VLM alignment retention. The indirect argument (higher rank → closer to full FT → more forgetting, per Biderman's rank observation and [LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://arxiv.org/abs/2410.21228)'s intruder-dimension analysis) is plausible but **not measured for VLM alignment**. At `r=16` / 0.039% of parameters we are far into the low-rank regime, and there is no evidence that moving within the usual 8–64 band matters here. **Do not tune rank for drift protection; tune LR.**

**Steps/epochs.** EMT reports significant degradation *"after just a single epoch"*, so short training does not bound the risk on its own — but everything else in that setup was aggressive (LLM + adapter, narrow classification data).

**Reversibility — an underrated safety property.** LoRA drift is undoable in a way full-FT drift is not: keep the adapter unmerged and the base model is still exactly the shipped VLM. Separately, [2412.03467](https://arxiv.org/abs/2412.03467) shows forgetting is largely a **weight-space offset recoverable by merging the base model back in** (task arithmetic at α=0.1 moved GSM8k 0.328 → 0.431 *and improved vision on 3 of 5 benchmarks*). That gives a cheap rollback lever if the eval does show drift.

---

## Q5 — Qwen3.8-specific findings

Everything below is read from primary source, not inferred.

- **Model exists and is a native VLM.** [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B): *"a native vision-language model that understands images and videos"*, `pipeline_tag: image-text-to-text`, Apache-2.0.
- **No DeepStack** (`deepstack_visual_indexes: []`) — see above. Single alignment interface at the embedding layer.
- **Vision tower is ~1.7% of the model**: 27-layer ViT, hidden 1152, patch 16, `out_hidden_size` 5120, merger = `norm` → `linear_fc1` → GELU → `linear_fc2` (4608 → 5120).
- **The checkpoint supports dropping the tower entirely at serve time**: `config.json` carries `"language_model_only": false`, and the Qwen3.6 card documents `vllm serve … --language-model-only` to *"skip the vision encoder and multimodal profiling to free up memory for additional KV cache"*. Useful for A/B serving the same adapter with and without the vision path.
- **The model card carries no fine-tuning guidance and no warning about degrading multimodal ability** — zero matches for `fine-tun|finetun|LoRA|degrad|forget`. Qwen's only statement, in the [GitHub README](https://github.com/QwenLM/Qwen3.8), is to point at Unsloth / ms-swift / LLaMA-Factory.
- **Framework support as of 2026-08-30:** ms-swift **supports** Qwen3.8 ([PR #9914](https://github.com/modelscope/ms-swift/pull/9914), merged 2026-08-14). LlamaFactory support is **still open** ([PR #10749](https://github.com/hiyouga/LlamaFactory/pull/10749)). `qwen-vl-finetune` does **not** handle this architecture ([Qwen3-VL #2092](https://github.com/QwenLM/Qwen3-VL/issues/2092), open) — it computes position IDs with `get_rope_index_3`, which is wrong for Qwen3.5's partial-RoPE + GatedDeltaNet.
- **Sharp edge if you enable sequence packing:** `neat_packing` + FlashAttention-2 has a history of producing invalid `cu_seqlens` and crashing the FLA GatedDeltaNet kernel on this architecture ([LlamaFactory #10452](https://github.com/hiyouga/LlamaFactory/issues/10452), closed; [#10450](https://github.com/hiyouga/LlamaFactory/issues/10450), open).
- **No documented case of a Qwen-VL model losing vision from text-heavy fine-tuning with the ViT frozen.** Searching QwenLM/Qwen3-VL, ms-swift and LlamaFactory turned up degradation reports, but all from *narrow multimodal overfitting* — [Qwen3-VL #600](https://github.com/QwenLM/Qwen3-VL/issues/600) (*"The model forgets that it has the ability to analyse images"* after 1 epoch on 26 videos of face attributes; the reporter adds that **LoRA reduced the problem**) and [#776](https://github.com/QwenLM/Qwen3-VL/issues/776) (grounding IoU halved after fine-tuning on a custom grounding set). Neither is our scenario.

### Published VL baselines to measure against

First-party numbers from the model card, for `Qwen3.8-27B` — these are the before-values for any regression check:

| capability | benchmark | Qwen3.8-27B |
|---|---|---:|
| Computer use | OSWorld-Verified | 84.3 |
| Multimodal software engineering | SWE-MM | 38.6 |
| Visual web development | Vision2Web | 62.9 |
| Application recreation | RecreationBench | 47.1 |
| Scientific chart analysis | CharXiv (RQ) | 83.7 (w/o CI) |
| Document intelligence | OmniDocBench 1.5 | 91.1 |
| Real-world perception | RealWorldQA | 85.9 |
| Embodied intelligence | ERQA | 65.5 |
| Visual math | MathVision | 90.0 (w/o CI) |

---

## How to measure this with the infrastructure we already have

The repo runs [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) against a served OpenAI-compatible endpoint (`packages/server/src/benchmarks/lm-eval-args.ts`). **That path cannot do image evals.** lm-eval's multimodal support is limited to the `hf-multimodal` and `vllm-vlm` model types — the API model types (`local-completions` / `local-chat-completions`) are not listed as multimodal, and the README itself is explicit that this is *"an in-progress feature"* and points elsewhere:

> *"we suggest they check out `lmms-eval` … for a broader range of multimodal tasks, models, and features."*

**[lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) is the right tool and it fits our architecture**: it exposes `--model openai`, and *"Any OpenAI-compatible endpoint works, including local vLLM/SGLang servers"* — i.e. it can point straight at the existing `/v1` gateway. It ships MMMU, MMBench, MME, POPE, TextVQA, ChartQA, DocVQA, AI2D, SeedBench, MathVista.

**Proposed minimal regression gate** (~1 evening of work, and it is the thing that makes the ratio decision empirical rather than theoretical):

1. Serve base `Qwen3.8-27B`, run the tripwire set below, record baselines.
2. Serve base + adapter, run the same set.
3. Gate on the delta.

Tripwire set, per the detector analysis: **GQA or TextVQA** (open-ended, strict matching — the sensitive one), **MMStar with the perception subscore read separately**, and **RealWorldQA** (we have a first-party baseline of 85.9 to compare against). Add a handful of hand-written out-of-format visual prompts — cheapest and most sensitive instrument available.

Two things make this unusually cheap here: the adapter can be served unmerged (so base and tuned differ by one flag), and `--language-model-only` lets you confirm the vision path is actually being exercised.

---

## Confidence, and where I would be wrong

| claim | confidence | why |
|---|---|---|
| The recipe never attached LoRA to the ViT on this model | **High** | Checkpoint tensor index + transformers source + exact parameter arithmetic reproducing 10,485,760 |
| Text-only batches give the ViT zero gradient | **High** | Read directly from the modeling source guard |
| The failure mode is LM drift, not encoder drift | **High** | Four papers observe severe forgetting with the ViT frozen throughout; two probing papers localise damage to mid-LLM layers |
| Much of the damage is output-format drift, not lost visual knowledge | **Medium-high** | CoIN quantifies it (−32.62 vs −11.75); the MaRVL-below-chance argument is my inference |
| Keep the ViT frozen | **High** | Converges from framework defaults, Qwen's own scripts, and a measured −9.90 pp at 1e-5 |
| LR matters more than the mix ratio | **Medium-high** | One strong measured table (−33.64 → −1.51); would like a second independent replication |
| ~10% image-bearing examples is a sufficient floor | **Medium** | Extrapolated across a setup gap — see below |
| `Qwen3.8-27B` is at the robust end of the spectrum | **Low-medium** | Plausibility argument from VFA's cross-model variance + Qwen's early-fusion claim. **Not measured.** |

**The gap that would change the answer.** No published experiment takes an already-aligned, instruction-tuned VLM, LoRA-fine-tunes it on a text-heavy mix, and sweeps the multimodal fraction toward zero while measuring vision. MLAN is the closest and it differs in three ways at once (stage-1 checkpoint, full FT, LR 2e-5). **The ~10% floor is therefore an engineering starting point justified by analogy, not a measured optimum** — which is precisely why the recommendation is paired with a measured hold-out rather than offered alone.

**What I would watch for.** If the chat3d trajectories turn out to be near-uniform in output format, format drift is a bigger risk than image count, and the mitigation shifts toward prompt/task diversity. And if the image-bearing fraction is high by example count but negligible by token count, the comfortable-looking ratio may be an illusion.

---

## Sources

**Direct evidence — text-heavy SFT of an aligned VLM**
- VFA: Empowering Multilingual MLLMs via Vision-Free Adaptation — https://arxiv.org/abs/2608.26155
- VOLD: Reasoning Transfer from LLMs to VLMs — https://arxiv.org/abs/2510.23497
- MLAN: Language-Based Instruction Tuning — https://arxiv.org/abs/2411.10557
- Breaking Language Barriers in VLMs (TR-3S) — https://arxiv.org/abs/2503.22577

**Forgetting mechanism and magnitude**
- Investigating Catastrophic Forgetting in MLLMs (EMT) — https://arxiv.org/abs/2309.10313
- CoIN: Continual Instruction Tuning benchmark — https://arxiv.org/abs/2403.08350
- Fine-tuning MLLMs Without Forgetting Is Easier Than You Think — https://arxiv.org/abs/2603.14493
- Mitigating Visual Knowledge Forgetting (MDGD) — https://arxiv.org/abs/2502.11740
- VIRAL: Visual Representation Alignment for MLLMs — https://arxiv.org/abs/2509.07979
- Progressive Multimodal Alignment (PMA) — https://arxiv.org/abs/2607.26947
- Training-Free Mitigation of Language Reasoning Degradation — https://arxiv.org/abs/2412.03467
- Wings: MLLMs without Text-only Forgetting — https://arxiv.org/abs/2406.03496

**Data mixture ablations**
- MM1 — https://arxiv.org/abs/2403.09611 · MM1.5 — https://arxiv.org/html/2409.20566v1
- Molmo / PixMo — https://arxiv.org/html/2409.17146v2
- VILA — https://arxiv.org/abs/2312.07533
- Cambrian-1 — https://arxiv.org/abs/2406.16860
- InternVL 2.5 — https://arxiv.org/html/2412.05271v4 · InternVL3 — https://arxiv.org/html/2504.10479v3
- Qwen2.5-VL — https://arxiv.org/html/2502.13923v1 · Qwen3-VL Technical Report — https://arxiv.org/abs/2511.21631
- Should VLMs be Pre-trained with Image Data? — https://arxiv.org/abs/2503.07603
- LLaVA-1.5 — https://arxiv.org/html/2310.03744v2

**Benchmark validity**
- MMStar: Are We on the Right Way for Evaluating LVLMs? — https://arxiv.org/abs/2403.20330
- MaRVL — https://arxiv.org/abs/2109.13238

**LoRA behaviour**
- LoRA Learns Less and Forgets Less — https://arxiv.org/abs/2405.09673
- LoRA vs Full Fine-tuning: An Illusion of Equivalence — https://arxiv.org/abs/2410.21228
- PEFT source, `lora/layer.py` — https://raw.githubusercontent.com/huggingface/peft/main/src/peft/tuners/lora/layer.py

**Model and framework primary source**
- `Qwen/Qwen3.8-27B` model card, `config.json`, `model.safetensors.index.json` — https://huggingface.co/Qwen/Qwen3.8-27B
- transformers `modeling_qwen3_5.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
- transformers `modeling_qwen3_vl.py` — https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py
- ms-swift command-line parameters — https://raw.githubusercontent.com/modelscope/ms-swift/main/docs/source_en/Instruction/Command-line-parameters.md
- LlamaFactory `finetuning_args.py` — https://raw.githubusercontent.com/hiyouga/LlamaFactory/main/src/llamafactory/hparams/finetuning_args.py
- `qwen-vl-finetune` README — https://raw.githubusercontent.com/QwenLM/Qwen3-VL/main/qwen-vl-finetune/README.md
- lm-evaluation-harness README — https://github.com/EleutherAI/lm-evaluation-harness
- lmms-eval README — https://github.com/EvolvingLMMs-Lab/lmms-eval

**Negative result**
- UI-TARS — https://arxiv.org/abs/2501.12326 — contains no discussion of data mixture, replay, or forgetting.
