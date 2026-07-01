# GLM-5.2 (AWQ-INT4, 15% pruned) Inference Benchmark on DGX Spark

Date: 2026-07-01
Hardware: DGX Spark cluster, 4× NVIDIA GB10 (124 GB unified memory each, sm_121), TP=4
Model: `CosmicRaisins/GLM-5.2-AWQ-INT4-15pct` (GlmMoeDsaForCausalLM — DeepSeek-Sparse-Attention MoE)
vLLM: `0.23.1rc1.dev190+gab6660699` in a custom-built image (`vllm-node-tf5-glm52-b12x:probe`)
Served as: `glm-5.2` — deployed and benchmarked through the dgx-manager API

## TL;DR

**GLM-5.2 runs on the 4-Spark cluster** and is genuinely strong at tool use.

- **Tool-eval: 97 / 100 — ★★★★★ Excellent** (`tool-eval-quick` preset)
- **Throughput: 35.8 tok/s**, TTFR 1552 ms (`quick-smoke` preset, cudagraph FULL, c=1)
- Tool calling, reasoning parser, and 32K context all work end-to-end
- Requires a **custom image** — the stock eugr-nightly image cannot run the DSA indexer
  (missing sm_121 DeepGEMM kernels). Build: [glm-5.2-custom-image-build.md](./glm-5.2-custom-image-build.md)

## Methodology

Deployed via `POST /api/deployments` (recipe `@community-kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer`,
4 nodes auto-clustered, `displayName: "glm-5.2"`), then benchmarked via `POST /api/benchmarks` —
the same server-side `llama-benchy` (throughput) and `tool-eval-bench` (tool use) harness the
dashboard uses. Serve config: `-tp 4`, `--kv-cache-dtype fp8_ds_mla`, cudagraph FULL,
`--tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45`, gpu-mem 0.88,
max-model-len 32768.

## Results

### Throughput (`quick-smoke`: pp=128, tg=32, c=1, latencyMode=api)

| metric | value |
|---|---|
| mean tok/s | **35.81** |
| mean TTFR | **1552 ms** |

cudagraph FULL (with b12x) roughly 4–7× the eager-mode rate observed on earlier stock-image
attempts (~5–10 tok/s). This is single-stream (c=1); aggregate throughput under concurrency
was not swept here.

### Tool use (`tool-eval-bench`, `tool-eval-quick` preset)

| metric | value |
|---|---|
| score | **97 / 100** |
| rating | **★★★★★ Excellent** |

Spot-check (single-turn, forced tool): the model emits a correct structured call plus a
natural preamble —

```
prompt: "What is the weather in Paris? Use the tool."
tool_call: get_weather({"city": "Paris"})
content:  "I'll check the weather in Paris for you right away!"
```

## Observations

- **The model is a reasoner** — it emits thinking tokens before answers (`--reasoning-parser glm45`),
  which lengthens tool-eval wall-clock but does not hurt the score.
- **Startup is slow: ~45 min** — ~12 min shard load + ~21 min silent AWQ/MoE weight-processing
  (GPU 0 %, appears wedged but is progressing) + cudagraph FULL capture.
- **Memory is tight on GB10:** at gpu-mem 0.88 the KV budget fits ~62 K tokens; 32 K is a safe,
  benchmark-friendly cap. Longer context needs a smaller model or fewer TP shards' worth of weights.
- **DSA sparse attention works on sm_121** via the Triton `sm12x` fallback kernels + the
  `deep_gemm.py` re-bind of all 6 impl-gated DeepGEMM entry points (see build doc).

## Recipe & artifacts

- Recipe: `kreuzhofer/community-recipe-registry` →
  `recipes/glm-5.2/kreuzhofer/glm-5.2-awq-15pct-vllm-kreuzhofer.yaml`
- Image: `vllm-node-tf5-glm52-b12x:probe` (custom build, all 4 nodes)
- Model cache: `/mnt/tank/models/hub/models--CosmicRaisins--GLM-5.2-AWQ-INT4-15pct`
- Benchmark runs (dgx-manager DB): throughput `cmr1a4tm02pp136k42hhf2d9u`,
  tool-eval `cmr1a5w812pqi36k4q6teol67`

## Iterations to first serve (each a distinct fix)

1. sm12x deep_gemm re-bind (3 fns) → cleared the `sm121_fp8_mqa_logits` JIT crash
2. gpu-mem 0.93 → 0.88 (GB10 free-memory gap) → model loads
3. added 4th fn `get_paged_mqa_logits_metadata` → past first warmup crash
4. max-model-len 262144 → 32768 (KV budget) → past the KV check
5. added 5th + 6th fns (`get_mk_alignment_for_contiguous_layout`, `get_col_major_tma_aligned_tensor`)
   → warmup clears, cudagraph captures, **serves**

## Bugs surfaced (dgx-manager)

- Benchmark uses `deployment.displayName ?? model.name` as the OpenAI model name, not the
  recipe's `served_model_name` → 404 unless `displayName` is set to match. Worked around with
  `displayName: "glm-5.2"`; the real fix is server-side.
- `POST /api/recipes/refresh` updates `sparkrun list` but not the run-cache `sparkrun run` reads;
  recipe edits require a `git pull` of the registry cache on each node.
- Deployment status flips to `running` on container-up, not API-ready (fixed separately in the
  agent: gate on the `/metrics` probe; commit `188f739`).
