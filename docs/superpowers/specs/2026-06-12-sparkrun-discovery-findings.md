# Sparkrun Discovery Spike — Findings

**Date:** 2026-06-12
**Node:** dgx-spark-01 (192.168.44.36), head of the live 4-node cluster
**sparkrun version:** **0.2.38** (via `uvx --from sparkrun sparkrun`)
**Status:** ✅ **COMPLETE.** V1/V2/V5 resolved from safe commands; V3/V4 resolved by a live
qwen3-1.7b run on a freed node (port 8011), then cleanly stopped. Major finding: first run does a
~15-min from-source image build (see §Major finding).

Captured fixtures (in `packages/agent/src/runtime/__fixtures__/`):
`sparkrun-list.json`, `sparkrun-show.txt`, `sparkrun-export-recipe.yaml`, `sparkrun-run-dryrun.txt`.

## V1 — Non-interactive setup ✅ RESOLVED

`sparkrun setup` is a command group with granular, scriptable subcommands — the interactive
`wizard` is optional, not required:

```
setup install         Install sparkrun and tab-completion.
setup ssh             Set up passwordless SSH mesh across cluster hosts.
setup earlyoom        Install and configure earlyoom OOM killer on cluster hosts.
setup cx7             Configure CX7 network interfaces on cluster hosts.
setup docker-group    Ensure user is a member of the docker group on cluster hosts.
setup fix-permissions Fix file ownership in HuggingFace cache on cluster hosts.
setup clear-cache     Drop the Linux page cache on cluster hosts.
setup wizard          Guided setup wizard (INTERACTIVE — avoid in provisioner).
```

**Implication for Task 11:** the provisioner runs the specific subcommands it needs
(`setup ssh`, `setup earlyoom`, `setup docker-group`, optionally `setup cx7`) rather than a
single `--non-interactive` flag. Each takes `-H/--hosts` (or `--cluster`). Capture each
subcommand's `--help` for its exact host flags during implementation.

## V2 — Machine-readable output ⚠️ PARTIAL (workable)

| Command | JSON? | Programmatic source to use |
|---|---|---|
| `list` | **Yes** `--json` | Primary catalog source. Clean array. |
| `show` | No | Use `export recipe <name>` (YAML) for defaults/metadata; parse `show` **text** for the VRAM estimate block. |
| `status` | No | Text, keyed by **cluster ID** (`sparkrun_<hash>`). For liveness prefer `cluster check-job`. |
| `export recipe <name>` | YAML | Normalized recipe: `defaults`, `metadata`, `runtime`, `container`, `command`, `builder`. |

### `list --json` shape (per entry)

```json
{
  "name": "@sparkrun-transitional/qwen3-1.7b-vllm",
  "file": "qwen3-1.7b-vllm",
  "path": "/home/daniel/.cache/sparkrun/registries/.../qwen3-1.7b-vllm.yaml",
  "model": "Qwen/Qwen3-1.7B",
  "description": "",
  "runtime": "vllm-distributed",
  "min_nodes": 1,
  "tp": 1,
  "gpu_mem": 0.3,
  "registry": "sparkrun-transitional"
}
```
- `name` is the launch ref (`@registry/file`); `file` is the short name; both accepted by `run`.
- `tp` / `gpu_mem` can be empty strings when the recipe doesn't set them — parser must tolerate `"" | number`.
- 48 recipes present; only the `sparkrun-transitional` registry is configured on this node.

### `export recipe` YAML shape (defaults source for admission/deploy)

```yaml
recipe_version: '2'
model: Qwen/Qwen3-1.7B
runtime: vllm
builder: eugr
container: ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5:latest
metadata: { description: ..., maintainer: ..., model_params: 1.7B, model_dtype: bf16 }
defaults: { port: 8000, host: 0.0.0.0, tensor_parallel: 1, gpu_memory_utilization: 0.3, served_model_name: qwen3-1.7b }
command: |
  vllm serve {model} --served-model-name {served_model_name} ...
```
- **No `metadata.model_vram`** in this recipe. VRAM estimate is computed by `show` (text block:
  `Per-GPU total: 3.17 GB`, `DGX Spark fit: YES`). So admission VRAM should parse the `show`
  text *or* keep DGX Manager's own estimate; do **not** rely on a `model_vram` field.
- Exported `runtime` is normalized to `vllm` even though `list` reports `vllm-distributed`.

### `run --dry-run` output

6 stages (Preparing → Building image → Distributing resources → Syncing tuning configs →
Launching runtime → Post-launch hooks). Emits `Mode: solo`, the resolved `Serve command`, and a
`Cluster: sparkrun_<hash>` ID. **The cluster ID is the workload handle** for `stop`/`status`.

## V3 — Detached lifecycle / reconnect ✅ RESOLVED (live run)

Confirmed on a freed node with `run qwen3-1.7b-vllm -H 192.168.44.36 --port 8011 --no-follow`:

- **Container naming:** `sparkrun_<hex>_solo` (here `sparkrun_8a0bcc7080b5_solo`). **NOT
  `vllm_node`** → no collision with eugr's container; the earlier collision worry is unfounded.
  Cluster mode presumably uses a `_cluster`-style suffix (verify when multi-node is tested).
- **Detached survival:** after `--no-follow` the launching `sparkrun run` process **exits**, yet
  the workload keeps running. `sparkrun status -H <host>` re-discovers it via SSH+`docker ps`.
  So reconnect/reconcile works without a live launcher — exactly what Design B needs.
- **Stable handle:** the short **cluster ID** `8a0bcc7080b5` (hex of `sparkrun_<hex>`). `status`
  prints `stop: sparkrun stop 8a0bcc7080b5` / `logs: sparkrun logs 8a0bcc7080b5`.
- **Liveness probe (clean):** `sparkrun cluster check-job <cluster-id|recipe> -H <hosts>` →
  **exit 0 = running**, 1 = not; add `--check-health` for a health gate. Prefer this over parsing
  `status` text for reconciliation.
- **Stop:** `sparkrun stop <cluster-id> -H <host>` → "Workload stopped on 1 host(s)."; container
  removed, port released, verified clean.

### `status` text format (fixture `sparkrun-status.txt`)

```
Job: @sparkrun-transitional/qwen3-1.7b-vllm  (tp=1)  [8a0bcc7080b5]  (1 container(s))
  solo       192.168.44.36                            Up 3 minutes              sparkrun-eugr-vllm-tf5
  logs: sparkrun logs 8a0bcc7080b5
  stop: sparkrun stop 8a0bcc7080b5
Total: 1 container(s) across 1 host(s)
```
Parse the `[<hex>]` cluster ID + `(tp=N)` + per-host line. **But for liveness, use
`cluster check-job` (exit code) — simpler and less brittle than this text.**

**Agent must capture the cluster ID at launch** — it appears as `Cluster: sparkrun_<hex>` in
`run` output. Store `{ deploymentId -> clusterId, hosts, tp }` for later `stop`/`check-job`.

### `run` / `stop` flag reference (from `--help`)

`run`: `-H/--hosts`, `--hosts-file`, `--cluster`, `--tp`, `--pp`, `--gpu-mem`, `--max-model-len`,
`--image`, `-o key=value`, `--port`, `--served-model-name`, `--dry-run`, `--foreground`,
`--ensure`, `--no-follow`, `--no-rm`, `--memory-limit`, `--rootful`, `--label`. Default is
detached + follow logs; `--no-follow` returns immediately (what the agent uses).

`stop [TARGET]`: TARGET = recipe name **or** cluster ID; `--all` discovers via `docker ps`; takes
`-H`, `--tp`, `--port`, `--served-model-name`. **`status`/`stop`/`check-job` require `-H/--hosts`**
(no default cluster here) — the agent always passes its own host(s).

## V4 — vLLM /metrics ✅ RESOLVED (live run)

`curl http://<host>:8011/metrics` returned standard vLLM Prometheus metrics. **Important: metric
lines carry Prometheus `{labels}`** — the parser regex MUST account for them:

```
vllm:num_requests_running{engine="0",model_name="qwen3-1.7b"} 0.0
vllm:num_requests_waiting{engine="0",model_name="qwen3-1.7b"} 0.0
vllm:kv_cache_usage_perc{engine="0",model_name="qwen3-1.7b"} 0.0
```

This vLLM (`0.22.1rc1`) uses **`vllm:kv_cache_usage_perc`** (not `gpu_cache_usage_perc`). Fixture:
`sparkrun-vllm-metrics.txt`. `GET /v1/models` confirms serving (`id: qwen3-1.7b`,
`max_model_len: 40960`). **Plan correction (Task 7):** `parseVllmMetrics` must match
`^vllm:<name>(\{[^}]*\})?\s+<value>` — the brace labels broke the plan's original regex.

## V5 — Version pin ✅

Pin **`sparkrun==0.2.38`** (`SPARKRUN_PKG = "sparkrun==0.2.38"`). The website docs reflect an
older 0.0.x line and disagree with the live CLI — trust the live `--help` over the website.

## Design-relevant surprises (fold into spec)

- **Sparkrun wraps eugr.** Default image `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5`,
  exported recipes carry `builder: eugr`. The migration retires eugr's *launch scripts*
  (`run-recipe.sh` et al.), not eugr's container. Lowers risk; metrics stay vLLM-shaped.
- **Registries are namespaced + cached** under `~/.cache/sparkrun/registries/<registry>/...`;
  recipe refs are `@registry/name`. Registry management is its own command group
  (`sparkrun registry …`) — relevant if we want DGX Manager to add custom registries.
- **Sparkrun ships its own LiteLLM proxy** (`sparkrun proxy`). Out of scope — DGX Manager keeps
  its own inference proxy/load balancer — but noted so we don't accidentally double-proxy.
- `run` has `--image` to override the container and `--ensure` (idempotent launch) — both useful
  for the agent launcher.

## ⚠️ Major finding — first run BUILDS the image from source (not a pull)

When `qwen3-1.7b-vllm` was actually launched (`run -H <self> --port 8011 --no-follow`), stage
**[2/6] Building image** did **not** pull `ghcr.io/spark-arena/dgx-vllm-eugr-nightly-tf5` as the
dry-run implied. Instead sparkrun ran:

```
~/.config/sparkrun/cache/eugr-spark-vllm-docker/build-and-copy.sh -t sparkrun-eugr-vllm-tf5 --tf5 --cleanup
  -> docker build -t sparkrun-eugr-vllm-tf5 \
       --build-arg TORCH_CUDA_ARCH_LIST=12.1a --build-arg FLASHINFER_CUDA_ARCH_LIST=12.1a ...
```

i.e. a **full from-source build of the eugr vLLM image** (PyTorch + FlashInfer for Blackwell
sm121), which takes tens of minutes to hours on a fresh node. The `--tf5` recipe variant maps to
a locally-built `sparkrun-eugr-vllm-tf5` tag rather than the ghcr image.

**Design implications (fold into spec/plan):**
1. **Provisioning must pre-warm the sparkrun image** (Task 11) — run a build/`run --dry-run` or an
   explicit image build during node provisioning, mirroring how DGX Manager already pre-warms the
   benchmark uvx cache and builds agent bundles. Otherwise the *first* deploy of a recipe family
   blocks for a very long time.
2. **Agent deploy phase-detection + timeouts (Task 6/8)** must treat "Building image" as a
   potentially multi-hour phase — no premature failure/timeout; surface build progress as a
   distinct status so the dashboard shows "building" rather than appearing hung.
3. **The eugr image is built locally per node** — so the eugr `build-and-copy.sh` is *not* fully
   retired; sparkrun vendors its own copy under `~/.config/sparkrun/cache/eugr-spark-vllm-docker`.
   We retire DGX Manager's *invocation* of `run-recipe.sh`, but the eugr build machinery lives on
   inside sparkrun.

## Safety note (why V3/V4 were initially deferred)

At spike time, `dgx-spark-01` was the head of a live `running` 4-node deployment
`nemotron-ultra-nomtp-caching` (Nemotron 3 Ultra NVFP4), holding port 8000 and ~96.5 GB/124 GB
VRAM. Launching even a small sparkrun workload risked (a) `vllm_node` container-name collision /
force-removal, (b) port 8000 conflict, (c) GPU oversubscription → OOM of the live deployment
(only ~28 GB free; qwen3-1.7b defaults to 0.3 util ≈ 36 GB). All four nodes belong to that
cluster, so no idle target existed. Per project guidance (don't disrupt active deployments), the
live run waits for a freed node.
