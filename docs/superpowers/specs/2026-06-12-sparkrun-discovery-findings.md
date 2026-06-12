# Sparkrun Discovery Spike — Findings

**Date:** 2026-06-12
**Node:** dgx-spark-01 (192.168.44.36), head of the live 4-node cluster
**sparkrun version:** **0.2.38** (via `uvx --from sparkrun sparkrun`)
**Status:** V1, V2, V5 resolved from safe commands (help / `--dry-run` / `export`). V3, V4
**deferred** — a live qwen3-1.7b run was not performed because an active 4-node Nemotron 3 Ultra
deployment was occupying ~96.5 GB/124 GB and port 8000 on this node (see §Safety). Will complete
V3/V4 when a node is freed.

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

## V3 — Detached lifecycle / reconnect ❓ DEFERRED

`run` flags (confirmed via `--help`): `-H/--hosts`, `--hosts-file`, `--cluster`, `--tp`, `--pp`,
`--gpu-mem`, `--max-model-len`, `--image`, `-o key=value`, `--port`, `--served-model-name`,
`--dry-run`, `--foreground`, `--ensure`, `--no-follow`, `--no-rm`, `--memory-limit`, `--rootful`,
`--label`. Default is detached + follow logs; `--no-follow` returns immediately.

`stop [TARGET]`: TARGET = recipe name **or** cluster ID; `--all` discovers via `docker ps`;
takes `-H`, `--tp`, `--port`, `--served-model-name` to match the run-time trimming. **`status`
and `stop` require `-H/--hosts` (or a saved cluster) — there is no default cluster configured
here.** The DGX Manager agent always knows its own host(s), so it passes them explicitly.

**Open questions for the live run:**
1. Exact `status` text format + the stable key (cluster ID vs recipe name) to store for `stop`.
2. Does a detached workload survive the launching process dying (reconnect via `status`)?
3. **Container naming — does sparkrun name its container `vllm_node` and `docker rm -f` it?**
   This is a collision/safety question (see §Safety) and decides whether two deployments can
   coexist on one node during the eugr→sparkrun migration.
4. Is `/metrics` reachable on `localhost:{port}` of the head (V4)?
5. `cluster check-job` output for a cleaner liveness probe than parsing `status`.

## V4 — vLLM /metrics ❓ DEFERRED

Dry-run confirms serve binds `0.0.0.0:8000`. Actual scrape pending the live run.

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

## Safety note (why V3/V4 were deferred)

At spike time, `dgx-spark-01` was the head of a live `running` 4-node deployment
`nemotron-ultra-nomtp-caching` (Nemotron 3 Ultra NVFP4), holding port 8000 and ~96.5 GB/124 GB
VRAM. Launching even a small sparkrun workload risked (a) `vllm_node` container-name collision /
force-removal, (b) port 8000 conflict, (c) GPU oversubscription → OOM of the live deployment
(only ~28 GB free; qwen3-1.7b defaults to 0.3 util ≈ 36 GB). All four nodes belong to that
cluster, so no idle target existed. Per project guidance (don't disrupt active deployments), the
live run waits for a freed node.
