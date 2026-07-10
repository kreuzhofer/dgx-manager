# GLM-5.2: from "it broke overnight" to 320K context

**Session: 2026-07-09 evening → 2026-07-10 morning.** Raw material for the external write-up
(see ROADMAP Phase 5, "External-facing write-up"). Everything below is measured, not inferred.

---

## 0. The setup

The unpruned GLM-5.2 (`QuantTrio/GLM-5.2-Int4-Int8Mix`, 377.6 GiB of weights) had been serving at
**256K context** across 4×GB10 (DGX Spark) since 2026-07-07, at 25–30 tok/s decode. Then it stopped
working — same image, same recipe, same everything.

Per rank, the 121.63 GiB unified pool holds: **weights 93.9 GiB** (377.6 ÷ tp4), **KV 7.0 GiB**, and
**~14 GiB** of CUDA graphs, activations and CUDA contexts. That's 83% before anything transient. Every
problem in this document is a fight over the remaining ~7 GiB.

---

## 1. Two memory walls, one root cause

The deploy died 44 seconds in, on all four ranks:

```
ValueError: Free memory on device cuda:0 (103.51/121.63 GiB) on startup is less than
desired GPU memory utilization (0.9, 109.46 GiB)
```

### Wall 1 — vLLM's startup guard

`request_memory()` requires `free ≥ gmu × total`. On GB10, the three vLLM processes
(`vllm` API server + `VLLM::EngineCore` + `VLLM::Worker`) reserve **~7 GiB of the unified pool inside
CUDA contexts that `/proc` never reports**. Caught live: host `MemFree` was 109.3 GiB while
`torch.cuda.mem_get_info()` saw ~102 GiB.

> A bare single-process container shows almost no gap (113.5 vs 113.7 GiB). Probing with one is
> actively misleading — it cost us an hour.

### Wall 2 — `vm.min_free_kbytes`

Lowering `gmu` cleared the guard, and the deploy then died *during* `torch.compile`:

```
CUDA out of memory. Tried to allocate 5.00 GiB.
GPU 0 ... of which 9.71 GiB is free.  This process has 101.86 GiB in use.
```

Failing to get 5.00 GiB **with 9.71 GiB free.** The kernel refuses to allocate below `min_free_kbytes`,
which node-prep had set to **5 GiB** on 2026-07-07 to *"guarantee capture headroom"* by keeping page
cache out. Usable was 4.71 GiB. **It missed by 0.29 GiB.** Every rank, identically:

| rank | free | usable (− 5 GiB reserve) | needed |
|---|---|---|---|
| 01 | 9.71 GiB | 4.71 | 5.00 |
| 02 | 8.78 GiB | 3.78 | 5.00 |
| 03 | 9.72 GiB | 4.72 | 5.00 |
| 04 | 9.84 GiB | 4.84 | 5.00 |

The reserve added to protect the capture is what **prevented** the capture. `min_free_kbytes` is a
floor the kernel withholds from *everyone*, vLLM included — it is not a page-cache-only reserve. The
agent's drop-cache loop already holds `Cached` under 1 GiB; a hard reserve is the wrong tool.

**Fixed: `min_free_kbytes` 5 GiB → 1 GiB** (`scripts/dgx-node-prep.sh`, `/etc/sysctl.d/90-dgx-dcp.conf`).

### Why it worked on 07-07 and not on 07-09

The memory note for the working config records *"node-prep min_free 1GiB"* — but the **script and the
persistent drop-in both contained 5 GiB.** The live value on 07-07 had been set by hand; the drop-in had
never been applied. **The nodes rebooted at 17:50 on 07-09 and `sysctl --system` armed it for the first
time.**

> A persistent sysctl that disagrees with the running value is a landmine set for the next reboot.

Later verified: after a deliberate reboot, `min_free_kbytes=1048576` on all four nodes. Defused.

### Dead ends (all disproven by measurement, not argument)

- Leftover containers — none running.
- gdm / the `maxoutmem` flag — gdm already `inactive`, default target `multi-user`.
- Page cache — the drop-loop fires every 500 ms; `Cached` ≈ 0.8 GiB.
- **Fragmentation / reboot** — fresh-boot free is **116.0–116.4 GiB**, *lower* than the 117.5 GiB after
  a clean teardown. A reboot buys nothing.
- Timing race — the failure was bit-for-bit deterministic across retries.

---

## 2. The manager was killing its own healthy deploys

A deploy cleared both walls, finished `torch.compile` on all ranks… and was torn down seconds later
with **zero errors in the vLLM log**. Two agent bugs, one incident:

1. **`dgxrun-dropcache.ts` used blocking `spawnSync("sync; echo 3 > drop_caches")` every 500 ms.**
   Under a ~400 GB NFS weight stream a single `sync` takes seconds — the kernel log shows drop_caches
   firing at **1–7 s gaps**, not 500 ms. `spawnSync` parks the agent's Node event loop for that whole
   time. Heartbeats stopped (`[staleness] node … marked offline`), and the agent starved its own
   docker probe.

2. **`inspectDgxrunContainer()` returned `null` on *any* non-zero exit** — including a 10 s timeout —
   and `dgxrun-metrics.ts` read `null` as *"container gone entirely"*, reporting `error: "container
   missing"`. The manager then tears down **every rank** (one dead rank hangs all).

A timed-out `docker inspect` is not evidence of absence.

**Fixed in agent 0.5.770** (`257b7d8`): async drop with an in-flight guard so a slow `sync` can never
queue behind itself; and a pure `classifyDockerInspect()` returning `found | absent | unknown` —
`unknown` skips the tick, `absent` must repeat twice. 16 new unit tests; full suite 903 green.

**Residual, verified:** the wedge is reduced, not eliminated. Staleness still fires during weight load
(`inspectDgxrunContainer`, `snapshotDgxrunLogs`, `captureCrashedDgxrunLogs` are all still `spawnSync`).
The deploy now **survives** it, because fix #2 decoupled the wedge from the teardown. Making the docker
inspect async is the clean next step (`checkDgxrunDeployments` is already `async`).

Same failure class as the earlier `cmd:update` incident: **never call `spawnSync` on the agent's hot path.**

---

## 3. The `gmu` coin-flip

While chasing something else, a deploy failed the startup guard at **109.41 GiB free vs 109.46 GiB
required — short by 0.05 GiB.** The shipped `gpu_memory_utilization: 0.90` was clearing the guard by
**under 100 MiB**. Every successful deploy had been near-luck.

`gmu` does **not** cap allocation here — `--kv-cache-memory-bytes` pins KV explicitly, and
`GPU KV cache size` is an identical 274,560 tokens at 0.88 and at 0.90. It is *purely* the guard
threshold. Lowering it is free margin.

**Fixed: `gmu` 0.90 → 0.88** (needs 107.03 GiB → ~2.4 GiB of headroom).

---

## 4. Chasing context: three wrong answers, then the right one

### The bisection (all kernel OOM-killed at capture)

| config | result |
|---|---|
| DCP2, KV 7.00 GiB, 262144 | **serves** |
| DCP2, KV 7.75 GiB, 294912 | kernel OOM |
| DCP2, KV 8.50 GiB, 327680 | kernel OOM |

Signature every time: **no torch OOM in the vLLM log**, deployment just reads `stopped`;
`dmesg -T` shows `Out of memory: Killed process … (VLLM::Worker_TP)`.

Wrong conclusion #1: *"256K is the ceiling; more context means trading decode speed."*

### DCP4 — a red herring

Community reference calls DCP4 the 640K config. Three attempts:

| attempt | result |
|---|---|
| dcp4, KV 7.00, FULL, 256K | kernel OOM at capture |
| dcp4, KV 4.00, FULL, 256K | free → **1.8 GB**, a rank stalled, head node fork-starved (`exec` *and* fork-free `diag` both failed) |
| dcp4, KV 7.00, **PIECEWISE**, 512K | torch OOM: `Tried to allocate 10.00 GiB` |

**None ever printed `GPU KV cache size`** — all died before KV allocation, so the DCP4 multiplier was
never measured. DCP4 costs *more* peak memory than DCP2 even with KV halved. And the image has **no
`VLLM_DCP_*` env at all** (`vllm.envs` enumerated), which is why `VLLM_DCP_SHARD_DRAFT=1` has always
logged as `Unknown vLLM environment variable`. DCP4 is unsupported in this build.

Wrong conclusion #2: *"the chunk-size env would bound it, but it doesn't exist in this image → needs an
image rebuild."* Also wrong — `sparse_mla_env.py` reads `envs.VLLM_TRITON_MLA_SPARSE_*`, which belongs
to vLLM's **Triton sparse MLA** backend. The DCP recipe runs `--attention-backend B12X_MLA_SPARSE`.
Porting those knobs would have bounded a code path we never execute.

### The actual answer: read the traceback

```
sparse_attn_indexer.py:1241  sparse_attn_indexer
  :986  _prewarm_b12x_paged_indexer_prefill
  :807  _run_b12x_paged_topk
    b12x/attention/indexer/paged.py:771  index_topk_fp8
      fold_indices = torch.empty(...)          <-- the 10.00 GiB
```

The b12x sparse-indexer **prewarm** allocates

```
fold_values + fold_indices = (profile_q_rows × total_slices, topk) × 8 bytes
```

with `profile_q_rows = max_num_batched_tokens` and `total_slices ∝ max_model_len`. So:

> **workspace ∝ max_num_batched_tokens × max_model_len**

Not the KV cache. Not the DCP degree. Not the image.

Checked against the data before touching anything: `q_rows=2048, topk=2048, 160 slices` →
**5.10 GiB** predicted vs **5.00 GiB** observed at 256K, and **10.00 GiB** observed at 512K. The model
reproduces both.

| mnbt | max_model_len | workspace |
|---|---|---|
| 2048 | 262144 | 5.00 GiB (observed) |
| 2048 | 524288 | 10.00 GiB (observed) |
| 2048 | 327680 | 6.25 GiB → OOM |
| **1024** | **327680** | **3.12 GiB → fits** |

**The lever was in the recipe all along: `--max-num-batched-tokens 2048 → 1024`.** Halving it halves
the workspace, freeing ~1.9 GiB — more than the +1.50 GiB the larger KV needs.

---

## 5. 320K, shipped and validated

`max_model_len 327680`, `--kv-cache-memory-bytes 9126805504` (8.50 GiB → **333,440 tokens**),
`--max-num-batched-tokens 1024`. **DCP2, tp=4, cudagraph FULL and the separate MTP drafter all
unchanged — no speed knob traded, no image rebuild.**

Validated rather than asserted:

- **300,036-token prompt** (92% of the window), needle buried at 73% depth → returned `VIOLET-HERON-77`,
  `finish: stop`, 893 tok/s prefill
- **210,034-token prompt** → `CRIMSON-OTTER-42`, correct
- Free memory never dipped below 1.73 GiB during either — the runtime uses the *reserved* indexer
  scratch, so long prefills don't spike (only the prewarm does)

**Cost:** ~11% prefill on short prompts (619.5 vs ~698 tok/s at pp=8192). **Decode unchanged**
(23.8 vs 22.3–23.2 tok/s). Long-prompt aggregate prefill: 893–959 tok/s.

Steady free while serving: **~1.7–1.9 GiB** (was ~2.9 GiB at 256K).

**To go further:** halve `mnbt` again (512 → 1.56 GiB workspace at 320K) and spend the savings on KV.
Each step buys context and costs prefill. It's arithmetic now, not bisection. Rate: **39,223 KV tokens/GiB.**

---

## 6. What generalises

1. **A persistent config that disagrees with the running value is a time bomb.** It detonates at the
   next reboot, arbitrarily far from the change that planted it.
2. **A safety reserve can cause the failure it was meant to prevent.** `min_free_kbytes` protected page
   cache by withholding memory from the application.
3. **"Unknown" is not "absent."** A timed-out probe told the manager a healthy container was gone, and
   it killed four ranks. Encode the third state.
4. **Never block the event loop of the process that reports health.** Blocking `spawnSync` made the
   agent look dead and made its own probes time out.
5. **Read the traceback before rebuilding the world.** Two hours of DCP4 and a planned image rebuild
   were avoided by four stack frames and a `sed -n` into the container.
6. **Model the allocation, then check it against the observation.** `20.0 KB/token` predicted both the
   5.00 GiB and 10.00 GiB numbers exactly — that's what made the one-line fix obvious and safe.
7. **A `max_model_len` you haven't exercised is a claim, not a capability.** Needle-in-a-haystack at
   92% window occupancy is the difference.

---

## Appendix: the numbers

| quantity | value |
|---|---|
| unified pool / node | 121.63 GiB (127,535,272 kB) |
| weights / rank | 93.9 GiB (377.6 GiB ÷ tp4) |
| CUDA-context reserve (3 procs, invisible to `/proc`) | ~7 GiB |
| cudagraphs + activations + contexts | ~14 GiB |
| KV rate | 39,223 tokens/GiB |
| indexer prewarm workspace | 20.0 KB/token × (mnbt ÷ 2048) |
| `gmu` guard requirement | `gmu × 121.63 GiB` of CUDA-visible free |
| DCP2 baseline (pp=8192, c=1) | prefill 686.7 tok/s, decode 24.7 tok/s |
| 320K (mnbt 1024) | prefill 619.5 tok/s, decode 23.8 tok/s |
