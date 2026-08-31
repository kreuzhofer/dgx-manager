# Which Nebius shape runs a GLM-5.3-Flash parameter-efficient tune, and what does the port cost?

> Research for [#69](https://github.com/kreuzhofer/dgx-manager/issues/69). Part of the
> [GLM-5.3-Flash wayfinder map](https://github.com/kreuzhofer/dgx-manager/issues/65).
>
> **Checkpoint sizes measured against the HuggingFace API on 2026-08-31.**
> **Nebius facts read from `docs.nebius.com`, `nebius.com/prices` and `github.com/nebius` on
> 2026-08-31, 20:20–20:45 UTC**, cross-checked against Nebius's own Soperator Terraform
> (`soperator-tf 2.0.2-1`). Where the two disagree, the docs win and the disagreement is noted.
>
> **Cost is not a deciding variable on this map.** Daniel has employee Nebius access with
> effectively no compute limit and data governance is settled. Prices appear below only as
> incidental context — they are never the argument. [#49](https://github.com/kreuzhofer/dgx-manager/issues/49)'s
> cost model is not re-run here and its conclusion does not transfer.
>
> Every number is labelled **measured**, **derived** or **extrapolated**.

---

## Recommendation

**One `gpu-h200-sxm` / `8gpu-128vcpu-1600gb` VM in `eu-north1`, self-service, no cluster.**

That single sentence carries four decisions, in order of weight:

1. **H200, not H100.** BF16 GLM-5.3-Flash is **598.5 GiB** (measured — `zai-org/GLM-5.3-Flash-BF16`,
   120 shards, 642,652,070,880 bytes). An 8×H100 node is 640 GiB of HBM. The weights *fit*, with
   **5.2 GiB per GPU left over** — which is not enough to train anything. 8×H200 is 1128 GiB and
   leaves **64 GiB per GPU**. The issue called 8×H100 "genuinely marginal"; the measured number
   makes it worse than marginal — it is a shape that can hold the model and cannot train it.
2. **One node, not a cluster.** 8×H200 turns this into a single-node job:
   `torchrun --nproc_per_node=8`. That deletes the entire multi-node surface — InfiniBand fabric,
   GPU clusters, Soperator, hostfile rank discovery, the `entrypoint.sh` sshd hack — which is
   most of what [#49](https://github.com/kreuzhofer/dgx-manager/issues/49) listed as
   Spark-specific porting work. **Picking the right shape is worth more than any porting effort
   saved elsewhere.**
3. **`eu-north1`, because all Nebius Blackwell is outside the EU.** H100 and H200 are the only
   GPU platforms Nebius offers in an EU region. B200 is `us-central1` / `me-west1`; B300 is
   `uk-south1`. Governance is settled on this map, but residency is still a free win.
4. **sm_90, not sm_100/sm_103.** H100 and H200 are the most-exercised training target in the
   ecosystem. B200/B300 are newer silicon whose third-party kernel support (Liger, flash-attn,
   fused-MoE, quantised-base kernels) is the exact class of thing that costs days.
   B200/B300 buy roughly **2× the BF16 throughput** and are worth having as the escape hatch,
   not as the first attempt.

Default quota in `eu-north1` is **32 H200 GPUs** — four such nodes — so this shape needs no quota
request at all. By contrast **B200's default quota is 0 in both its regions**, so the Blackwell
escape hatch costs a support ticket before it costs anything else.

**The access friction for this shape is essentially zero.** A single 8-GPU node needs no GPU
cluster (NVLink is intra-node), so the documented prerequisite list is *"Generate an SSH key
pair"* and nothing else. And #49's belief that multi-node InfiniBand was sales-gated turns out to
be **wrong** — see §3. What is gated is the *managed* Slurm product, not the fabric.

The **fallback** if BF16 turns out not to be the method — because
[#67](https://github.com/kreuzhofer/dgx-manager/issues/67) finds no framework can build a
trainable `Glm5NextForConditionalGeneration` in BF16, or
[#71](https://github.com/kreuzhofer/dgx-manager/issues/71) finds a quantised base is required —
is **8×H100**, which has ample room for the 305.8 GiB FP8 or 181.3 GiB NVFP4 checkpoints. The
constraint then moves from hardware to framework support, which is not a Nebius question.

### The thing that actually costs wall-clock

For a first probe, **the GPU is not the bottleneck**. Derived and extrapolated, per job:

| Fixed cost | Estimate | Basis |
|---|---:|---|
| Pull 642.7 GB from HuggingFace into Nebius | **~20–90 min** | *unknown* — no documented VM egress/ingress bandwidth; see [#70](https://github.com/kreuzhofer/dgx-manager/issues/70) |
| Load the checkpoint from shared FS into 8 GPUs | **~19 min** | **measured precedent**: the prior-art demo loaded a 470 GB, 118-shard Qwen3-235B-A22B across 16×H100 from the same Nebius shared filesystem in **~14 min** |
| A 15 M-token probe run (scenario S) | **~17 min** | derived, §4 |

**A first probe spends more time moving the model than training it.** That makes the *access
path* and the *storage layout* the wall-clock levers, not the GPU model — and it argues for one
long-lived VM with the checkpoint already staged on a generously-sized shared filesystem, rather
than a cluster torn down between runs. (There is no local NVMe on H100/H200/B200 — see §1.)

---

## 1 — What Nebius actually offers

Read from `docs.nebius.com/compute/virtual-machines/types`,
`docs.nebius.com/compute/clusters/gpu`, `docs.nebius.com/overview/regions` and
`docs.nebius.com/compute/resources/quotas-limits` on **2026-08-31, 20:20–20:35 UTC**, and
cross-checked against Nebius's own Soperator Terraform
(`soperator/modules/available_resources/{platform,preset,gres,region}.tf`, `soperator-tf 2.0.2-1`,
pinned in `/mnt/tank/src/github/nebius-slurm-ml-training-and-inference-demo`). The two agree.

### GPU platforms relevant to this workload

| Platform ID | GPU | HBM/GPU | Mem BW | 8-GPU preset | Front-end NIC | Regions |
|---|---|---:|---:|---|---|---|
| `gpu-h100-sxm` | H100 SXM (HBM3) | **80 GB** | 3.35 TB/s | `8gpu-128vcpu-1600gb` | ConnectX-6, 100 Gbps | `eu-north1` |
| `gpu-h200-sxm` | H200 SXM (HBM3e) | **141 GB** | 4.8 TB/s | `8gpu-128vcpu-1600gb` | BlueField-3, 200 Gbps | `eu-north1`, `eu-west1`, `us-central1`, `eu-north2`\* |
| `gpu-b200-sxm` | B200 SXM (HBM3e) | **180 GB** | 8 TB/s | `8gpu-160vcpu-1792gb` | BlueField-3, 400 Gbps | `us-central1` |
| `gpu-b200-sxm-a` | B200 SXM | **180 GB** | 8 TB/s | `8gpu-160vcpu-1792gb` | BlueField-3, 400 Gbps | `me-west1` |
| `gpu-b300-sxm` | B300 SXM (Blackwell Ultra) | **288 GB** | 10 TB/s | `8gpu-192vcpu-2768gb` | BlueField-3, 400 Gbps | `uk-south1`, `eu-west2`\*, `us-north1`\* |

\* private regions — existing deployments only.

Also offered but not viable here: `gpu-rtx6000` (RTX PRO 6000, 96 GB GDDR7, **PCIe — no NVLink,
no InfiniBand**) and `gpu-l40s-a` / `gpu-l40s-d` (48 GB, PCIe, no IB).

**There is no 94 GB H100 NVL and no 192 GB B200 on Nebius.** H100 is 80 GB only; B200 is 180 GB
only.

Four structural facts fall out, and all four are decision-relevant:

- **On SXM you rent 1 GPU or 8 GPUs. Nothing in between.** Every SXM platform offers exactly two
  presets: `1gpu-*` and `8gpu-*`. (Only the PCIe `gpu-l40s-d` has 2- and 4-GPU shapes, and it has
  no InfiniBand.) The unit of serious multi-GPU work on Nebius is a whole 8-GPU node.
- **InfiniBand exists only inside a GPU cluster, and only for 8-GPU presets.** Docs, verbatim:
  *"Other presets and platforms are not compatible with GPU clusters"*, and *"You can assign a
  GPU cluster only when creating a VM"* — you cannot retrofit IB onto a running VM. Nebius's own
  Terraform agrees: in `preset.tf` every 1-GPU and CPU preset is
  `gpu_cluster_compatible = false`; only the three 8-GPU presets are `true`.
- **`eu-north1` (Finland) is still the only H100 region, and there is still no German region.**
  [#49](https://github.com/kreuzhofer/dgx-manager/issues/49)'s finding holds unchanged.
  **All Nebius Blackwell is outside the EU** — B200 in `us-central1` / `me-west1`, B300 in
  `uk-south1`. Inside the EU you get H100 and H200 and nothing else.
- **Default quota is zero on some platforms that show a tick in the region matrix.** A region flag
  is not permission to launch.

### Default GPU quotas — the friction the region table hides

From `docs.nebius.com/compute/resources/quotas-limits`:

| Platform | Default quota |
|---|---|
| H100 | **32** in `eu-north1` (= four 8-GPU nodes) |
| H200 | **32** in `eu-north1`, **8** in `eu-west1`, **0** in `us-central1` |
| B200 | **0** in both its regions |
| B300 | **32** |
| RTX PRO 6000 | 0 |
| L40S | 2 |
| **GPU clusters** | **5 per region** (0 in `eu-north2`) |

So `eu-north1` H200 at quota 32 admits the recommended shape *and* three more nodes with no
request at all — while **B200 needs a quota increase before it can be launched anywhere.**

### InfiniBand fabrics, by platform and region

Verbatim from `docs.nebius.com/compute/clusters/gpu`:

| Fabric | Platform | Region |
|---|---|---|
| `fabric-2`, `-3`, `-4`, `-6` | `gpu-h100-sxm` | `eu-north1` |
| **`fabric-7`** | **`gpu-h200-sxm`** | **`eu-north1`** |
| `fabric-5` | `gpu-h200-sxm` | `eu-west1` |
| `us-central1-a` | `gpu-h200-sxm` | `us-central1` |
| `us-central1-b` | `gpu-b200-sxm` | `us-central1` |
| `me-west1-a` | `gpu-b200-sxm-a` | `me-west1` |
| `uk-south1-a` | `gpu-b300-sxm` | `uk-south1` |

Docs, verbatim on aggregate bandwidth: *"Each GPU in a VM is connected through a network
interface card (NIC) that provides 400 Gbps. As a compute VM for GPU clusters consists of 8 GPUs,
the total bandwidth for a node is 3.2 Tbps."* GPUDirect RDMA is used. B300 is 800 Gbps per GPU
(ConnectX-8).

Soperator additionally ships fabric health checks — `active_checks_scope = "prod_quick"` runs
all-reduce, IB-bandwidth and CUDA benchmarks on every node before accepting work. The prior-art
demo measured this at **~10 minutes on H100 nodes**.

### Host resources per 8-GPU node

| Platform | vCPU | Host RAM | GPU HBM (node) | Local NVMe |
|---|---:|---:|---:|---|
| `gpu-h100-sxm` | 128 | 1600 GiB | **640 GiB** | **none** |
| `gpu-h200-sxm` | 128 | 1600 GiB | **1128 GiB** | **none** |
| `gpu-b200-sxm` / `-a` | 160 | 1792 GiB | **1440 GiB** | **none** |
| `gpu-b300-sxm` | 192 | 2768 GiB | **2304 GiB** | 6 × 3.84 TB, `uk-south1` only |

**Local NVMe is essentially not offered.** Per
`docs.nebius.com/compute/storage/local-disks#availability`, local SSD exists **only** on
`gpu-b300-sxm` + `8gpu-192vcpu-2768gb`, and Nebius's Terraform allow-list narrows that further to
`uk-south1` alone. Every other GPU shape's scratch tier is network storage. This corrects a
natural misreading of the prior-art demo, whose `/mnt/local-data` "1 TB local SSD" was a
Soperator-carved network disk, not host NVMe. **Plan storage accordingly — see §6.**

Host RAM matters more than usual here: a 642.7 GB checkpoint passes through page cache on its way
to the GPUs. 1600 GiB comfortably holds one copy and nowhere near eight.

*NVIDIA's HBM "GB" figures are effectively GiB (H200's "141GB" reports as 143,771 MiB ≈ 140.4 GiB
under `nvidia-smi`). All fit arithmetic below uses nameplate GiB minus a flat 2 GiB/GPU reserve
for CUDA context and allocator slack — an explicit, conservative assumption, not a measurement.*

### Anything newer than B200

| Offering | Status |
|---|---|
| **HGX B300** (`gpu-b300-sxm`) | **GA and self-service**, public in `uk-south1`. Nebius's own page: *"launch GPU instances… all without contacting sales"* |
| **GB300 NVL72** | Live (Nebius claims Europe's first deployment) but **sales-gated** — priced "Contact us" |
| **`gpu-gb300`** | **An undocumented platform ID present in Nebius's own Terraform but absent from the docs**: `cpu_platform = arm64` (Grace), `eu-north1`, single preset **`4gpu-112vcpu-800gb`**, `gpu_cluster_compatible = true`. Note the implications — a **4-GPU** SXM shape, in the **EU**, on **arm64**. If this ever becomes self-service it revives the arm64 container question #49 retired |
| GB200 NVL72 | Sales-only and being de-emphasised — still on the price list, no longer on `nebius.com/compute`, no platform ID in docs or Terraform |
| Vera Rubin NVL72 | **Announced only**, "starting H2 2026" |
| "AI Cloud 3.0 Aether" | A compliance/platform release (SOC 2 Type II, HIPAA, GDPR, ISO 27001), **not hardware** |

**Contradiction to flag:** B300 memory is stated as **288 GB** in the docs, **270 GB** on
`nebius.com/compute/b300`, and **279 GB** on the GB300 page. All three are Nebius-primary and
mutually inconsistent. Not resolvable without a console.

### Prices — incidental context only

Per-GPU-hour on-demand, verbatim from `nebius.com/prices` (2026-08-31; USD only, and **no region
is stated — prices appear to be global**): H100 $3.85, H200 $4.50, B200 $7.15, B300 $7.85.
Preemptible: $2.15 / $2.45 / $3.95 / $4.30. GB200 and GB300 NVL72 are "Contact us". Billing
granularity is 1 second. An 8×H200 node is ~$36/hr. Unchanged from #49's 2026-08-30 check.

**This is not an input to the decision.** It is recorded so a future reader does not mistake the
absence of a cost section for an oversight.

---

## 2 — Which shapes fit

### The three checkpoints (measured, HuggingFace API, 2026-08-31)

| Checkpoint | Format | Shards | Bytes | GiB |
|---|---|---:|---:|---:|
| `zai-org/GLM-5.3-Flash-BF16` | BF16 | 120 | 642,652,070,880 | **598.5** |
| `zai-org/GLM-5.3-Flash` | FP8 e4m3, block-quantised | 62 | 328,337,455,672 | **305.8** |
| `LibertAIDAI/GLM-5.3-Flash-NVFP4` | NVFP4 (modelopt) | 121 | 194,665,046,744 | **181.3** |

An official BF16 sibling **does exist** — `zai-org/GLM-5.3-Flash-BF16`, updated 2026-08-31. The
`zai-org/GLM-5.3-Flash` repo everyone calls "the model" is *already quantised* (FP8 e4m3 with
`weight_block_size`, `quant_method: "fp8"`). That is worth knowing before anyone tries to LoRA
"the base model". The full component-level breakdown belongs to
[#66](https://github.com/kreuzhofer/dgx-manager/issues/66); only the byte counts are used here.

### Fit table

Per-GPU headroom after an 8-way weight shard and a 2 GiB/GPU reserve. **Derived.**
(The Spark row is per *node*, in decimal GB, because a GB10's memory is unified and pooled
across 4 nodes = 498 GB.)

| Shape | HBM/GPU | Weights/GPU (BF16) | Headroom (BF16) | Headroom (FP8) | Headroom (NVFP4) |
|---|---:|---:|---:|---:|---:|
| 8 × H100 | 80 GiB | 74.8 GiB | **3.2 GiB** | 39.8 GiB | 55.3 GiB |
| **8 × H200** | 141 GiB | 74.8 GiB | **64.2 GiB** | 100.8 GiB | 116.3 GiB |
| 8 × B200 | 180 GiB | 74.8 GiB | **103.2 GiB** | 139.8 GiB | 155.3 GiB |
| 8 × B300 | 288 GiB | 74.8 GiB | **211.2 GiB** | 247.8 GiB | 263.3 GiB |
| 16 × H100 (2 nodes) | 80 GiB | 37.4 GiB | **40.6 GiB** | 57.5 GiB | 66.7 GiB |
| *4 × DGX Spark (GB10)* | *124.5 GB/node* | *160.7 GB/node* | **✗ over by 36 GB/node** | *42.4 GB/node free* | *75.8 GB/node free* |

### How much headroom a PEFT tune needs

Derived for `seq=16384`, micro-batch 1, gradient checkpointing on, adapters on attention
projections only (**not** `all-linear` — see below):

| Component | GiB/GPU @16k | Note |
|---|---:|---|
| Checkpointed layer-boundary activations | 5.6 | 45 layers × 16384 × 4096 × 2 B |
| Recompute peak inside one block | ~4 | MoE top-8 expert intermediates dominate |
| Loss head | ~1 *with* Liger fused CE | **~14 GiB without it** — vocab 154,880 × 16384, upcast to fp32 |
| Adapters + AdamW state | <1 | attention-only targets at r=32 |
| NCCL buffers + allocator fragmentation | 4–8 | |
| **Total** | **≈ 15–20** | **≈ 25–30 at `seq=32768`** |

**Verdicts:**

| Shape | LoRA on BF16 @16k | LoRA on BF16 @32k | Over FP8 base | Over NVFP4 base |
|---|---|---|---|---|
| 8 × H100 | ✗ **short by ~15 GiB/GPU** | ✗ | ✓ on memory | ✓ on memory |
| **8 × H200** | ✓ **~3× margin** | ✓ ~2× margin | ✓ | ✓ |
| 8 × B200 | ✓ | ✓ | ✓ | ✓ |
| 8 × B300 | ✓ | ✓ | ✓ | ✓ |
| 16 × H100 | ✓ but needs cross-node EP/ZeRO | ✓ | ✓ | ✓ |
| 4 × DGX Spark | ✗ weights alone exceed 498 GB pooled | ✗ | ✓ on memory, but only 42 GB/node spare | ✓ on memory, 76 GB/node spare |

"✓ on memory" is not "✓". Whether anything can *train* over an FP8- or NVFP4-quantised
`glm5_next` is [#67](https://github.com/kreuzhofer/dgx-manager/issues/67)'s question, and the
per-method budget is [#71](https://github.com/kreuzhofer/dgx-manager/issues/71)'s. This table
answers only "does the hardware admit it".

### Two traps the table hides

1. **`all-linear` LoRA targeting is fatal here, in every venue.** 42 MoE layers × 288 experts ×
   3 projections at r=32 is **~14 GiB of adapter parameters** before optimizer state — larger
   than the entire trainable surface of the Qwen3.8-27B run by three orders of magnitude. Target
   attention projections and the router explicitly, never `all-linear`. *(Derived from
   `config.json`: `n_routed_experts: 288`, `moe_intermediate_size: 2048`, `hidden_size: 4096`.)*
2. **[#49](https://github.com/kreuzhofer/dgx-manager/issues/49)'s "use ZeRO-1, not ZeRO-3"
   conclusion is void here.** That rested on the weights fitting one device. They do not: a
   642.7 GB checkpoint cannot be replicated per rank against 1600 GB of host RAM, let alone
   80–141 GiB of HBM. **Sharded loading is mandatory** — ZeRO-3 / FSDP / expert-parallel, with
   a sharded `from_pretrained`. Do not carry `"stage": 1` across from the Qwen3.8 recipe.

---

## 3 — Access paths and their friction

Checked against `docs.nebius.com` and `github.com/nebius` on **2026-08-31, 20:20–20:35 UTC**
(the docs changelog is current through the week of 2026-08-24).

### #49's finding is half right, and the wrong half is the important one

**Confirmed, unchanged:** Managed Soperator GPU worker nodes still require capacity block groups,
and capacity block groups are still not self-service. Four independent statements say so:

> *"**GPU worker nodes in Soperator are only available if you have capacity block groups that
> reserve GPUs.**"* — `/slurm-soperator/deploy/overview`

> Worker node set fields include *"**Reservation ID from your capacity block group.**"* —
> `/slurm-soperator/managed-soperator/manage`

> *"The quotas on GPUs by type for regular VMs without reservations don't apply to Managed
> Soperator nodes **because they require capacity block groups that reserve GPUs**."* —
> `/slurm-soperator/managed-soperator/resources/quotas`

> *"**To reserve GPUs by using capacity block groups, send a request to your Nebius manager. They
> create capacity block groups for you and prepare an addendum for commitment discounts
> billing.**"* — `/overview/limits/capacity-block-groups`

There is also a **structural** proof, which is stronger than any prose: the
`capacity-block-group` API exposes only `get`, `get-by-resource-affinity`, `list` and
`list-resources`. There is **no create/update/delete** in the CLI, the REST API, or Terraform,
where it exists only as a *data source*. You cannot make one yourself even if you want to.

*(Note the contradiction: `nebius.com/services/soperator` markets it as
*"Sign up for the console, add billing details and set up your cluster parameters. That's it!"*
and never mentions capacity blocks. Believe the docs.)*

**Wrong, and this is what changes the answer:** #49 concluded that *"the self-service path on
default quota is a plain 8-GPU VM"*, implying multi-node InfiniBand was sales-gated. It is not.

> *"**If you use the web console, you don't need to complete any prerequisites.**"* —
> `/compute/clusters/gpu` (on creating a GPU cluster, i.e. an InfiniBand fabric)

> *"**You do not need to complete any prerequisites if you create or modify node groups in the
> web console.**"* — `/kubernetes/node-groups/manage`

On Managed Kubernetes the reservation field is *"(Optional)"* and is *"only displayed if you have
capacity block groups"*; the docs spell out the alternative — *"**Without reservations**: The
resources are allocated from a common pool."* In Terraform, `mk8s_v1_node_group` requires only
`template.resources`; `gpu_cluster` and `reservation_policy` are both optional. And Nebius's own
`nebius_compute_v1_gpu_cluster` resource has exactly **one** required field,
`infiniband_fabric` — there is no reservation argument on it at all.

The open-source Soperator Terraform agrees: in
`nebius-solutions-library/soperator/installations/example`, `reservation_policy` is
`optional(...)` with no default and is **commented out** in the shipped `terraform.tfvars`,
labelled *"Use reservation_policy to leverage compute reservations (capacity blocks)"*. The
clearest first-party statement in the whole GitHub org is in `nebius-physical-ai`:
`capacity_block_group` — *"Optional capacity block group ID… **Leave empty for on-demand
capacity.**"*

**So: what is sales-gated is the *managed* Slurm product, not multi-node InfiniBand.**

### The paths

| Path | Self-service? | Needs | Gets you | IB | Multi-node |
|---|---|---|---|---|---|
| **Compute GPU VM (PAYG)** | ✅ | **an SSH key** | 1 × 8-GPU VM | via GPU cluster | up to quota |
| **GPU cluster (IB fabric)** | ✅ | nothing (console) | P-key-isolated IB fabric, 3.2 Tbps/node | ✅ | ✅ |
| **Managed Kubernetes GPU node group** | ✅ | nothing (console) | 8-GPU nodes + topology labels | ✅ | ✅ |
| **Serverless AI job** (`nebius ai job create`) | ✅ | account + CLI | 1 container, ≤8 GPUs. **Official Axolotl + LoRA fine-tuning tutorial exists** | ❌ | ❌ |
| Standalone apps (JupyterLab, SkyPilot) | ✅ | nothing | 1 VM, ≤8 GPUs | ❌ | ❌ |
| **Managed Soperator (Managed Slurm)** | ❌ | **capacity block group → Nebius manager + commitment addendum** | managed Slurm | ✅ | ✅ |
| Self-deployed Soperator on MK8s | ✅ | Terraform + CLI + kubectl + jq | Slurm-on-K8s you operate | ✅ | ✅ |
| Pro Solution for Soperator | ❌ | sales | expert-run Slurm | ✅ | ✅ |
| NVL instance groups (GB200/GB300) | ❌ | sales; no public platform ID | rack-scale NVLink | NVL72 | ✅ |
| Nebius Token Factory | ✅ | OAuth + API key | inference **and** fine-tuning API — **you get a model, not GPUs** | n/a | n/a |

Managed Soperator carries a **second, independent** friction beyond the capacity block: it has
**no CLI, REST or Terraform surface at all** (only `msp/mlflow` and `msp/postgresql` exist). It is
web-console-only.

### The fastest route from "account exists" to "a job runs"

**For the recommended shape, the friction is essentially zero.** A single 8-GPU node needs no GPU
cluster at all — NVLink is intra-node; a GPU cluster only buys inter-node InfiniBand. So the whole
path is:

1. Generate an SSH key. *(The quickstart lists this as the only prerequisite.)*
2. Create one `gpu-h200-sxm` / `8gpu-128vcpu-1600gb` VM in `eu-north1`. Default quota is 32 H200
   GPUs there — no request, no ticket, no call.
3. Create and attach a shared filesystem. Pull the checkpoint. Run `torchrun --nproc_per_node=8`.

**If multi-node is ever needed** (it should not be — see §2), the self-service route is *Managed
Kubernetes*, not Soperator. Nebius's own `/kubernetes/gpu/nccl-test` page walks the whole thing
with no reservation anywhere:

```bash
nebius compute gpu-cluster create --name k8s-gpus --infiniband-fabric fabric-3
nebius mk8s cluster create --name nccl --control-plane-version 1.35 ...
nebius mk8s node-group create --fixed-node-count 2 \
  --template-resources-platform "gpu-h100-sxm" \
  --template-resources-preset "8gpu-128vcpu-1600gb" \
  --template-gpu-cluster-id $GPU_CLUSTER_ID
```

That is 16 GPUs on an InfiniBand fabric, inside the default 32-GPU quota, with no sales contact.
**It also means the prior-art repo is more reusable than #49 concluded** — the open-source
Soperator recipe can be laid on top of a self-service MK8s cluster; only the *managed* product is
gated.

### Worth knowing before relying on any of this

- **Serverless AI jobs may be the fastest path of all for exactly our shape.** It is single-node
  and ≤8 GPUs — which is precisely the recommendation — it uses Compute quotas with no separate
  allocation, and Nebius ships an official **Axolotl + LoRA fine-tuning tutorial** on it. What is
  *not* documented is how a 642.7 GB checkpoint persists between job runs, and
  `nebius ai job create` takes only `--platform`/`--preset` with no GPU-cluster parameter, so it
  is structurally single-node forever. **Worth 30 minutes of evaluation on #70.**
- **PAYG GPU capacity is not sticky.** Docs: *"Without reservations, GPU capacity is taken from a
  shared pool and returned when a VM is stopped (for example, by you or a maintenance event)."*
  You may not get it back. This independently reinforces §6's "keep the VM" conclusion — and it
  means a maintenance event can take the node away mid-campaign. Maintenance can be deferred
  twice, 7 days each, via support.
- **Quota is not capacity.** *"Data provided by the capacity advisor is accurate as of a specific
  timestamp… and doesn't guarantee availability of GPU resources at creation time."*
- **Onboarding friction is low.** OAuth signup (Google/GitHub/Microsoft), a tenant and a
  per-region project auto-created, card billing with a **$25 minimum first payment**. **No
  documented KYC hold, no approval gate, no zero-until-approved quota.** No free tier; the
  startup programme requires *"at least $5M+ USD from one of our approved VC partners"*.
- **Nebius's own agent skill sits in tension with the published quota table.** Its comment reads
  `preemptible: true  # on-demand GPU quota is often 0; preemptible works`. That skill is
  B200-oriented, and B200 / H200-in-`us-central1` / RTX-PRO-6000 defaults genuinely *are* 0 — but
  H100 and H200 in `eu-north1` are 32. Do not let the blanket phrasing talk you out of checking.
  (Preemptible is the documented fallback — 8 VMs by default, ~44% cheaper — but preemption is
  exactly the wrong trade for a wall-clock-driven decision.)
- **Tooling** is healthy: `nebius` CLI, Go/Python/JS SDKs, REST + gRPC, and
  `nebius/terraform-provider-nebius` at **v0.6.49 published 2026-08-31**, cut roughly daily.
  Managed Soperator is the conspicuous hole — its `cluster_service.tfgen.go` is *named* in the
  provider's managed-files manifest but absent from the published tree.
- **GB300 NVLink is a capacity gate, not a tooling gate.** The open-source Soperator tfvars already
  carries `# nvlink = { enabled = true, type = "GB300" }` (*"Required for GB300 workers. This
  creates one NVLink instance group per node group"*), and `variables.tf` defaults its `type` to
  `"GB300"`. So NVL instance groups are wired; what you cannot self-serve is the capacity and the
  price ("Contact sales"). Irrelevant to the recommendation, but it means a future rack-scale run
  is a commercial conversation, not an engineering one.

---

## 4 — Wall-clock, first order

The map says wall-clock decides. This is the roofline; the calibrated version belongs to
[#71](https://github.com/kreuzhofer/dgx-manager/issues/71).

**FLOP per training token.** Frozen base + adapters + gradient checkpointing → `6 × P_active`
(forward `2P`, input-gradient backward `2P`, recompute `2P`; weight gradients are skipped for the
frozen base). Summing the matmul-active parameters from `config.json` *(derived)*:

| Component | Active params |
|---|---:|
| MoE, 42 layers × (8 routed + 1 shared) × 3 × 4096 × 2048 | 9.51 B |
| KDA linear attention, 34 layers | ~4.8 B |
| Sparse MLA, 11 layers (`q_lora_rank` 1536, `kv_lora_rank` 512) | ~1.1 B |
| Dense MLP, 3 layers (`first_k_dense_replace: 3`) | 0.45 B |
| `lm_head` (vocab 154,880 × 4096) | 0.63 B |
| **Total active** | **≈ 16.5 B** |

That corroborates the map's working "~18 B active" independently. `6 × 17e9 ≈ 102` GFLOP/token;
attention adds little because 34 of 45 layers are O(n) linear attention and the other 11 are
top-2048 sparse. **Call it ~105 GFLOP/token at 16k.**

**The striking consequence: a PEFT tune of this 320B MoE costs *less* compute per token than the
27B dense model did.** [#49](https://github.com/kreuzhofer/dgx-manager/issues/49) measured
Qwen3.8-27B at 171 GFLOP/token. GLM-5.3-Flash is **~1.6× cheaper per token**. This problem is
entirely a capacity problem, not a FLOPs problem.

**Throughput.** At an assumed **20% MFU** — materially below the 35% used for dense models in
#49, because MoE training loses throughput to routing, all-to-all and small per-expert GEMMs
(**assumption, not measured on any fleet**):

| Shape | Dense BF16 peak | Time / 1M tokens |
|---|---:|---:|
| 8 × H100 or 8 × H200 | 7.92 PFLOPS | **~1.1 min** |
| 8 × B200 | 18 PFLOPS | **~0.5 min** |
| 8 × B300 | ~18 PFLOPS | **~0.5 min** |
| 4 × DGX Spark (NVFP4 base only, if it works) | ~0.5 PFLOPS, ~10% MFU | **~35 min** |

Against #49's corpus brackets, ×3 epochs — **extrapolated, low confidence, no MoE MFU has been
measured anywhere on either fleet**:

| Scenario | Tokens ×3 ep | 8×H200 | 8×B200 | 4 Sparks |
|---|---:|---:|---:|---:|
| S — 300 trajectories | 15 M | **17 min** | 8 min | 8.7 h |
| M — turn-level ×8 | 72 M | **1.3 h** | 36 min | 1.8 d |
| L — corpus grows | 180 M | **3.3 h** | 1.5 h | 4.4 d |
| XL — large corpus | 720 M | **13 h** | 6 h | 17.5 d |

Read this next to the fixed costs in the Recommendation: **at scenario S the ~19-minute
checkpoint load exceeds the ~17-minute training run.** The Sparks column is included for
completeness and is doubly hypothetical — it assumes a QLoRA-over-NVFP4 path that
[#67](https://github.com/kreuzhofer/dgx-manager/issues/67) has not yet shown to exist.

---

## 5 — The porting delta: what is *new* versus #49

[#49](https://github.com/kreuzhofer/dgx-manager/issues/49)'s audit of `lib/patches.py`,
`launch.sh`, `entrypoint.sh` and `packages/agent/src/runtime/finetune.ts` still holds and is not
repeated. Its verdict, in one line each:

- **Delete on Nebius:** `patch_pynvml`, `patch_safetensors_cache`, `flush_page_cache`, the
  dual-rail RoCE / `NCCL_IB_HCA` block, the `sm_121` NCCL build, hostfile rank discovery, the
  `entrypoint.sh` sshd hack, and the hardcoded `NCCL_SOCKET_IFNAME=enp1s0f0np0` in
  `packages/agent/src/runtime/finetune.ts`.
- **Keep — mislabelled as Spark-specific:** `patch_nvtx_dummy_domain`, the long NCCL timeout
  (lower the value, drop `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`), the PEFT torchao disable.
- **Void concern:** arm64. `nvcr.io/nvidia/pytorch:25.11-py3` is a multi-arch manifest.

### What a 320B multimodal MoE adds

| # | New item | Why it is new | Size |
|---|---|---|---|
| N1 | **Sharded loading is mandatory** | #49 concluded ZeRO-1 because the weights fit one device. 642.7 GB does not fit one device *or* one host. `ds_config.json` must stay sharded, and `from_pretrained` must load shard-wise, not per-rank. | Config, but reverses a #49 decision |
| N2 | **Expert parallelism** | 288 experts. ZeRO-3 treats experts as ordinary parameters and will all-gather all 288 every micro-step — correct, and catastrophically slow. A real EP implementation (FSDP2 + EP, DeepSpeed-MoE, or whatever [#67](https://github.com/kreuzhofer/dgx-manager/issues/67) selects) is required. **The largest unknown in this document.** | Framework-dependent; blocked on #67 |
| N3 | **`target_modules` must exclude the experts** | `all-linear` produces ~14 GiB of adapters. Needs an explicit allowlist and a `print_trainable_parameters()` assertion in CI. | Small, but a silent-failure class — cf. [#47](https://github.com/kreuzhofer/dgx-manager/issues/47) |
| N4 | **The MTP head must survive save/merge** | `num_nextn_predict_layers: 1` at layer 45. The manager's merge + quantize steps must not drop it. Venue-independent, but it lands in the same code path as the port. | Small; risk is silent |
| N5 | **The checkpoint is the data-movement problem** | #49's §4 concluded data movement was a non-issue at 0.1–6 GB. That was the *dataset*. The *model* is 642.7 GB. See §6. | Process change, not code |
| N6 | **`launch.sh` gets simpler, not harder** | On a single 8×H200 node the whole hostfile / rank-discovery / sshd / IB-interface block collapses to `torchrun --nproc_per_node=8`. This is the single largest porting saving available and it is bought by choosing the shape. | **Negative** effort |
| N7 | **Liger fused CE is load-bearing again** | Vocab 154,880 × 16384 tokens = ~14 GiB of fp32 logits without it. #49 flagged this at vocab 248,320; it does not go away at 154,880. Confirm Liger has a `glm5_next` patch, or write the fused head. | Medium if Liger lacks the model |
| N8 | **Blackwell kernel risk, if B200/B300** | NGC `pytorch:25.11-py3` is CUDA 13.0.2 / PyTorch 2.10.0a0 / TE 2.9 and has been "optimized for Blackwell" since 25.01 *(NVIDIA release notes)*, so the base image is fine. The risk is third-party kernels on sm_100/sm_103. **Choosing H200 (sm_90) avoids this entirely.** | Zero on H200; unknown on Blackwell |
| N9 | **`patch_safetensors_cache`'s motivation changes but does not vanish** | #49 deleted it as a unified-memory workaround. On Nebius, 8 ranks streaming a 642.7 GB checkpoint through 1600 GB of page cache is a genuine pressure, just a different one. Keep the mechanism available; do not assume it is dead. | Small |
| N10 | **Servability round-trip** | The artifact has to come home as NVFP4 to be served on our fleet. Merging into a 598.5 GiB BF16 base and re-quantising needs ~1.3 TB of scratch and a GPU. Do it **on Nebius**, ship back only the ~181 GiB result. | Medium; overlaps `research/nvfp4-merge-path` |
| N11 | **The shared-FS dataset-cache race is real, and #49 called it right** | The prior-art demo hit exactly the failure `keep_in_memory=True` guards against: 16 ranks racing HuggingFace `datasets` Arrow cache files on shared NFS, with `main_process_first()` *not* being sufficient. Their fix was `datasets.disable_caching()`. Independent corroboration that #49's "keep it" verdict was correct — and their better suggestion (a separate single-node preprocess job with a Slurm dependency) is worth adopting. | Small; already-known-good |

### Unchanged and still the critical path

`lib/dataset.py` still has **no image path at all** — no `AutoProcessor`, no `pixel_values`;
`train.py` loads `AutoModelForCausalLM`. Render screenshots cannot reach the model in *either*
venue. Tracked on [#55](https://github.com/kreuzhofer/dgx-manager/issues/55). #49 called this the
critical path and it still is: **do not spend the porting budget on the venue before this
exists.**

---

## 6 — Data movement at 182–643 GB

**Pull from HuggingFace inside Nebius. Never push from here.**

| Route | Bytes | Time |
|---|---:|---|
| Push BF16 from `/mnt/tank` over a ~40 Mbit/s domestic upstream | 642.7 GB | **~36 h of saturated upstream** — and repeated per checkpoint variant |
| Pull BF16 from HuggingFace onto a Nebius VM | 642.7 GB | **~20–90 min** *(unmeasured — see below and #70)* |
| Pull NVFP4 from HuggingFace onto a Nebius VM | 194.7 GB | ~6–27 min, same caveat |
| Upload the training corpus | 0.1–6 GB *(#49's bracket, blocked on [#50](https://github.com/kreuzhofer/dgx-manager/issues/50))* | ~20 min |
| Bring the tuned NVFP4 artifact home | ~194.7 GB | *unmeasured downstream link* — see #70 |

The asymmetry is total: the corpus is ~0.5% of the model. #49's data-movement section answered a
question that no longer exists — **the model is the payload now.**

### Cost of bytes: effectively zero, in both directions

Confirmed on `docs.nebius.com/object-storage/resources/pricing`,
`docs.nebius.com/vpc/resources/pricing` and `nebius.com/prices`, 2026-08-31:

- **Ingress is free everywhere** — internet → VM and internet → Object Storage. The Object Storage
  price page has no ingress line item at all.
- **VM egress to the internet is free**; the whole Virtual Networks service is free of charge.
- **Object Storage egress is charged** ($0.015/GiB on Standard) — but **bucket → VM within the
  same region is free**. Keep the bucket and the GPUs in the same region and the bytes cost $0.
- Cross-region or internet egress of a 650 GB checkpoint from Standard storage would be ~$9.75.

### The undocumented number that matters most

**Nebius publishes no per-VM internet ingress bandwidth figure and no throttling cap.** The only
per-flow throughput number anywhere in the docs is a *tuning target*:
`docs.nebius.com/compute/virtual-machines/tcp-window-tuning` sizes buffers for
*"a single flow of about 3 Gbit/s over a 300 ms round-trip-time path"*. That is not a platform
cap, but it is a strong hint that a single-stream long-haul pull will disappoint. Nebius's own
guidance leans on parallelism everywhere:

- `github.com/nebius/ml-cookbook/tree/main/common/hf-downloader` — the closest thing to an
  official recipe: `huggingface-cli download` into the shared filesystem, explicitly to
  *"avoid loosing time on repeated downloads"*.
- Nebius's vLLM examples set **`HF_HUB_ENABLE_HF_TRANSFER=1`** with `HF_HOME` on a persistent
  volume and a `REDOWNLOAD: "false"` guard.
- `docs.nebius.com/slurm-soperator/storage/download-data` — *"For smaller files (up to 10 TiB),
  use AWS CLI. For larger files (up to 100 TiB), use rclone"*, with a tuned `sbatch`:
  `--transfers=32 --buffer-size=128Mi --multi-thread-streams=24 --multi-thread-cutoff=4Gi`.

**There is no documented HuggingFace mirror, cache or peering arrangement.** HF is just an
upstream. So the download rate is the single biggest unknown in the whole wall-clock model, and
[#70](https://github.com/kreuzhofer/dgx-manager/issues/70) should measure it before anything else.

Note the front-end NIC differs by platform — H100 100 Gbps (ConnectX-6), H200 200 Gbps
(BlueField-3), B200/B300 400 Gbps. None of those is the binding constraint on an HF pull; the
long-haul TCP path is.

### Where the checkpoint should live

**Not on local NVMe — there isn't any.** As established in §1, only `gpu-b300-sxm` in `uk-south1`
has host NVMe. On H100/H200/B200 the choices are the shared filesystem or a network SSD.

The shared filesystem is the right answer, and **it must be sized for bandwidth, not capacity**
(`docs.nebius.com/compute/storage/types`):

| Property | Value |
|---|---|
| Attach mode | `READ_WRITE` is *the only supported value* — genuinely RW-many, mountable by many VMs at once (same project only), over virtiofs |
| Per-client bandwidth | up to **12 GiB/s read**, 8 GiB/s write |
| Aggregate | up to 940 GiB/s read, 475 GiB/s write |
| **Scaling rule** | **per 4 TiB provisioned: +3.70 GiB/s read, +1.89 GiB/s write** |
| Capacity | 1 GiB – 5 PiB; max file size 2 TiB at the default 4 KiB block |
| Price | $0.08/GiB-month; WEKA variant $0.10 |
| Default quota | 32 filesystems, **4 TiB total in `eu-north1`** (2 TiB `eu-west1`, 5 TiB others) |

That scaling rule explains the prior-art measurement and turns it into a lever. The demo's
2 TB filesystem yields ~1.85 GiB/s read; a 470 GB model would take ~4.2 min at the storage layer,
and the demo measured **~14 min** end-to-end — so most of that was deserialisation and
host-to-device copy, not storage. Scaled to 642.7 GB, **~19 min per job start**, of which perhaps
6 min is storage-bound and shrinks if the filesystem is provisioned larger.

Working-set sizing: BF16 base (642.7 GB) + merged output (642.7 GB) + re-quantised NVFP4
(194.7 GB) ≈ **1.5 TB**, against a 4 TiB `eu-north1` default. It fits, but a 4 TiB filesystem only
buys 3.7 GiB/s — provision generously if load time starts to matter.

Object Storage is the cheaper cold tier ($0.0147/GiB-month Standard) with two caveats worth
knowing: Standard and Intelligent classes are throttled to **20 GBps per tenant per region**, and
the unthrottled **Enhanced Throughput** class — which Nebius markets explicitly for *"Write or
read checkpoints"* and *"Stream… model weights to GPU"*, with free operations and free egress —
carries a **1 TiB default quota**, which one BF16 checkpoint nearly fills. Nebius's own training
guidance is *"Checkpoints: SSD shared filesystems, then Object Storage buckets"*.

### Three consequences

1. **Do the merge and re-quantise on Nebius**, where the 598.5 GiB base already sits, and bring
   home only the ~181 GiB NVFP4 result. Shipping a merged BF16 model home to quantise locally is
   3.3× the bytes on the slowest link in the system.
2. **Provision the shared filesystem for read bandwidth**, not just for the 1.5 TB working set.
   Load time is a per-job cost paid on every iteration.
3. **Keep the VM.** Because the download and the load dominate a short run, tearing the
   environment down between iterations is the expensive act — not the GPU-hour. That is the
   opposite of #49's "rent per shift" advice, and it is the direct consequence of the model being
   12× larger.

---

## 7 — What only an account can settle (hand-off to #70)

Every item below was reasoned from public docs or from Nebius's own Terraform and **cannot be
confirmed without a tenant**. [#70](https://github.com/kreuzhofer/dgx-manager/issues/70) should
answer them in this order:

1. **Is `gpu-h200-sxm` / `8gpu-128vcpu-1600gb` visible and creatable in `eu-north1`, self-service,
   on the employee tenant's default quota?** This is the single load-bearing claim of this
   document. If it is not, the recommendation changes.
2. **What is the actual GPU quota on the employee tenant, per region and per platform?** "No
   compute limit" is a statement about budget, not about quota rows. Record the numbers.
3. **Are B200 (`us-central1`) and B300 (`uk-south1`) visible on this tenant at all?** They are the
   2× throughput escape hatch and the only shapes with real room at `seq=32768` and above.
4. **Confirm on this tenant that creating a `gpu_cluster` (InfiniBand fabric) and an MK8s GPU node
   group needs no reservation.** The docs say it does not (§3); confirm before relying on it.
   Only matters if 8×H200 turns out insufficient and multi-node becomes necessary.
5. **Measured HuggingFace → Nebius VM download throughput**, for the 642.7 GB BF16 checkpoint.
   The single biggest unknown in the wall-clock model. Run `hf download` and time it.
6. **Measured download throughput Nebius → `/mnt/tank`**, for bringing the ~181 GiB artifact home.
7. **Shared-filesystem quota and achievable read bandwidth** on the shapes actually granted. The
   1.5 TB working set in §6 has to land somewhere, and — since H200 has no local NVMe — the
   filesystem's provisioned size *is* the load-time lever (+3.70 GiB/s read per 4 TiB).
8. **Whether `eu-north2`** (which appears in the platform table but not in #49's price table) is
   a real, selectable region with H200 capacity.
9. **Does the tenant carry pre-existing capacity reservations / commitment terms** that change any
   of the friction in §3? If the employer already holds capacity block groups, Managed Soperator
   stops being gated and the prior-art Terraform becomes directly reusable.
10. **Is the Managed Soperator worker-node-set "Reservation ID" field genuinely unskippable in the
    live console wizard?** The docs say yes four times over and the API has no create verb for
    capacity block groups — but the marketing page flatly contradicts them, so it is worth 60
    seconds in the console to settle.
11. **Evaluate Serverless AI jobs (`nebius ai job create`) for this workload — 30 minutes.** It is
    single-node and ≤8 GPUs, which is exactly the recommended shape, and Nebius ships an official
    Axolotl + LoRA fine-tuning tutorial on it. The undocumented part is how a 642.7 GB checkpoint
    persists between job runs. If it does, this is a shorter path than a VM.
12. **Confirm the "PAYG capacity is not sticky" behaviour in practice.** Docs say a stopped VM
    returns its GPUs to the shared pool and you may not get them back. That is a real campaign
    risk and it decides whether the VM ever gets stopped.

---

## Confidence

| Claim | Status |
|---|---|
| Checkpoint byte counts and shard counts for all three GLM-5.3-Flash variants | **Measured** — HuggingFace API, 2026-08-31 |
| `config.json` dimensions (45 layers, 288 experts, top-8, vocab 154,880, MTP=1, vision depth 24) | **Measured** — read from the checkpoint |
| Nebius platform IDs, presets, GPUs-per-preset, `gpu_cluster_compatible`, platform→region map | **Measured** — Nebius's own Terraform, `soperator-tf 2.0.2-1` |
| Per-GPU HBM (H100 80, H200 141, B200 180, B300 288) | **Measured** — NVIDIA product pages |
| 470 GB checkpoint loads in ~14 min from Nebius shared NFS; local-SSD checkpointing saves 38% | **Measured** — prior-art demo on this exact platform |
| Fit table (headroom per GPU) | **Derived** — arithmetic on the two measured rows, with a stated 2 GiB/GPU reserve |
| Headroom *required* per method (~15–20 GiB/GPU at 16k) | **Derived, medium confidence** — a component budget, not a profile |
| ~16.5 B active params; ~105 GFLOP/token | **Derived** from `config.json`; independently corroborates the map's ~18 B |
| **20% MoE MFU, and therefore every wall-clock figure** | **Extrapolated, low confidence.** No MoE training MFU has been measured on H200, B200, or GB10. Treat the scenario table as ±2×. |
| Managed Soperator is still capacity-block-gated; MK8s + GPU clusters are self-service | **Measured** — four verbatim doc statements plus the absence of a create verb on the capacity-block-group API |
| Default quotas (H100 32, H200 32/8/0, B200 0, GPU clusters 5) | **Measured** — `docs.nebius.com/compute/resources/quotas-limits` |
| HuggingFace → Nebius download throughput | **Unknown** — blocked on [#70](https://github.com/kreuzhofer/dgx-manager/issues/70) |
| Whether any framework can build a trainable `Glm5NextForConditionalGeneration` at all | **Unknown** — [#67](https://github.com/kreuzhofer/dgx-manager/issues/67). If the answer is "no", this whole document is moot. |

---

## Sources

- `zai-org/GLM-5.3-Flash-BF16`, `zai-org/GLM-5.3-Flash`, `LibertAIDAI/GLM-5.3-Flash-NVFP4` —
  HuggingFace model API (`?blobs=true`) and `config.json`, read 2026-08-31
- Nebius Soperator Terraform `soperator-tf 2.0.2-1`,
  `soperator/modules/available_resources/{platform,preset,gres,region}.tf` and
  `soperator/modules/k8s/k8s_ng_workers_v2.tf`, in
  `/mnt/tank/src/github/nebius-slurm-ml-training-and-inference-demo`
- `/mnt/tank/src/github/nebius-slurm-ml-training-and-inference-demo` — `README.md`,
  `DEMO_SUMMARY.md` (the 2 × 8 × H100 cluster shape, the 235B-MoE load timing, the local-SSD
  checkpoint finding, the 32B LoRA OOM analysis)
- `docs.nebius.com` — `/compute/virtual-machines/types`, `/compute/clusters/gpu`,
  `/compute/resources/{pricing,quotas-limits}`, `/compute/storage/{types,use,local-disks}`,
  `/compute/virtual-machines/tcp-window-tuning`, `/overview/regions`,
  `/overview/limits/capacity-block-groups`, `/kubernetes/{node-groups/manage,gpu/nccl-test}`,
  `/slurm-soperator/{deploy/overview,managed-soperator/manage,managed-soperator/resources/quotas,storage/download-data}`,
  `/object-storage/{resources/pricing,storage-classes,performance-cost-best-practices}`,
  `/vpc/resources/pricing`, `/changelog` — all read 2026-08-31, 20:20–20:45 UTC
- `nebius.com/prices`, `nebius.com/compute`, `nebius.com/compute/b300`, `nebius.com/services/soperator`
- `github.com/nebius` — `nebius-solutions-library/soperator`, `terraform-provider-nebius` (v0.6.49,
  2026-08-31), `ml-cookbook/common/hf-downloader`, `nebius-ps-services/examples/inference/vllm`,
  `nebius-physical-ai/deploy/cluster/variables.tf`
- NVIDIA DGX B200 and H200 product pages; HGX B300 specifications; NGC PyTorch 25.11 release notes
- [`docs/research/training-venue-sparks-vs-nebius.md`](https://github.com/kreuzhofer/dgx-manager/blob/research/training-venue/docs/research/training-venue-sparks-vs-nebius.md)
  (branch `research/training-venue`) — the porting-effort audit, reused not redone
- Memory `glm53-flash-is-a-different-model`; `recipes/dgxrun/glm-5.3-flash-nvfp4-2x.yaml`
