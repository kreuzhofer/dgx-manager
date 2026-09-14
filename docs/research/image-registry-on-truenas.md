# A private OCI registry on TrueNAS, for the Spark cluster

**Research date:** 2026-09-14 · **Asked as:** *"How should we best deploy a private image registry on a
TrueNAS server, and what are the real trade-offs?"* · with the mid-task correction *"they run the
LATEST TrueNAS, and its app store offers Sonatype Nexus Repository."*

**Method.** Primary sources only. iXsystems' own documentation hub and apps site; the **`truenas/apps`
catalog repo read directly from the GitHub API** (every app manifest quoted below is the real
`app.yaml` / `ix_values.yaml` / `templates/docker-compose.yaml`, not a description of one);
`truenas/middleware` source for what TrueNAS actually writes into `/etc/docker/daemon.json`; the CNCF
`distribution` source at tag **`v3.1.1`** — the exact version the TrueNAS app ships — plus its issue
tracker and **release ancestry** (whether a fix is in a release was checked with
`repos/distribution/distribution/compare/<merge-sha>...<tag>`, not by comparing dates); **moby/moby
source** for how Docker resolves registries, because the documentation does not answer the
pull-through question and the code does; Harbor's compose template and server-side allow-list at tag
`v2.15.2`; zot's config defaults in `pkg/api/config/config.go`; Sonatype's help site and
`sonatype/nexus-public` issues.

**Labelling.** Claims I derived by reading code myself are marked **[source-verified]**. Arithmetic of
mine is marked `[inference]`. Single unconfirmed bug reports are labelled as such — several of the
sharpest findings below are one person's report, and I say so every time. Two premises I was given
are **confirmed**, and one thing I was told about a bug's root cause turned out to be **wrong**, and
is corrected in §5.1.

---

## BOTTOM LINE

**2026-09-14.** **Yes, there is a pre-packaged registry app — and it is not Nexus.** The official
`truenas/apps` community train ships **`distribution`**, which is CNCF Distribution (the upstream of
`registry:2`) at **`registry:3.1.1`**, maintained by iXsystems themselves (`dev@truenas.com`), with
first-class UI fields for a TrueNAS-managed TLS certificate, htpasswd users, and a dataset host
path. Nexus is *also* in the catalog (`nexus`, `sonatype/nexus3:3.96.1`, added 2026-07-16), and it is
the wrong tool here: it is a 2 CPU / 4 GB JVM by default, its Community Edition is hard-capped at
**40,000 components or 100,000 requests/day**, its Docker repositories need **one TCP port per
repository**, and the exact version the app ships — 3.96.1 — has an **open, unanswered report that
plain `docker push` to a docker-hosted repo returns HTTP 500**. Harbor and Zot are **not in the
catalog at all** (enumerated: 442 apps across all five trains, zero hits).

**The decisive technical finding is that your premise about `registry-mirrors` is correct, and the
escape hatch is narrower than "use containerd".** I read moby's source: on the **classic** image store
`lookupV2Endpoints` consults `registry-mirrors` **only** when the hostname is `docker.io`/
`index.docker.io`; for every other host it returns exactly one endpoint, the host itself. With the
**containerd image store** the daemon additionally reads containerd-format
`/etc/docker/certs.d/<host>/hosts.toml`, which *does* mirror arbitrary registries — and that landed in
**Docker Engine 28.0.0**, verified by commit ancestry. TrueNAS's own daemon is on the classic store
(`'storage-driver': 'overlay2'`, [source-verified] in `daemon.json.py`), so TrueNAS's shiny new 25.10
"Registry Mirrors" setting mirrors **Docker Hub and nothing else** — it will not help your Sparks and
it will not cache `ghcr.io` or `nvcr.io`.

**Recommendation: two `distribution` app instances, plus TLS from an imported certificate.** One
hosted registry for your own vLLM/CUDA overlays, one pull-through cache if you want one — they cannot
be the same instance, because a proxy registry refuses blob uploads [source-verified]. And **three
things will bite you if you do not act on them before the first push**:

1. The app does **not** set `REGISTRY_STORAGE_DELETE_ENABLED`, so `DELETE` is refused and garbage
   collection has nothing to collect. One env var fixes it.
2. `registry garbage-collect --delete-untagged` **deletes any manifest no tag points at** — which is
   exactly what digest-pinned-only images are. Multi-arch indexes are safe (the fix is in 3.x,
   verified by ancestry); untagged images are not.
3. TrueNAS **25.10 removed the built-in Certificate Authority**, so you can no longer mint a
   self-signed server cert in the UI. Generate the CA and cert offline, import the cert, and drop the
   CA on each client at `/etc/docker/certs.d/<host>:<port>/ca.crt` — which needs **no daemon restart
   at all**, unlike `insecure-registries`.

| Question | Verdict |
|---|---|
| **Is there a pre-packaged registry app?** | **Yes — `distribution`, in the official community train**, `registry:3.1.1`, maintained by iX, added 2024-08-02, `min_scale_version: 24.10.2.2`. Also `nexus` (Sonatype 3.96.1), `gitea`, `forgejo`. **No Harbor, no Zot, no Quay** — enumerated across all five trains. (§2) |
| **Is Nexus the right answer because it's in the store?** | **No.** Per-repository ports, a CE cap of 40k components / 100k requests per day, a JVM, a documented GC task that **irreversibly deletes digest-only manifests**, and an open `docker push` → HTTP 500 regression **in the shipped version 3.96.1**. (§4.4) |
| **Does `registry-mirrors` only work for Docker Hub?** | **Confirmed, in source.** `lookupV2Endpoints` gates mirrors on `hostname == DefaultNamespace \|\| hostname == IndexHostname`. Non-Hub mirroring needs the **containerd image store + `/etc/docker/certs.d/<host>/hosts.toml`**, first available in **Docker Engine 28.0.0**. (§5.5) |
| **Can one registry both host our images and cache ghcr/nvcr?** | **No, not with `distribution`.** *"a proxy registry does not support blob uploads"* [source-verified], and *"It's currently possible to mirror only one upstream registry at a time."* Run separate instances. (§5.5) |
| **Multi-arch: does anything mishandle manifest lists?** | **`distribution` 3.x is safe** — the `--delete-untagged` index bug (#3178) was fixed by #4285 and is in v3.0.0/v3.1.1 but **not v2.8.3**, verified by ancestry; I also read the `unmarkReferencedManifest` pass that implements it. **Harbor's proxy cache is a minefield** (#20920, #18335, #23185 all open). **Nexus's manifest-list support is documented in a forum post, not the docs.** (§5.4) |
| **Garbage collection** | **`distribution`: stop-the-world.** *"You should ensure that the registry is in read-only mode or not running at all."* **Harbor: online**, no read-only needed, but seven open "GC doesn't reclaim" bugs. **Zot: inline, on by default, 1 h interval, no downtime.** **Nexus: soft-delete then compact.** (§5.1) |
| **Biggest risk to a 39 GB image** | **Zot's 60-second HTTP read/write timeouts** (default since v2.1.17). Go treats them as whole-request deadlines, not idle timeouts. Issue #4140 is **open** with a 10 GB image failing *even at `15m`* on v2.1.20. Do not put a 39 GB push through zot without testing it first. (§5.6) |
| **Does ZFS dedup help here?** | **No.** The registry is already content-addressed: one blob per digest, per-repo link files [source-verified in `paths.go`]. Shared layers between variants are *already* stored once. Dedup would buy you the DDT and nothing else. (§6.2) |
| **Registry downtime = deploy downtime?** | **Yes, and the mitigations are real but partial.** A `distribution` pull-through cache *does* degrade gracefully — *"If the remote is unavailable the local association is returned"* [source-verified]. A hosted registry has no such fallback. (§7) |
| **What could I not determine?** | Whether `distribution`'s proxy actually authenticates against **nvcr.io**; whether the 39 GB push survives the TrueNAS app end-to-end; and whether your Sparks run Docker ≥ 28 with the containerd image store. All three are one command away on your hardware. (§9) |

---

## §1 — IDENTIFY: which TrueNAS are you on, and which half of this note applies

Everything below forks on two questions: **CORE or SCALE**, and **before or after 24.10**. Run one of
these.

| Where | Command / place | What you get |
|---|---|---|
| Shell, either product | `cat /etc/version` | e.g. `25.10.7` (SCALE/CE) or `13.0-U6.8` (CORE) |
| Shell, SCALE 24.10+ | `midclt call system.version` | full version string from middleware |
| Shell, SCALE 24.10+ | `midclt call system.version_short` | e.g. `25.10.7` |
| UI | **System → General → Version**, or the dashboard System Information card | product name + version |
| Shell, CORE only | `freenas-version` | CORE reports here; SCALE does not have it |

The current release map, from iXsystems' own
[Software Status page](https://www.truenas.com/docs/softwarestatus/) as of 2026-09-14:

| Line | Code name | Latest | Status |
|---|---|---|---|
| TrueNAS 26 | *(no fish name — annual cadence, plain numbers)* | 26.0.0-BETA.3, 2026-08-20 | Early Release / beta |
| **TrueNAS 25.10** | **Goldeye** | **25.10.7, 2026-09-02** | **General — the current recommendation** |
| TrueNAS 25.04 | Fangtooth | 25.04.2.6, 2025-10-30 | Maintenance |
| TrueNAS 24.10 | Electric Eel | 24.10.2.4, 2025-08-07 | EOL |
| TrueNAS CORE 13.3 | — | 13.3-U1.2, 2025-04-29 | *"no longer under active development"* |
| TrueNAS CORE 13.0 | — | 13.0-U6.8, 2025-07-14 | security updates only |

**"The latest TrueNAS" in September 2026 means 25.10 Goldeye** (released 2025-10-28), unless you are
deliberately on the 26 beta. I have written §2–§8 primarily for **25.10**, and flagged every place
where 26-BETA, 25.04, 24.10 or CORE differ. I have *not* assumed which one you are on for any claim
that matters — where the version changes the answer, both answers are given.

### 1.1 Route by version

| You are on | Route |
|---|---|
| **SCALE / CE 24.10 or later** (Electric Eel, Fangtooth, Goldeye, 26) | Apps are **Docker Compose**. Install the catalog **`distribution`** app, §2.1. This is almost certainly you. |
| **SCALE 24.04 or earlier** (Dragonfish and back) | Apps are **k3s/Kubernetes**. The old `truenas/charts` repo has a `distribution` chart but it is **`registry:2.8.3`** and last touched 2025-03-06. **Upgrade first** — 2.8.3 is missing the multi-arch GC fix (§5.4) and the whole k8s app stack is gone upstream. |
| **CORE 13.x (FreeBSD)** | **There is no container story.** No Docker, no catalog app, no Compose. Your options are a FreeBSD jail running a registry built from ports, or — far better — move the registry to a Linux host. CORE is *"no longer under active development"*. Do not build new infrastructure on it. |
| **TrueNAS 26 BETA** | Same Docker Compose apps, Docker Engine **29.0.4**. Note the beta warning and one breaking change that matters to automation: *"The TrueNAS REST API was deprecated in TrueNAS 25.04 and is removed in TrueNAS 26."* |

Sources: [25.10 Version Notes](https://www.truenas.com/docs/scale/25.10/gettingstarted/versionnotes/),
[26 Version Notes](https://www.truenas.com/docs/scale/26/gettingstarted/versionnotes/),
[24.10 Version Notes (archived)](https://www.truenas.com/docs/scale/24.10/gettingstarted/scalereleasenotes/),
[Software Status](https://www.truenas.com/docs/softwarestatus/).

### 1.2 The 24.10 dividing line, stated precisely

> "TrueNAS 24.10 (Electric Eel) moves the TrueNAS Apps feature backend from Kubernetes to Docker to
> streamline App deployment and management."
> — [24.10 Version Notes](https://www.truenas.com/docs/scale/24.10/gettingstarted/scalereleasenotes/)

Three consequences that outlive the release:

- **`ix-applications` → `.ix-apps`.** *"During the migration process, 24.10 creates a hidden Docker
  dataset on the apps pool that is mounted at `/mnt/.ix-apps`… App storage ix-volumes present in
  `ix-applications` are cloned under the new Docker dataset and promoted."* The old dataset is
  retained, which is what makes rollback possible.
- **Third-party catalogs did not come along.** *"Applications from third-party catalogs, such as
  TrueCharts, do not automatically migrate to 24.10."* TrueCharts subsequently left the platform
  entirely — see §2.4.
- **A manual Docker install from the 24.04 era breaks apps.** iX documents the symptom:
  *"Users who manually installed Docker on TrueNAS 24.04 or earlier can experience TrueNAS Apps
  failure in 24.10 or later due to conflicts between the manually installed and native Docker
  configurations"*, with the error `app_lifecycle.compose_action … 'group_add[0]' expected type
  'string', got unconvertible type 'int', value: '568'`.

### 1.3 One more thing to check on the *clients*, not the NAS

Two facts about each DGX Spark and the eval host decide §5.5 entirely:

```bash
docker version --format '{{.Server.Version}}'          # need >= 28.0.0 for hosts.toml
docker info --format '{{.DriverStatus}}{{.Driver}}'    # "overlayfs"+containerd vs "overlay2"
docker info --format '{{json .}}' | grep -i snapshotter
```

Docker Engine **29.0.0** (2025-11-10) made the containerd image store the default *for fresh
installs only* — *"The containerd image store is the default storage backend for Docker Engine 29.0
and later on fresh installations"*, and upgrades keep the legacy graph driver until you opt in
([containerd image store](https://docs.docker.com/engine/storage/containerd/)). DGX OS images are
routinely older than that. **I could not determine what your Sparks run**; the answer changes whether
transparent mirroring of `ghcr.io`/`nvcr.io` is available to you at all.

---

## §2 — IS THERE A PRE-PACKAGED APP? Yes. Two, and one of them is right.

### 2.1 `distribution` — CNCF Distribution, packaged by iXsystems

Path: [`truenas/apps` → `ix-dev/community/distribution`](https://github.com/truenas/apps/tree/master/ix-dev/community/distribution).
Catalog page: [apps.truenas.com/catalog/distribution_community](https://apps.truenas.com/catalog/distribution_community/).

Verbatim from its [`app.yaml`](https://github.com/truenas/apps/blob/master/ix-dev/community/distribution/app.yaml):

```yaml
annotations:
  min_scale_version: 24.10.2.2
app_version: 3.1.1
date_added: '2024-08-02'
description: Distribution is a toolkit to pack, ship, store, and deliver container content
maintainers:
- email: dev@truenas.com
  name: truenas
run_as_context:
- description: Container [distribution] can run as any non-root user and group.
  gid: 568
  uid: 568
train: community
version: 1.3.8
```

and from [`ix_values.yaml`](https://github.com/truenas/apps/blob/master/ix-dev/community/distribution/ix_values.yaml):

```yaml
images:
  image:
    repository: registry
    tag: 3.1.1
consts:
  ssl_cert_path: /certs/tls.crt
  ssl_key_path:  /certs/tls.key
  htpasswd_path: /auth/htpasswd
  data_path:     /var/lib/registry
```

**How maintained is it?** It is an iXsystems-maintained app (not community-contributed), and
`3.1.1` is the **current upstream release** — CNCF Distribution v3.1.1 shipped 2026-05-01 and there
is nothing newer ([releases](https://github.com/distribution/distribution/releases)). That is a good
sign; contrast the abandoned k3s-era chart at `registry:2.8.3` (§2.3).

**What it actually configures** — I read the
[Jinja compose template](https://github.com/truenas/apps/blob/master/ix-dev/community/distribution/templates/docker-compose.yaml)
line by line. It sets exactly these environment variables and nothing else:

| Set by the app | When |
|---|---|
| `REGISTRY_HTTP_ADDR=0.0.0.0:<port>` | always (default port **30095**) |
| `REGISTRY_HTTP_SECRET` | always (required field in the UI) |
| `REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/var/lib/registry` | when "Use Filesystem Backend" is on (default) |
| `REGISTRY_HTTP_TLS_CERTIFICATE` / `_KEY` | when a **Certificate ID** is selected — the cert and key come from TrueNAS's own certificate store via `values.ix_certificates` |
| `REGISTRY_AUTH_HTPASSWD_REALM=basic-realm` / `_PATH` | when you add **Basic Auth Users**; the hash is **bcrypt, cost 12** ([`library/2.3.11/functions.py`](https://github.com/truenas/apps/blob/master/library/2.3.11/functions.py)), which is what distribution requires — *"The only supported password format is `bcrypt`"* |
| anything you put in **Additional Environment Variables** | `c1.environment.add_user_envs(values.distribution.additional_envs)` |

**The limits, stated plainly:**

- **`REGISTRY_STORAGE_DELETE_ENABLED` is never set.** Upstream default is `false` — *"Use the `delete`
  structure to enable the deletion of image blobs and manifests by digest. It defaults to false"*
  ([configuration](https://distribution.github.io/distribution/about/configuration/)). Out of the box
  this app **cannot delete anything**, and `registry garbage-collect` will therefore find nothing to
  reclaim. This is the single most important thing to fix, and it is one row in the Additional
  Environment Variables list.
- **No proxy/pull-through fields.** You set `REGISTRY_PROXY_REMOTEURL` etc. by hand in Additional
  Environment Variables.
- **No scheduled garbage collection.** There is no cron inside the app. You drive GC from TrueNAS's
  own **System → Advanced → Cron Jobs**, which is a real, documented facility
  ([Managing Cron Jobs](https://www.truenas.com/docs/scale/25.10/scaletutorials/systemsettings/advanced/managecronjobsscale/)) —
  but the command has to be a `docker exec`, and §3.5 explains why iX will not support that.
- **Resource defaults are 2 CPUs / 4096 MB** (`questions.yaml`). Fine for `distribution`, which
  streams uploads to disk and holds essentially nothing in memory.
- **Storage is a single mount.** One dataset at `/var/lib/registry`, either an ixVolume or a Host
  Path. §3 argues hard for Host Path.

### 2.2 `nexus` — Sonatype Nexus Repository 3, also in the catalog

Path: [`ix-dev/community/nexus`](https://github.com/truenas/apps/tree/master/ix-dev/community/nexus).
`app_version: 3.96.1`, `date_added: '2026-07-16'`, catalog `version: 1.0.6`, image
`sonatype/nexus3:3.96.1`, `train: community`, maintainer `dev@truenas.com`.

From its [compose template](https://github.com/truenas/apps/blob/master/ix-dev/community/nexus/templates/docker-compose.yaml)
and `questions.yaml`:

- Web UI port defaults to **30451**, mapped to container **8081**.
- **`additional_ports`** exists, with host port + container port + protocol — so you *can* publish the
  extra Docker connector ports Nexus needs (§4.4). Good.
- **`additional_java_opts`** feeds `INSTALL4J_ADD_VM_PARAMS`, so `-Xmx` is settable.
- Storage is a single `/nexus-data` mount.
- **There is no `certificate_id` field.** Unlike `distribution`, the Nexus app has **no TLS option at
  all** — grep of `questions.yaml` for `certificate` returns nothing. Plain HTTP, or you put a
  reverse proxy in front.
- Resource defaults **2 CPUs / 4096 MB**.

Why this is the wrong tool for this job is §4.4. The short version: per-repository ports, a JVM, a
Community Edition usage cap, a GC model that deletes digest-only manifests, and an open push
regression in this exact version.

### 2.3 What the k3s-era catalog had (SCALE ≤ 24.04)

[`truenas/charts` → `community/distribution`](https://github.com/truenas/charts/tree/master/community/distribution)
exists, but only at chart **1.2.4 / `registry:2.8.3`**, last touched 2025-03-06. **2.8.3 is the final
2.x release** (2023-10-02) and it is on the wrong side of the multi-arch GC fix — see §5.4. If you are
somehow still here, upgrade before you build anything on it.

### 2.4 What is *not* there — and the TrueCharts situation

**Harbor, Zot and Quay do not exist in any TrueNAS catalog.** Verified by enumerating every directory
in all five trains of `truenas/apps` — `stable` (17), `enterprise` (4), `test` (2), `dev` (2),
`community` (417), **442 apps** — and by GitHub code search over the repo: `harbor` → 0, `zot` → 0,
`quay` → 0.

**TrueCharts, the historical third-party answer, is gone from TrueNAS.** Its `dockerregistry` chart
still exists but is Kubernetes-only, and the organisation has renamed itself away from the platform
(`truecharts/public` now resolves to `trueforge-org/truecharts`). Their own announcements:

- 2024-05-30 — [Deprecation of TrueNAS SCALE Apps](https://truecharts.org/news/scale-deprecation/)
- 2024-08-22 — [SCALE FAQ](https://truecharts.org/news/scale-faq/): *"With the upcoming removal of
  Kubernetes from TrueNAS Scale, our Helm Charts can no longer be installed using the native app
  system"*
- 2024-10-27 — [Leaving SCALE](https://truecharts.org/news/leaving-scale/): *"TrueCharts has also
  ended support for TrueNAS SCALE Apps."*

**Conclusion for §2: the pre-packaged app you want exists and is the right software.** Harbor or Zot
would have to go in through **Install via YAML**, which is supported but unblessed (§3.5) — and for
Harbor specifically that is a poor fit, because its installer is not a compose file but a `prepare`
container that generates nginx/config/secrets onto the host filesystem first (§4.3).

---

## §3 — THE MANUAL ROUTE: storage, ports, and what upgrades do

Even if you use the catalog app, **§3.1–3.3 are the part you must get right**, because the app's
defaults do not.

### 3.1 Where the data should live: a dedicated dataset, not an ixVolume

TrueNAS gives you two choices at install time, and its own docs are blunt about the difference
([App Storage](https://apps.truenas.com/getting-started/app-storage/)):

> **ixVolume** — *"enable TrueNAS to automatically create an app storage path inside the hidden
> `ix-apps` dataset"*… designed for rapid testing rather than permanent data, as they *"complicate
> tasks like app data backup."*
>
> **Host Path** — *"allow users to mount existing TrueNAS datasets to paths within the app
> container."* … *"Mounting a host path does not automatically configure appropriate permissions to
> deploy the application."*

and:

> The **ix-apps dataset** is *"internally managed by TrueNAS and hidden to prevent user
> misconfiguration."* … *"you cannot use this as the parent when you create required application
> datasets."*
> Also: *"The ix-apps dataset does not inherit encryption if an encrypted pool is selected as the pool
> for applications."*

**Use a Host Path on a dataset you created and named.** An ixVolume lives inside a hidden,
iX-managed dataset that you are told not to snapshot-share, not to parent, and which does not inherit
pool encryption. 1.4 TB of registry blobs is not "rapid testing" data.

Create the dataset **before** you start the install wizard — iX warns *"You cannot save settings and
exit the configuration wizard to create data storage or directories in the middle of the process."*
And in the app's Storage section, turn on **Enable ACL** and grant uid/gid **568** (`apps`), because
the app runs as 568 and host paths do not get permissions automatically.

### 3.2 Dataset properties — what to set and why

| Property | Set to | Reasoning |
|---|---|---|
| `recordsize` | **1M** | Blobs are large, written once sequentially, read sequentially. OpenZFS: *"recordsize can be set to any power of 2 from 512 bytes to 1 megabyte"*, default 128K; *"Software that writes in fixed record sizes… will benefit from a matching recordsize."* A 39 GB layer at 128K is ~320,000 records of metadata for no benefit. `[inference]` — no primary source benchmarks a registry on ZFS; the reasoning is the standard large-sequential-file case. Note it only affects **new** files, so set it before the first push. |
| `compression` | **lz4** (the default) — leave it on | OpenZFS: *"Since OpenZFS 2.2.0 `compression` defaults to `on`, which selects LZ4"*, and *"incompressible data will be stored without compression such that reads of incompressible data with compression enabled will not be subject to decompression."* LZ4's early-abort makes it ~free on gzip'd layer blobs, and the small stuff (manifests, link files, `_layers` trees — thousands of tiny JSON and hex-digest files) *does* compress. Turning it off saves nothing measurable and costs you the metadata win. |
| `atime` | **off** (or `relatime=on`) | OpenZFS: *"Set either `relatime=on` or `atime=off` to minimize IOs used to update access time stamps."* A registry reads every link file on every GC mark pass; there is no consumer of atime here. |
| `dedup` | **off. Never turn it on for this.** | See §6.2 — the registry already deduplicates by content digest, so there is nothing left for ZFS to find, and TrueNAS's own guidance is *"When data is not sufficiently duplicated, deduplication wastes resources, slows the server down, and has no benefit"*, plus *"High-quality mirrored SSDs configured as a special vdev for the DDT… are strongly recommended"* and *"The only way to convert existing current data to all deduplicated or non-deduplicated… is to create a new copy while new settings are active."* It is a one-way door with a RAM tax for a benefit you already have. |
| `sync` | leave default (`standard`) | Registry writes are ordinary buffered writes; there is no fsync-per-record workload here. No primary source says otherwise. |
| `quota` / `refquota` | **set one** | See §6.1. A runaway proxy cache or a GC that never runs will otherwise eat the pool. |
| `snapshot task` | daily, keep ~2 weeks | §6.3. |

Sources: [OpenZFS Workload Tuning](https://openzfs.github.io/openzfs-docs/Performance%20and%20Tuning/Workload%20Tuning.html),
[TrueNAS ZFS Deduplication](https://www.truenas.com/docs/references/zfsdeduplication/),
[Adding and Managing Datasets](https://www.truenas.com/docs/scale/datasets/managingdatasets/).

### 3.3 Exposing the port

The app publishes one port, default **30095**, with a "Port Bind Mode" of *Publish* and an optional
**Host IPs** list. Bind it to the NAS's LAN address explicitly rather than `0.0.0.0` if the box has
more than one interface. Clients then use `nas.lan:30095/...`.

There is no HTTP→HTTPS redirect and no path prefix: `distribution` serves the registry API at `/v2/`
on whatever port you give it. If you want it on 443 you need a reverse proxy in front
(`nginx-proxy-manager`, `traefik` and `zoraxy` are all in the community train), and then you must set
`REGISTRY_HTTP_HOST` so generated blob-upload `Location` headers are right.

### 3.4 What survives upgrades — and what has actually broken

**Data in a Host Path dataset survives everything**, because it is just a ZFS dataset the container
bind-mounts. That is the main reason to use one.

Documented breakage, in iX's own words:

| Event | What is documented |
|---|---|
| 24.04 → 24.10 (k8s → Docker) | Catalog apps migrate automatically; **third-party catalog apps do not**. `ix-applications` is retained so rollback works. Failed migrations can be re-run: `midclt call -job k8s_to_docker.migrate <poolname>`. |
| A manual Docker install from before 24.10 | Breaks apps with `app_lifecycle.compose_action … 'group_add[0]' expected type 'string', got unconvertible type 'int', value: '568'`. |
| 25.10.0.1 | *"Apps configured to use SMB or NFS shares as storage can experience an occasional race condition during boot that causes them to show a 'crashed' status."* Workaround is restarting the affected apps. **This is an argument for keeping the registry dataset on a local pool, not a share.** |
| 25.10 | Docker service startup on slow disks: *"The service timeout is extended to 960 seconds (16 minutes) to accommodate slower disk scenarios."* — i.e. app startup after a reboot can legitimately take a quarter of an hour. |
| 25.10 | **Certificate Authorities removed** (§5.2). Any cert you were minting in the UI, you now mint elsewhere. |
| 26 | REST API removed (deprecated in 25.04). Irrelevant to the registry, relevant to anything you automate against the NAS. |
| Any version | *"Updating TrueNAS using `apt` or any method other than the TrueNAS web interface can make the system inoperable."* |

**App version upgrades** are a separate axis from OS upgrades: the catalog app's `app_version` tracks
upstream `registry`, and iX ships `app_migrations.yaml` for breaking config changes — `distribution`
has exactly one, `ip_port_migration`, from catalog versions ≤ 1.1.13 to ≥ 1.2.0.

### 3.5 Running it outside the app framework — and iX's position on that

Two documented ways to run a non-catalog container
([Installing Custom Apps](https://apps.truenas.com/managing-apps/installing-custom-apps/)):

1. **Custom App wizard** (the `ix-app` entry in the `stable` train) — a form for one container.
2. **Install via YAML** — *"go to **Apps > Discover**. Click ⋮ then select **Install via YAML**."* …
   *"Enter the Compose YAML file in **Custom Config**."* … *"Begin the Compose file with a top-level
   element, such as `name:`, `services:`, or `include:`."* … *"Generally, any container that follows
   the Open Container Initiative specifications can be deployed."*

with the caveats spelled out:

> *"Installing custom applications via YAML requires advanced knowledge of Docker Compose and YAML
> Syntax. Users should be prepared to troubleshoot and debug their own installations."*
>
> *"TrueNAS applies basic YAML syntax validation to custom applications, but does not apply additional
> validation of configuration parameters before executing the file as written."*

**On using the shell directly:** I could not find a docs page that names `docker` or `docker compose`
on the CLI and declares it supported or unsupported — that widely-quoted line comes from forum posts,
not documentation. What iX *does* say covers it by policy:

> *"The supported mechanisms for making configuration changes are the TrueNAS WebUI and API
> exclusively. All other are not supported and result in undefined behavior that can result in system
> failure!"* — [Using the Shell](https://www.truenas.com/docs/scale/systemsettings/shell/usescaleshell/)
>
> *"Developer mode is for developers only. Users that enable this functionality will not receive
> support on any issues submitted to iXsystems."* … *"These changes do not persist across updates and
> `install-dev-tools` must be re-run after every system update."*
> — [Developer Mode (Unsupported)](https://www.truenas.com/docs/scale/systemsettings/advanced/developermode/)

The Docker daemon is genuinely there (it is the apps backend), so `docker exec` works. But the root
filesystem is replaced wholesale on update, so **anything you put outside a dataset is gone after the
next upgrade**. Practical reading: use the catalog app for the registry, and confine shell use to
`docker exec` of GC, driven from a Cron Job, accepting that this specific action is outside the
supported envelope.

---

## §4 — WHICH REGISTRY SOFTWARE

Weighted as asked: low operational burden, correct multi-arch, working GC.

### 4.1 CNCF `distribution` (`registry:3`) — the recommendation

| | |
|---|---|
| **Footprint** | One static Go binary in an Alpine image. Streams uploads straight to the storage driver — `linkedBlobStore` writes progressively, so a 39 GB push is not buffered in RAM. 2 CPU / 4 GB is generous. |
| **Multi-arch** | Correct. Manifest lists and OCI image indexes are ordinary manifests to it, and the GC mark pass walks `manifest.References()` recursively [source-verified in `garbagecollect.go`]. The historical `--delete-untagged` index bug is fixed in 3.x (§5.4). |
| **GC** | Works, but **stop-the-world** and **off by default** (§5.1). This is its weakest point. |
| **Auth** | htpasswd only, and **all-or-nothing**: `Authorized()` ignores its `accessRecords` argument entirely and returns a blanket grant to any authenticated user [source-verified in `registry/auth/htpasswd/access.go`]. No read-only accounts, no per-repo scoping. Token auth exists but needs a separate token server. |
| **Pull-through** | Yes, but **one upstream per instance** and **read-only** (§5.5). |
| **Referrers API** | **Not implemented.** `registry/api/v2/routes.go` at v3.1.1 registers exactly `base, manifest, tags, blob, blob-upload, blob-upload-chunk, catalog` — there is no `/v2/<name>/referrers/<digest>` route [source-verified]. [Proposal #3716](https://github.com/distribution/distribution/issues/3716) has been open since 2022. Matters only if you want cosign/SBOM discovery by referrers; buildx attestations still round-trip fine because they ride inside the image index. |
| **On TrueNAS** | **Packaged, by iX, at the current upstream version.** No YAML to write. |

### 4.2 Zot — the best software here, and the one thing that might disqualify it

Zot is genuinely the most attractive design: Apache-2.0, CNCF Sandbox
([acceptance record](https://lists.cncf.io/g/cncf-toc/message/7743)), a single static binary, and
*"data on the disk conforms to the OCI Image Layout Specification"*, which means the on-disk store is
readable by any OCI tool rather than being registry-private
([storage](https://zotregistry.dev/v2.1.21/articles/storage/)).

**What it does better than `distribution`:**

- **GC is inline and on by default.** *"Garbage collection in zot is an **inline** feature meaning
  that it is **not** necessary to take the registry offline."* Defaults in
  [`pkg/api/config/config.go`](https://github.com/project-zot/zot/blob/main/pkg/api/config/config.go):
  `GC: true`, `GCDelay: 1h`, `GCInterval: 1h`. No cron job, no read-only window, no `docker exec`.
- **Dedupe by default, via hardlinks.** *"If the server filesystem supports hard links, you can
  optimize storage space by enabling inline deduplication… Deduplication is enabled by default."*
  ZFS supports POSIX hardlinks, so this engages — but note the failure mode is silent: if
  `ValidateHardLink` fails, zot logs one `WARN` (*"input storage root directory filesystem does not
  supports hardlinking, disabling dedupe functionality"*) and carries on. Grep the startup log.
- **Pull-through caching of arbitrary registries is a first-class, documented feature.** Its own
  shipped [`examples/config-popular-registries.json`](https://github.com/project-zot/zot/blob/main/examples/config-popular-registries.json)
  configures **ghcr.io**, quay.io, gcr.io, registry.k8s.io and Docker Hub with `"onDemand": true`.
  And unlike `distribution`, one zot instance can host *and* mirror *and* mirror several upstreams.
- **Real arm64 artifacts.** `ghcr.io/project-zot/zot:latest` is an OCI index covering
  `linux/amd64, linux/arm64, freebsd/*`. Irrelevant on TrueNAS (x86-64 only) but relevant if you ever
  move the registry to a Spark.

**The thing that might kill it for you:** since **v2.1.17** zot defaults `http.readTimeout` and
`http.writeTimeout` to **60 seconds** (`pkg/cli/server/root.go`), and Go's `http.Server` treats those
as **whole-request deadlines, not idle timeouts** — a transfer is killed mid-stream while bytes are
still flowing. The docs acknowledge it: *"Set larger timeout values when handling large image pushes
or pulls over slower networks. Set timeout values to `0` to disable the timeout."*
([security posture](https://zotregistry.dev/v2.1.21/articles/security-posture/)). Reported failures:
[#4079](https://github.com/project-zot/zot/issues/4079) (1.5 GB layer aborted at exactly 60 s;
maintainer: *"Indeed… for large layers use case, you will need to bump these up"*) and — the one that
matters — **[#4140](https://github.com/project-zot/zot/issues/4140), still open**, a 10 GB LLM-model
image where the reporter states on 2026-08-13 with v2.1.20 and both timeouts set to `15m`: *"The
added timeouts to the config did not help, once again a timeout with zot .20."* The streaming fix,
[PR #4149](https://github.com/project-zot/zot/pull/4149), is unmerged.

**Your largest image is 39 GB.** No issue in that tracker mentions a blob anywhere near that size.
I cannot tell you whether zot handles it; I can tell you the closest documented data point is a 10 GB
image failing with generous timeouts, on a recent release, unresolved. That is a *test before you
commit*, not a *deploy and hope*.

Two more zot caveats worth carrying: on-demand sync converts Docker schema-2 to OCI by default, which
**changes digests** — you want `preserveDigest: true` plus `http.compat: ["docker2s2"]`, and *"zot
refuses to start without `http.compat` when `preserveDigest` is enabled"*. And the first pull of an
uncached image **blocks with zero bytes sent** for the whole upstream copy
([#4323](https://github.com/project-zot/zot/issues/4323), open), which for a 39 GB image is a client
timeout waiting to happen.

### 4.3 Harbor — powerful, and not justified here

**It is not disqualified by architecture.** The subagent research flagged that Harbor has no released
arm64 server images (maintainer `wy65701436`, 2026-08-18: *"v2.15.2 only supports single-arch
(amd64)"*; multi-arch merged in [PR #22311](https://github.com/goharbor/harbor/pull/22311), targeted
at v2.16.0, no release date). **That does not matter for you** — TrueNAS is x86-64 only
([Hardware Guide](https://www.truenas.com/docs/scale/gettingstarted/scalehardwareguide/)), and a
registry serves arm64 *content* regardless of its own architecture.

What disqualifies it is weight and fit:

- **Nine containers minimum** (`harbor-log`, `registry`, `registryctl`, `harbor-db`, `harbor-core`,
  `harbor-portal`, `harbor-jobservice`, `redis`, `nginx`), ten with Trivy, eleven with metrics —
  from the [v2.15.2 compose template](https://github.com/goharbor/harbor/blob/v2.15.2/make/photon/prepare/templates/docker_compose/docker-compose.yml.jinja).
  You would be operating PostgreSQL 15, Valkey/Redis 7.2, Trivy, and a **patched distribution 2.8.3**
  (`REGISTRYVERSION=v2.8.3-patch-redis` in the Makefile) — i.e. Harbor's registry core is the *old*
  2.x line.
- **Official minimums are 2 CPU / 4 GB / 40 GB, recommended 4 / 8 / 160**
  ([prerequisites](https://goharbor.io/docs/latest/install-config/installation-prereqs/)).
- **It does not install from a compose file.** The documented upgrade path is
  `docker compose down` → back up `harbor/` and `/data/database` → `docker run -it --rm -v /:/hostfs
  goharbor/prepare:[tag] migrate -i harbor.yml` → `./install.sh`
  ([upgrade](https://goharbor.io/docs/latest/administration/upgrade/)). A `prepare` container that
  writes generated nginx configs and secrets onto the host is a poor match for TrueNAS's
  paste-a-compose-file model, and every OS upgrade puts you back in that flow.
- **Its proxy cache + multi-arch story is actively broken**, which is precisely the feature you would
  be adopting it for: [#20920](https://github.com/goharbor/harbor/issues/20920) *"Proxy Cache not
  creating additional multi-arch images under OCI Image Index"*,
  [#18335](https://github.com/goharbor/harbor/issues/18335) (proxy cache doesn't tag saved artifacts
  when the client pulls by digest — which is what containerd does, and what **you** do),
  [#23185](https://github.com/goharbor/harbor/issues/23185),
  [#21454](https://github.com/goharbor/harbor/issues/21454) — all open, with the common fix
  [PR #23355](https://github.com/goharbor/harbor/pull/23355) unmerged since 2026-06-12.
- **And a trap that combines the two:** Harbor creates a **7-day retention policy on every new proxy
  cache project** by default (`defaultDaysToRetentionForProxyCacheProject = 7` in
  `src/server/v2.0/handler/project.go`), and that policy's selector matches **tagged** artifacts only
  — while digest-pulled cache entries land **untagged**. A GC run with "delete untagged artifacts"
  then destroys the whole cache.
  [#23570](https://github.com/goharbor/harbor/issues/23570), open since 2026-07-16; maintainer
  `stonezdj`: *"it requires to update the GC code to exclude the proxy cache project when deleting the
  untagged artifacts. it is a requirement worth considering."*

**Where Harbor genuinely wins** and would be worth the weight *if you needed it*: **online GC that
does not require read-only mode** — *"Harbor runs garbage collection without interrupting your ability
to continue use Harbor, for example you are able to push, pull, or delete artifacts while garbage
collection is running"* — real RBAC, projects, robot accounts, replication, and a proxy-cache
allow-list that includes **ghcr.io natively** (`github-ghcr`). For a six-client home lab with no
multi-tenancy requirement, none of that is worth nine containers and a PostgreSQL migration on every
upgrade.

### 4.4 Nexus Repository — the convenient answer, and why it is the wrong one

*(This section incorporates a dedicated investigation against help.sonatype.com,
`sonatype/nexus-public` and community.sonatype.com. One important research constraint: Sonatype's
public Jira at `issues.sonatype.org` **was decommissioned in January 2024** and now redirects to an
[FAQ](https://central.sonatype.org/faq/what-happened-to-issues-sonatype-org/) — so **no NEXUS-xxxxx
ticket has a publicly checkable status or fix version**. Ticket IDs below come from release-note
tables or from Sonatype staff naming them in forum posts. Absence of a public ticket proves nothing.)*

#### The premise about ports is half out of date — and that is a point in Nexus's favour

The restriction is real, and here is the sentence
([Docker Registry](https://help.sonatype.com/en/docker-registry.html)):

> *"Docker clients have strict requirements for images paths and **do not allow a repository path to
> be included as part of the path to a docker registry**."*

But Sonatype fixed it server-side in **3.83.0** with **Path-Based Routing**, which the current docs
label **"preferred"** while labelling port connectors **"legacy"**:

> `docker pull nexus.example/repository-name/namespace/image:latest`
>
> *"Path-Based Routing is designed to meet the security requirements of Nexus Repository Cloud. While
> self-hosted deployments are encouraged to deploy this model as well, it is the only method available
> for Nexus Repository Cloud deployments."*
>
> *"Only one routing method may be used at a given time."*

So the answer to "how many ports for hosted + ghcr proxy + nvcr proxy + group":

| Routing method | Ports needed | Edition |
|---|---:|---|
| **Path-Based Routing** (3.83.0+, preferred) | **0 extra** — everything on the one web port | Community |
| Reverse proxy | 0 extra on Nexus; the proxy owns 443 | Community |
| **Port connectors** (legacy) | **1 per repo you address directly**; 2 minimum if you use a group for pull + hosted for push; 8 if you want HTTP *and* HTTPS on each of four | Community |
| **Subdomain connector** | 0 extra, but needs wildcard DNS + wildcard TLS | **Pro only** — the OSS source itself gates it: `subdomainVisibility = NX.State.getEdition() === 'PRO'` |

The TrueNAS app's `additional_ports` field means even the legacy route is workable. **This is not the
reason to reject Nexus.**

#### The one thing Nexus does that nothing else here does: nvcr.io, officially

Sonatype documents proxying NVIDIA NGC by name
([Docker Registry](https://help.sonatype.com/en/docker-registry.html)):

> *"Create Docker Proxy registry in Nexus Repository using the name `nvidia-nim` for the path based
> routing."* · *"Set the remote storage URL to `https://nvcr.io/`"* · Username `$oauthtoken`,
> Password: API TOKEN
>
> `docker pull example.repo/nvidia-nim/nim/deepseek-ai/deepseek-coder-v2-lite-instruct:1`

That matches NVIDIA's own auth model —
[NGC](https://docs.nvidia.com/ngc/latest/ngc-private-registry-user-guide.html): *"For the username,
enter `$oauthtoken` exactly as shown. It is a special name that indicates that you will authenticate
with an API key."* **No other option here has nvcr.io in its documentation.** Harbor has one 2023 user
report ([#11548](https://github.com/goharbor/harbor/issues/11548): *"I have configured nvcr.io with
the docker registry provider and the pull action is working"*); zot has zero hits for `nvcr` in code
or issues; `distribution` says nothing either way.

And the client-side reference form is exactly what you guessed: **you address the proxy directly and
rewrite the image reference.** `nexus.lan/nvidia-nim/nim/...` instead of `nvcr.io/nim/...`. That
sidesteps the `registry-mirrors` limitation entirely — at the cost of every recipe in this repo
carrying a host that is specific to your LAN.

#### Why it is still the wrong choice here

1. **Community Edition is hard-capped, and a container registry is the worst possible workload for
   that cap.** [Usage Center](https://help.sonatype.com/en/usage-center.html): *"the Community Edition
   usage limits of **40,000 total components or 100,000 requests per day**"* and *"In Community
   Edition deployments exceeding the 40,000 total components or 100,000 requests per day limitation,
   **users are not able to add new components** until the deployment returns to being under both of
   these limits."* A single `docker pull` is a token request plus a manifest plus a HEAD and a GET per
   layer — dozens of HTTP requests. A four-node fleet re-pulling a 40-layer image is thousands of
   requests in one deploy.
2. **The TrueNAS app is a single container with no database — which Sonatype says is unsupported.**
   [Database options](https://help.sonatype.com/en/database-options.html): H2 supports *"maximum
   200,000 requests per day and 100,000 components"*, and **container-based deployments are not
   supported on H2**; external PostgreSQL is recommended. The catalog app renders exactly one service
   (`nexus`) plus the permissions init container — there is no Postgres. So the packaged app is, by
   Sonatype's own definition, an unsupported configuration.
3. **The shipped version has an open push regression.**
   [nexus-public #1059](https://github.com/sonatype/nexus-public/issues/1059), filed 2026-09-10:
   plain `docker push` to a `docker-hosted` repo fails after upgrading 3.95.1 → **3.96.0** with
   `unexpected status from HEAD request to …/manifests/sha256:…: 500 Server Error`, **still present in
   3.96.1** (confirmed by the reporter 2026-09-14), no Sonatype response. **The TrueNAS app ships
   `sonatype/nexus3:3.96.1`.** One unconfirmed report, but it is against your exact version and the
   exact operation you need.
4. **Its GC model punishes digest pinning, irreversibly.**
   [Tasks](https://help.sonatype.com/en/tasks.html), on *Docker - Delete unused manifests and images*:
   *"This task deletes manifests and layers if they are pulled or pushed by the Docker digest… because
   those are not referenced by any tag. In hosted repositories, **this data loss is irreversible**."*
   And [Cleanup Policies](https://help.sonatype.com/en/cleanup-policies.html): *"Cleanup only
   evaluates tagged manifests for Docker."* Reclaiming space is a **four-task chain** — cleanup
   policies soft-delete *tags*, the Docker GC task handles manifests and layers, and *Compact blob
   store* actually frees bytes — and it has a bad track record: NEXUS-28247 (GC *"deleted live
   images"*, 3.30.0–3.31.1, Sonatype told users to disable the task), NEXUS-52536 (Docker GC only
   evaluated the first 100 unreferenced blobs, 3.91.0–3.92), NEXUS-54717 (**HEAD requests did not
   update `lastDownloaded`, so actively-used images became cleanup-eligible** — 3.94.0–3.95.2;
   containerd uses HEAD).
5. **Multi-arch is documented in a forum post.** The only primary statement that hosted `docker`
   repos support manifest lists is a Sonatype staffer in 2020 citing NEXUS-18546 and apologising that
   it never made the release notes
   ([community 1753](https://community.sonatype.com/t/nexus-3-oss-multi-arch-docker-images-for-docker-repository/1753)).
   Asked in May 2026 whether **proxy** repos handle multi-arch, Sonatype staff answered *"I would
   expect it to work"*
   ([community 16311](https://community.sonatype.com/t/nexus-rm-proxy-repository-docker-multi-arch-manifest-support/16311)).
   The new native **`oci` format (3.94.0, 2026-07-09)** *does* document image indexes and the OCI 1.1
   Referrers API — but it is two months old, has open bugs
   ([#1040](https://github.com/sonatype/nexus-public/issues/1040) chunked manifest PUT → 411,
   [#1061](https://github.com/sonatype/nexus-public/issues/1061) relative-redirect 502s), and **no
   cleanup task or GC is documented for it at all**.
6. **A JVM, and no TLS in the app.** 2 CPU / 4 GB by default; Sonatype's own sizing is indexed by
   requests/hour and puts a lab in the 8–16 GB band. `-Xms` must equal `-Xmx`; note that since 3.78
   `nexus.vmoptions` is **ignored in the Docker image** — the TrueNAS app is already doing the right
   thing by using `INSTALL4J_ADD_VM_PARAMS`.

**Honest counterweight:** Nexus imposes no upload size limit (uploads stage in the blob store's own
`content/tmp` and are **renamed** into place, so no 2× space), it has a documented nvcr.io recipe, and
if you already wanted one artifact manager for Python wheels *and* images it would earn its keep.
For a registry that holds 61 container images, it is a lot of moving parts and one hard usage cap
between you and a deploy.

### 4.5 Side by side

| | **distribution 3.1.1** | **Zot 2.1.21** | **Harbor 2.15.2** | **Nexus 3.96.1 CE** |
|---|---|---|---|---|
| In the TrueNAS catalog | **yes, by iX** | no | no | yes, by iX |
| Containers to operate | 1 | 1 | **9–11** | 1 (+ Postgres if you follow the docs) |
| GC | offline / read-only | **inline, on by default** | online, no read-only | 4 scheduled tasks, online |
| GC track record | 1 open data-loss report (§5.1) | 1 fixed repo-wide abort | 7 open "doesn't reclaim" | 3 historic data-loss bugs |
| Multi-arch | correct, verified in source | correct, explicit index branch | hosted ok, **proxy broken** | forum-documented only |
| Pull-through, arbitrary upstream | **1 upstream per instance, read-only** | **many, on-demand, can also host** | ghcr native; nvcr community-only | **nvcr documented**; ghcr only on `oci` |
| Auth granularity | **all-or-nothing** | users/groups + policies | full RBAC | full RBAC |
| Hard usage cap | none | none | none | **40k components / 100k req/day** |
| Big-blob risk | none known | **60 s timeouts, #4140 open** | nginx tuned for it | none known |
| Referrers API | **no** | yes | yes | only on `oci` format |

---

## §5 — THE THINGS THAT BITE

### 5.1 Garbage collection

**Registries never reclaim on their own.** Every option here needs a deliberate act.

#### `distribution` — stop-the-world, and off by default

[Garbage collection](https://distribution.github.io/distribution/about/garbage-collection/):

> *"You should ensure that the registry is in read-only mode or not running at all. **If you were to
> upload an image while garbage collection is running, there is the risk that the image's layers are
> mistakenly deleted leading to a corrupted image.**"*

```
registry garbage-collect [--dry-run] [--delete-untagged] [--quiet] /etc/distribution/config.yml
```

(the config path moved to `/etc/distribution/config.yml` in 3.0 — an explicit deprecation in the
[v3.0.0 release notes](https://github.com/distribution/distribution/releases/tag/v3.0.0)).

Three facts that are not in the docs and that I verified in the v3.1.1 source:

- **`REGISTRY_STORAGE_DELETE_ENABLED=true` gates the DELETE *API*, not the GC command.**
  `linkedBlobStore.Delete` starts `if !lbs.deleteEnabled { return distribution.ErrUnsupported }`, but
  `MarkAndSweep` reclaims through `NewVacuum(ctx, storageDriver)` — straight to the storage driver,
  ungated. **Consequence:** without it, GC still works, but you cannot *deliberately* remove an image;
  the only way to orphan a manifest is to overwrite its tag. Set it. [source-verified]
- **Read-only mode is a config change, which on TrueNAS means a container restart.** There is no
  runtime toggle: `app.readOnly` is read once in `NewApp` from `storage.maintenance.readonly.enabled`.
  On the TrueNAS app that means Edit → add `REGISTRY_STORAGE_MAINTENANCE_READONLY_ENABLED=true` →
  Save (restart) → GC → remove it → Save (restart). **Stopping the app entirely is simpler and
  strictly safer.** [source-verified]
- **Abandoned uploads are cleaned up automatically.** `uploadPurgeDefaultConfig()` is
  `enabled=true, age=168h, interval=24h`. A failed 39 GB push leaves its staging data for a week, then
  goes away by itself. [source-verified]

**The open data-loss report you should know about.**
[#4939](https://github.com/distribution/distribution/issues/4939), filed **2026-08-24**, against
**`registry 3.1.1`** — the exact version the app ships: *"Running `garbage-collect --delete-untagged`
… occasionally deletes a blob that is still referenced by multiple live, tagged manifests."* The
reporter shows a blob correctly marked on two prior nightly runs and then deleted on a third with no
tag change. **One report, zero comments, unreproduced, no maintainer response.** Note also that the
reporter describes *nightly runs*, which is precisely the *"do not run this on a live registry"*
scenario the docs warn about. I would not call this confirmed — but I would run GC against a stopped
app, and I would run `--dry-run` first, which the docs recommend anyway.

Two more open GC issues worth knowing:
[#3851](https://github.com/distribution/distribution/issues/3851) (*"garbage-collect does not purge
unused `_layers`"*, open since 2023 — link files survive, which costs inodes not bytes) and
[#4249](https://github.com/distribution/distribution/issues/4249) (switching `proxy.ttl` from 0 to
>0 leaks blobs).

**On TrueNAS, GC has to be driven from System → Advanced → Cron Jobs**
([docs](https://www.truenas.com/docs/scale/25.10/scaletutorials/systemsettings/advanced/managecronjobsscale/)),
because the app itself has no scheduler. The awkward bit is that the safe way (app stopped) and the
convenient way (`docker exec`) are mutually exclusive — a stopped container cannot be `exec`'d into —
so the offline form runs the same image as a throwaway container against the dataset:

```sh
# A. offline GC against the dataset, with the app stopped — safest
midclt call -job app.stop registry-hosted
docker run --rm -v /mnt/tank/registry:/var/lib/registry \
  -e REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/var/lib/registry \
  registry:3.1.1 garbage-collect --delete-untagged /etc/distribution/config.yml
midclt call -job app.start registry-hosted

# B. in-place, app running, accepting the documented risk — use --dry-run first
docker exec "$(docker ps --format '{{.Names}}' | grep -m1 distribution)" \
  registry garbage-collect --dry-run --delete-untagged /etc/distribution/config.yml
```

Shape **A** is the one to schedule. Note both use `docker` from the shell, which is outside iX's
supported envelope (§3.5) — this is the single unavoidable place where the packaged app does not
cover the job. **I have not run either command**; the flags and config path are from the docs and the
image's `entrypoint.sh`, and the `midclt app.*` verbs are the documented middleware calls.

#### The others, briefly

- **Zot** — *"Garbage collection in zot is an **inline** feature meaning that it is **not** necessary
  to take the registry offline."* On by default, `GCDelay: 1h`, `GCInterval: 1h`. Retention policies
  (`storage.retention`) replaced the old `untaggedImageRetentionDelay` in v2.0.0. Caveats: GC
  serialises against the store lock (open [PR #4404](https://github.com/project-zot/zot/pull/4404):
  *"a manifest write or garbage-collection pass on one repository blocks reads and writes of every
  other repository"*), and [#4271](https://github.com/project-zot/zot/issues/4271) saw one dangling
  digest abort GC repo-wide with *~90 GiB of collectable data* accumulating silently (fixed in
  v2.1.21). **This is the best GC story of the four.**
- **Harbor** — online, schedulable (None/Hourly/Daily/Weekly/cron), Dry Run available, with a 2-hour
  protection window for recent uploads. But seven open "GC doesn't reclaim" issues, including
  [#23178](https://github.com/goharbor/harbor/issues/23178) (*silently fails on S3-compatible storage
  and reports success with fake freed space*) and
  [#18014](https://github.com/goharbor/harbor/issues/18014), where a maintainer states the cosign
  behaviour is *"a designed behavior where you have to remove the signature first and then perform the
  GC"*. Also: *"Stopping a run will only stop the garbage collection task from processing more
  artifacts. Harbor will not restore any artifact that has already been garbage collected."*
- **Nexus** — §4.4 item 4. Soft-delete then compact; four tasks; no downtime required.

### 5.2 TLS vs `insecure-registries` — and the 25.10 curveball

**Do TLS. Here is why the usual "it's a trusted LAN" argument loses.**

`insecure-registries` has to be configured on **every** client, and it is a *daemon* setting, not a
per-user one. Adding it later, or removing it later, is a fleet-wide change. Worse, it is a one-way
trust decision: `IsInsecureRegistry` sets `TLSClientConfig.InsecureSkipVerify = true` **and** appends
a plain-`http://` endpoint, so the daemon will happily fall back to cleartext
[source-verified in `service_v2.go` / `hosts.go`]. And `distribution`'s own htpasswd package says:

> *"This authentication method MUST be used under TLS, as simple token-replay attack is possible."*

**The 25.10 curveball: TrueNAS removed its Certificate Authority.** From the
[25.10 Version Notes](https://www.truenas.com/docs/scale/25.10/gettingstarted/versionnotes/):

> *"Removes **Certificate Authorities (CA)** screens and support for CAs, which means **you cannot
> sign CSRs or create self-signed certificates**."*
>
> *"Users can continue to manage certificates by creating Certificate Signing Requests (CSRs) to be
> signed by external certificate authorities"* … *"Users can import a certificate created by an
> external certificate authority."* … plus a new **Create ACME Certificate** screen.

So on 25.10 the `distribution` app's **Certificate ID** dropdown will only contain certificates you
**imported**. Two routes:

- **Private CA, generated off-box** (what I would do on a flat LAN with no public DNS): make a CA and
  a server cert with SANs for both the hostname and the IP, on the Pi manager; import the **server
  cert + key** into TrueNAS (Credentials → Certificates → Import); select it in the app; distribute
  the **CA cert** to clients. On 25.04 and earlier you could have done the CA in the TrueNAS UI; you
  can't now.
- **ACME**, if the NAS has a resolvable public DNS name and you add a DNS authenticator. Renewals are
  then automatic, and clients need no CA distribution at all. Cleaner if it fits.

**Client-side, exactly.** Docker's own reference
([dockerd](https://docs.docker.com/reference/cli/dockerd/)):

> *"A secure registry uses TLS and a copy of its CA certificate is placed on the Docker host at
> `/etc/docker/certs.d/myregistry:5000/ca.crt`. An insecure registry is either not using TLS (i.e.,
> listening on plain text HTTP), or is using TLS with a CA certificate not known by the Docker
> daemon."*

So on each Spark, the eval host and the Pi:

```bash
sudo mkdir -p /etc/docker/certs.d/nas.lan:30095
sudo cp lan-ca.crt /etc/docker/certs.d/nas.lan:30095/ca.crt
# no daemon restart, no reload
```

**The directory name must match the registry reference exactly, port included.** And — this is the
part worth knowing — **no restart or reload is needed**: the TLS config is built inside the endpoint
lookup on every pull [source-verified: `newTLSConfig(ctx, hostname, ...)` inside `lookupV2Endpoints`;
and on the containerd path `loadTLSConfig(certsDir)` inside `RegistryHosts`]. containerd documents
the same fallback: *"If no hosts.toml configuration exists in the host directory, containerd will
fallback to check certificate files based on Docker's certificate file pattern."*

Contrast the insecure route, which *does* touch the daemon:

```json
/etc/docker/daemon.json
{ "insecure-registries": ["nas.lan:30095"] }
```

The good news is that `insecure-registries` and `registry-mirrors` are both **SIGHUP-reloadable**, so
`systemctl reload docker` applies them **without stopping running containers**
([dockerd reload table](https://docs.docker.com/reference/cli/dockerd/); confirmed in
`daemon/reload.go`'s own doc comment: *"These are the settings that Reload changes: … Insecure
registries · Registry mirrors …"*). **This matters a lot on the Sparks**: a full
`systemctl restart docker` would kill every running vLLM deployment, because `live-restore` is
*"disabled by default"* and *"when the daemon restarts, all running containers are shut down"*
([live restore](https://docs.docker.com/engine/daemon/live-restore/)). Use `reload`, never `restart`.

**Sizing the undo cost.** Dropping a CA file is per-host and reversible. `insecure-registries` is a
daemon config change on six hosts that you will have to undo on six hosts. The CA route costs about
ten extra minutes once.

### 5.3 Auth — worth it, but know what you are buying

On a trusted LAN with no ingress, authentication buys you **accident prevention, not security**: it
stops a mistyped push from landing in the wrong place, and it makes the registry safe to expose later
without re-plumbing clients.

**What `distribution` + htpasswd actually gives you, which is less than you'd assume:**

`accessController.Authorized()` takes `accessRecords ...auth.Access` and **ignores them entirely**,
returning `&auth.Grant{User: ...}` for any user whose bcrypt hash matches [source-verified in
`registry/auth/htpasswd/access.go`]. There is **no read-only user, no per-repository scope, no
push/pull split**. Everyone who can authenticate can push, delete (if delete is enabled), and read
everything. The only coarse lever is `storage.maintenance.readonly`, which applies to everybody.

**And it costs CPU on every request.** `authenticateUser` calls
`bcrypt.CompareHashAndPassword` per request with **no caching**, and the TrueNAS app generates hashes
at **cost 12** (`_htpasswd(self, username, password, rounds: int = 12)`). Basic auth is sent on every
HTTP request, so a 40-layer pull is ~40+ bcrypt verifications. `[inference]` cost 12 is 4096 key
derivation rounds, order 10² ms each on a NAS CPU — tens of seconds of CPU across a big pull, not
fatal, but not free either. If it shows up, the fix is a token auth service, not a weaker hash.

**Adding auth later breaks exactly one thing, predictably:** every client needs
`docker login nas.lan:30095` once, and anything non-interactive needs the credential in
`~/.docker/config.json` (or `DOCKER_CONFIG`) for the **user the daemon's client runs as**. For this
repo that means the agent on each Spark and whatever CI builds the overlays. Nothing about the stored
images changes. **Adding TLS later is the disruptive one** — the registry's hostname:port identity
changes trust, and every `insecure-registries` entry has to come back out.

### 5.4 Multi-arch — one real historical hazard, and it is behind you

Your clients are arm64 (4× Spark, sm_121a) and amd64. The question is whether anything mishandles
`application/vnd.docker.distribution.manifest.list.v2+json` / `application/vnd.oci.image.index.v1+json`.

**`distribution` is correct, and I verified the mechanism rather than trusting the changelog.** The
hazard was [#3178](https://github.com/distribution/distribution/issues/3178), *"Registry
garbage-collect --delete-untagged removes multi-arch manifests"* — because a per-platform child
manifest has no tag of its own. It was fixed by
[PR #4285](https://github.com/distribution/distribution/pull/4285), merged 2024-04-21 (`df98374`).
Ancestry check:

```
df98374...v2.8.3  -> diverged   (NOT in 2.8.3, the last 2.x release)
df98374...v3.0.0  -> ahead      (in)
df98374...v3.1.1  -> ahead      (in)
```

The implementation is a second pass, `unmarkReferencedManifest(manifestArr, markSet, …)`, which drops
from the delete list any manifest that the recursive `markManifestReferences` walk reached as a child
of something tagged [source-verified]. **So: `registry:3.x` is safe; `registry:2.8.3` — which is what
the old TrueNAS k3s chart shipped, and what Harbor still embeds — is not.**

**The hazard that remains, and it is aimed squarely at you:** `--delete-untagged` deletes any manifest
that **no tag points at and no tagged index references**. A digest-pinned image with no tag is exactly
that. You told me you pin by digest because a dated tag resolved to two different images on two nodes
— good practice, and it interacts badly with untagged GC on **every** option here (`distribution`
§5.1, Nexus §4.4 item 4, Harbor #23570). **Rule: always push with a tag as well as relying on the
digest.** The tag is the GC anchor; the digest is what the recipe uses. That costs nothing and
removes the entire class.

Other options:

- **Zot** — correct, and explicitly so: `pkg/extensions/sync/destination.go` branches on
  `case ispec.MediaTypeImageIndex, mediatype.Docker2ManifestList:` and walks every child. It is on the
  [OCI distribution-spec conformance list](https://github.com/opencontainers/oci-conformance/tree/main/distribution-spec#project-zotzot).
  One wrinkle: on-demand sync pulls **all** architectures, not just the requested one
  ([#3846](https://github.com/project-zot/zot/issues/3846)) — fixed by
  [PR #4410](https://github.com/project-zot/zot/pull/4410) merged 2026-09-11, i.e. **on `main`, after
  v2.1.21, not in any release**. For a 39 GB image that is 2× the transfer.
- **Harbor** — hosted is fine; **proxy cache + multi-arch is broken** (§4.3).
- **Nexus** — forum-documented on the `docker` format; documented properly only on the new `oci`
  format (§4.4 item 5).

**TrueNAS's own architecture is irrelevant to this.** TrueNAS is x86-64 only
([Hardware Guide](https://www.truenas.com/docs/scale/gettingstarted/scalehardwareguide/)), and a
registry is a content store — it serves arm64 manifests and blobs without caring what it runs on.

### 5.5 Pull-through cache — your premise is correct, and here is the code

**Claim:** *Docker Engine's `registry-mirrors` only applies to Docker Hub; mirroring other registries
needs containerd `hosts.toml`.*

**Verdict: confirmed, with one important refinement — Docker itself now reads `hosts.toml`, but only
on the containerd image store and only since Engine 28.0.**

**Documentation first.** [Mirror the Docker Hub library](https://docs.docker.com/docker-hub/image-library/mirror/):
*"It's currently not possible to mirror another private registry. Only the central Hub can be
mirrored."* And [the distribution mirror recipe](https://distribution.github.io/distribution/recipes/mirror/):
*"Currently Docker daemon supports only mirrors of Docker Hub. It is not possible to run the Docker
daemon against a pull through cache with another upstream registry."* plus *"It's currently possible
to mirror only one upstream registry at a time."*

**Now the source, because the docs do not cover the containerd case.** moby/moby has two pull paths:

| Path | Used by | Mirror behaviour |
|---|---|---|
| **Classic image store** — `daemon/images/image_pull.go` → `daemon/internal/distribution` → `Service.LookupPullEndpoints` → `lookupV2Endpoints` | the default on any Engine < 29.0, and on every upgraded install | `if hostname == DefaultNamespace \|\| hostname == IndexHostname { …append mirrors… }` — **mirrors are consulted for `docker.io` only.** Every other host returns exactly one endpoint: itself. |
| **containerd image store** — `daemon/containerd/image_pull.go` → `resolver.go` → `Daemon.RegistryHosts` (`daemon/hosts.go`) | Engine ≥ 28.0 with `features.containerd-snapshotter`, default on fresh 29.0+ installs | `hostconfig.ConfigureHosts(ctx, HostOptions{HostDir: hostconfig.HostDirFromRoot(registry.CertsDir())})` — reads **containerd-format `hosts.toml` for any registry**. Legacy config is then merged, and again gated: `if host == "docker.io" && len(sc.Mirrors) > 0`. |

`registry.CertsDir()` is `"/etc/docker/certs.d"` on Linux. So the mirror file is
**`/etc/docker/certs.d/<registry>/hosts.toml`** — Docker's certs directory, containerd's file format.

**Which Engine?** The commits that introduced this (`8b4cb6f5` *"Update host resolver to use
containerd host config"*, `b3569ebd`, `2aaae08a`, all 2024-10-25) compare as:

```
8b4cb6f5...v27.5.1 -> diverged   (not in 27.x)
8b4cb6f5...v28.0.0 -> ahead      (in)
```

**Docker Engine 28.0.0 is the floor.** Containerd's own docs confirm arbitrary hosts are in scope,
naming *"docker.io, quay.io, gcr.io, and ghcr.io"*, with a `_default` fallback namespace
([hosts.md](https://github.com/containerd/containerd/blob/main/docs/hosts.md)).

**So you have three shapes, and they are not equivalent:**

| Shape | Client reference | Works for | Requires |
|---|---|---|---|
| **A. Rewrite the reference** | `nas.lan:30096/nvidia/cuda:…` | **anything** | nothing on the client; every recipe in this repo changes |
| **B. `hosts.toml`** | `nvcr.io/nvidia/cuda:…` — unchanged | **anything** | Docker ≥ 28.0 **and** the containerd image store on every client |
| **C. `registry-mirrors`** | `ubuntu:24.04` — unchanged | **Docker Hub only** | nothing |

A `hosts.toml` for a `distribution` proxy instance is as simple as it looks, because the proxy keeps
upstream repository names unchanged:

```toml
# /etc/docker/certs.d/nvcr.io/hosts.toml
server = "https://nvcr.io"

[host."https://nas.lan:30096"]
  capabilities = ["pull", "resolve"]
```

(For a **Harbor** proxy you would additionally need `override_path = true` and the project in the
host URL, because Harbor addresses cached content as `<harbor>/<project>/<repo>` — `[inference]` from
Harbor's documented addressing, not something I verified against a running Harbor.)

**`distribution` as a cache: what you get and what you don't.**

- **One upstream per instance, and it cannot also host.** `NewApp` sets `isCache = config.Proxy.RemoteURL != ""`, and the code comment is explicit: *"a proxy registry does not support blob
  uploads"*. `proxyManifestStore.Put` returns `ErrUnsupported`. [source-verified] So: one app instance
  for your own images, one per upstream you want to cache.
- **It degrades gracefully when the upstream is down.** `proxyTagService.Get`'s own doc comment:
  *"Get attempts to get the most recent digest for the tag by checking the remote tag service first
  and then caching it locally. **If the remote is unavailable the local association is returned**."*
  And `proxyManifestStore.Get` checks local storage first and never contacts the remote on a hit.
  [source-verified] **This is the single best availability property in this whole note** (§7).
- **The cache expires by default.** `ttl` defaults to **168h (7 days)**; `proxy.ttl: 0` means *never
  expire* (`else { ttl = nil }`). [source-verified]
- **But the blob TTL never actually reclaims space, and the published root cause is wrong.**
  [#4957](https://github.com/distribution/distribution/issues/4957) (filed 2026-09-14, against
  v3.1.1) reports that blob expiry always errors and nothing is freed, and attributes it to
  `proxyBlobStore.Delete` being an unconditional `ErrUnsupported` stub. **I read the code and that
  attribution is wrong**: `OnBlobExpire`'s closure calls `registry.Repository(ctx, r)` on the
  **embedded/local** namespace passed into `NewRegistryPullThroughCache`, not on the proxy wrapper —
  so it reaches `linkedBlobStore.Delete`, whose first line is
  `if !lbs.deleteEnabled { return distribution.ErrUnsupported }`. Same symptom, different cause, and
  **a different fix: set `REGISTRY_STORAGE_DELETE_ENABLED=true` on proxy instances too.**
  `[inference from source — I did not test it]`. Either way, put a dataset **quota** on a cache and
  watch it.
- **Credentials leak scope.** Docker's own warning, which applies exactly to an NGC key:
  *"If you specify a username and password, it's very important to understand that private resources
  that this user has access to Docker Hub is made available on your mirror. **You must secure your
  mirror** by implementing authentication if you expect these resources to stay private!"* An
  unauthenticated nvcr.io cache on the LAN re-serves everything that key can reach.
- **Whether it authenticates against nvcr.io at all: I could not determine.** `configureAuth` builds a
  standard bearer-token challenge handler, `nvcr.io` advertises
  `Bearer realm="https://nvcr.io/proxy_auth"` and is a v2 registry, so it *should* work with
  `REGISTRY_PROXY_USERNAME='$oauthtoken'`. But no primary source says so, and the one historical
  non-Hub proxy bug on record — [#3530](https://github.com/distribution/distribution/issues/3530),
  *"Mirroring ghcr.io image returns 400 MANIFEST_INVALID"* — is instructive: the reporter root-caused
  it to `manifest/schema1/config_builder.go` trying to push a gzipped empty tar into a store that
  refuses writes. **That path no longer exists** — `manifest/schema1/` is 404 at v3.1.1 and present at
  v2.8.3, i.e. schema1 was removed in 3.0 — so the ghcr bug is almost certainly dead. That is
  reassuring by analogy, not by test.

**And one non-technical consideration you raised yourself:** your images derive from
`nvcr.io/nvidia/*` and third-party ghcr images *"with redistribution terms"*. A pull-through cache
re-serves upstream content to everyone who can reach it. On a six-host private LAN that is almost
certainly fine, but it is a licensing question, not an engineering one, and I am not the right source
for it.

### 5.6 Large blobs — the 39 GB question

`distribution`'s filesystem path is boring in the good way. An upload streams into
`<root>/v2/repositories/<name>/_uploads/<id>/data` and is committed with
`os.Rename(source, dest)` [source-verified in `filesystem/driver.go`] — **same dataset, so no 2×
space at commit**, just the in-flight copy. Nothing is buffered in memory, so the app's 4 GB limit is
irrelevant. There is no documented or observed blob-size ceiling.

The risks are elsewhere:

- **Zot: 60-second whole-request timeouts** (§4.2). This is the one thing in this note most likely to
  cost you an afternoon.
- **Reverse proxies.** If you front the registry with nginx you must set `client_max_body_size 0`
  and generous `proxy_read_timeout`; Harbor's own template already does
  (`client_max_body_size 0`, `proxy_request_buffering off`, `proxy_read_timeout 900`), and Sonatype's
  shipped nginx example caps at `client_max_body_size 1G` — which would break every one of your
  images.
- **`max-concurrent-uploads`** on the pushing client defaults to 5; five multi-GB layers in flight is
  five times the staging space. Lower it if the dataset is tight.

### 5.7 The table

| # | Bites you when | What happens | Fix |
|---|---|---|---|
| **B1** | You install the app and start pushing | `DELETE` is refused; you can never purge an image on purpose | Additional Env Var `REGISTRY_STORAGE_DELETE_ENABLED=true` **before** first push |
| **B2** | You push digest-pinned images with no tag, then GC | `--delete-untagged` deletes them; *"this data loss is irreversible"* on Nexus, silent on `distribution` | **Always push a tag too.** The tag is the GC anchor |
| **B3** | You run GC while the app is running | Docs: *"risk that the image's layers are mistakenly deleted leading to a corrupted image"*; one open report against 3.1.1 (#4939) | Stop the app, run GC in a throwaway container against the dataset, start it. `--dry-run` first |
| **B4** | You are on 25.10 and reach for a self-signed cert | The CA screens are **gone**: *"you cannot sign CSRs or create self-signed certificates"* | Mint the CA + cert off-box, import the cert, select it in the app |
| **B5** | You choose `insecure-registries` | Six hosts to change now and six to un-change later; daemon also gains a cleartext fallback endpoint | Use TLS + `/etc/docker/certs.d/<host>:<port>/ca.crt`, which needs **no restart or reload** |
| **B6** | You edit `/etc/docker/daemon.json` on a Spark | `systemctl restart docker` kills every running deployment — `live-restore` is off by default | `systemctl reload docker`; `insecure-registries` and `registry-mirrors` are both SIGHUP-reloadable |
| **B7** | You expect `registry-mirrors` to cache ghcr/nvcr | It silently does nothing — mirrors apply to `docker.io` only | Rewrite references (A), or containerd image store + `hosts.toml` (B), Engine ≥ 28.0 |
| **B8** | You point one `distribution` at an upstream and also try to push | Pushes fail: *"a proxy registry does not support blob uploads"* | Separate app instances, separate ports, separate datasets |
| **B9** | You leave a pull-through cache unattended | Blob TTL expiry errors out and nothing is reclaimed (#4957, and independently: `deleteEnabled` gate) | Enable delete on the proxy too; set a dataset **quota**; consider `proxy.ttl: 0` + periodic manual reset |
| **B10** | You put a 39 GB image through zot | 60 s read/write timeouts kill it mid-stream; #4140 open with 10 GB failing at `15m` | Set `http.readTimeout`/`writeTimeout` to `0`, and test a full round-trip before committing |
| **B11** | You use `registry:2.8.3` (old TrueNAS chart, or Harbor's embedded core) | `--delete-untagged` destroys multi-arch child manifests (#3178, fix is 3.0+ only) | Use `registry:3.x` |
| **B12** | You pick Nexus because it is in the store | CE caps at 40k components / **100k requests per day** and then refuses new components; H2 in a container is unsupported; #1059 breaks `docker push` on 3.96.1 | Don't, for this workload |
| **B13** | You put the registry dataset on an SMB/NFS share | 25.10.0.1: *"an occasional race condition during boot that causes them to show a 'crashed' status"* | Local pool dataset |
| **B14** | You turn on ZFS dedup to "save space on shared layers" | DDT RAM tax, a slower pool, and no gain — the registry already deduplicates by digest | Leave `dedup=off`. It is a one-way door |

---

## §6 — SIZING AND OPERATIONS

### 6.1 How big the dataset needs to be

Your numbers: **~61 unique images, ~1.4 TB un-deduplicated, largest single image ~39 GB, heavy layer
sharing between variants.**

**The registry deduplicates that sharing for you, before ZFS sees it.** From `registry/storage/paths.go`:

```
<root>/v2
├── blobs
│   └── <algorithm>
│       └── <split directory content addressable storage>
└── repositories
    └── <name>
        ├── _layers        └── <layer links to blob store>
        ├── _manifests     ├── revisions/<digest>/link   └── tags/<tag>/current/link
        └── _uploads
```

> *"The storage backend layout is broken up into a content-addressable blob store and repositories.
> The content-addressable blob store holds most data throughout the backend, keyed by algorithm and
> digests of the underlying content. Access to the blob store is controlled through links from the
> repository to blobstore."*

**One blob per digest, globally, no matter how many images reference it.** [source-verified] So your
1.4 TB "un-deduplicated" figure is an upper bound and the stored size will be materially smaller —
how much smaller depends entirely on how many of those 61 images share a CUDA/vLLM base, which I
cannot compute from here. `[inference]` For a set of vLLM/CUDA overlays over a handful of bases, a
2–4× reduction would not be surprising, and I would **not** plan on it.

**Provision for:**

| Component | Size |
|---|---|
| Blobs, worst case (no sharing) | **1.4 TB** |
| In-flight uploads | up to `max-concurrent-uploads` (default **5**) × largest layer. Purged after 168 h automatically |
| GC working space | none — mark/sweep is in-memory over digests, and deletes are unlinks |
| A pull-through cache, if you run one | unbounded until B9 is solved. **Quota it separately** |

**Recommendation: a 2 TB `refquota` on the hosted registry's dataset, and a separate, smaller,
quota'd dataset per cache instance.** A quota turns "the pool filled up and the NAS misbehaved" into
"pushes started failing", which is a much better failure.

### 6.2 Does ZFS compression help or hurt? And dedup?

**Compression: leave LZ4 on. It neither helps nor hurts on the blobs, and it helps on everything
else.** OCI layers are gzip-compressed tarballs, so LZ4 will bounce off them — but OpenZFS documents
the early-abort: *"incompressible data will be stored without compression such that reads of
incompressible data with compression enabled will not be subject to decompression."* Meanwhile the
`repositories/` tree is thousands of tiny files (`link` files are a 71-byte ASCII digest each) and
JSON manifests, which compress well and would otherwise each burn a full record. `[inference]` — no
primary source benchmarks this specific workload; the mechanism is documented, the application to a
registry is my reasoning.

**Dedup: no, and this is not a close call.**

1. **The win you are imagining is already taken.** "Layers are highly shared between variants" is
   *exactly* what content-addressable blob storage handles — identical layers have identical digests
   and are stored once (§6.1). ZFS would be looking for duplication that the registry already removed.
2. **What remains is not deduplicable.** Two *different* builds of the same content produce different
   gzip streams; ZFS block-level dedup on compressed archive data finds essentially nothing.
3. **The cost is permanent and front-loaded.** TrueNAS:
   *"Pools suitable for deduplication, with deduplication ratios of 3x or more, might only need 1-3 GB
   of RAM per 1 TB of data"* — and you will not get 3×. *"High-quality mirrored SSDs configured as a
   special vdev for the DDT (and usually all metadata) are strongly recommended."* *"A deduplicated
   pool does not reach the same speeds as a non-deduplicated pool."* And the door only opens one way:
   *"The only way to convert existing current data to all deduplicated or non-deduplicated… is to
   create a new copy while new settings are active."*
4. iX's own summary is the last word: *"When data is not sufficiently duplicated, deduplication wastes
   resources, slows the server down, and has no benefit."*

*(Zot is the one option where a filesystem feature does matter: its `dedupe` uses POSIX hardlinks
(`os.Link`), which ZFS supports. But it is doing the same job the blob store already does, at the
OCI-layout level.)*

### 6.3 Backup and replication

A registry dataset is unusually friendly to ZFS replication, for one reason: **blobs are immutable and
content-addressed**, so an incremental `zfs send` after a week of pushes carries almost exactly the
new blobs and nothing else. There is no rewrite-in-place, no database file churn.

- **Periodic Snapshot Task** on the dataset — daily, keep ~14. This is also your GC undo: if a GC run
  eats something it shouldn't (B3), a snapshot from before the run is a complete recovery, which is
  *not* true for Nexus (*"this data loss is irreversible"*) or Harbor (*"Harbor will not restore any
  artifact that has already been garbage collected"*).
- **Replication Task**, local or remote. iX documents both
  ([Replication Tasks](https://www.truenas.com/docs/scale/dataprotection/replication/)); remote needs
  SSH access for the admin user on the target, a key, a home directory and sudo permission, and the
  SSH service running when the task fires.
- **Snapshot *before* GC, not after.** Order the cron job so the snapshot task has already run.
- One nuance worth planning around: snapshots hold deleted blobs on disk. A GC run that frees 300 GB
  frees nothing until the snapshots referencing it expire. Size the quota with that in mind, or
  accept a two-week lag between GC and actual free space. `[inference]` — standard ZFS behaviour,
  applied here.

---

## §7 — AVAILABILITY: registry downtime *is* deploy downtime

You are right to raise it, and the honest answer is that a single-box registry is a single point of
failure for deploys. What varies is how gracefully each failure degrades.

| Failure mode | What happens | Mitigation |
|---|---|---|
| **App restarted by an app-version update** | Seconds of downtime. In-flight pulls fail; Docker retries | Update apps deliberately, not during a deploy window |
| **TrueNAS OS upgrade / reboot** | Minutes to **a quarter of an hour**: 25.10 extended the Docker service timeout *"to 960 seconds (16 minutes) to accommodate slower disk scenarios"* | Don't upgrade the NAS and the cluster in the same window |
| **Apps pool not imported / dataset unmounted** | The app cannot start; `distribution` fails its healthcheck. `midclt call docker.config` shows no pool | Keep the registry dataset on the **apps pool**, or at least the same pool, so there is one thing to be up. Never on an SMB/NFS share (B13) |
| **Pool degraded** | ZFS keeps serving from a degraded vdev; reads slow, nothing stops | Ordinary pool hygiene; a scrub task |
| **Pool faulted / unrecoverable** | Registry gone. Every deploy that needs a non-local image fails | This is the real risk. §7.1 |
| **The NAS is simply off** | Same | §7.1 |
| **Upstream (ghcr/nvcr) is down**, if you run a cache | **A `distribution` proxy keeps serving what it has cached** — *"If the remote is unavailable the local association is returned"* [source-verified] | This is a genuine availability *gain* over pulling upstream directly |

### 7.1 What actually protects a deploy

**The strongest mitigation is not high availability — it is that you pin by digest.** A digest-pinned
image that is already in a node's local image store is never fetched again. The registry is only on
the critical path for a **first** pull on a given node. Concretely:

1. **Pre-pull before you need it.** After pushing a new overlay, pull it to all four Sparks
   immediately, while you are watching. Then the registry being down at deploy time is harmless for
   that image. This is worth wiring into whatever ships a new build.
2. **Do not garbage-collect node-local images aggressively.** The stated goal of this registry is *"so
   they become safe to delete from node-local disk"* — which is right for the 39 GB one-offs, but
   keeping the two or three images you actually deploy resident is the cheapest availability you will
   ever buy.
3. **Keep a fallback path in the recipe.** If a recipe can name either `nas.lan:30095/foo@sha256:…`
   or `ghcr.io/spark-arena/foo@sha256:…`, a dead NAS is an edit, not an outage. The digest is the same
   either way — that is the whole point of content addressing.
4. **Snapshots + replication** (§6.3) turn "pool faulted" from data loss into downtime.

### 7.2 What does *not* help, and why

- **Running two `distribution` instances against one dataset.** Two registries writing one blob store
  is not a supported topology, and `REGISTRY_HTTP_SECRET` exists precisely because upload state is
  shared across instances that sit behind a load balancer. Don't.
- **Zot scale-out.** Documented, but it wants shared S3 plus a shared remote cache (DynamoDB/Redis),
  an identical ordered `cluster.members` list on every node, and is *"not self-healing when an
  instance fails"*. Its own docs say BoltDB — the default cache — rules out replicas outright:
  *"Because BoltDB does not provide concurrent access for writes, multiple instances/replicas of zot
  are not supported with a BoltDB configuration."* And there is an unfixed silent tag-loss race in the
  naive shared-storage topology ([PR #4337](https://github.com/project-zot/zot/pull/4337): *"Two zot
  instances sharing one storage backend can silently destroy each other's tags… Both pushes return
  `201`… There is no error anywhere for an operator to find."*).
- **Harbor HA.** Real, and it needs external PostgreSQL, external Redis and shared object storage.
  Nothing about that belongs on one NAS.

**Single instance is the right answer for six clients.** Buy availability with pre-pulls and
snapshots, not with a second registry.

---

## §8 — VERDICT

**Install the official `distribution` app twice: once as your hosted registry, once (optionally) as a
pull-through cache per upstream. Give each a dedicated ZFS dataset with `recordsize=1M`, `atime=off`,
LZ4 on, dedup off, and a `refquota`. Put TLS on it with an imported certificate. Turn on delete
before the first push. Always push a tag alongside the digest. Schedule offline GC weekly with the
app stopped.**

Not Nexus, despite it being in the store: the CE usage cap, the unsupported-H2-in-a-container
posture, the digest-deleting GC, and an open `docker push` → 500 against the exact shipped version are
four independent reasons, any one of which would be enough. Not Harbor: nine containers and a
PostgreSQL migration per upgrade to get features you don't need, plus a proxy-cache/multi-arch bug
cluster in the one area you'd adopt it for. Zot is the most *elegant* answer and the one I'd
reconsider if the 60-second-timeout situation resolves — but you have 39 GB images and it has an open
issue about 10 GB images, and it isn't packaged.

### 8.1 Step-by-step for the most likely case (SCALE/CE 25.10 Goldeye)

**0 — Confirm the ground truth.**
```bash
cat /etc/version                 # expect 25.10.x
midclt call docker.config        # confirm the apps pool is set
docker version --format '{{.Server.Version}}'    # on each Spark — >= 28.0 matters for §5.5
```

**1 — Datasets, before touching the Apps UI.** Storage → create, under your data pool:
```
tank/registry          recordsize=1M  compression=lz4  atime=off  dedup=off  refquota=2T
tank/registry-cache    (only if you want §5.5; same properties, refquota=500G)
```

**2 — Certificate.** On the Pi, make a CA and a server cert with SANs for **both** `nas.lan` and the
NAS's IP (clients that use an IP will reject a name-only cert). Import the **server cert + key** into
TrueNAS: Credentials → Certificates → Import. Keep the CA cert; you will hand it to six machines.
*(On 25.04 or earlier you could have done this entirely in the TrueNAS UI — on 25.10 you cannot, §5.2.)*

**3 — Install the app.** Apps → Discover → **Distribution** → Install.
- **Name:** `registry-hosted`
- **HTTP Secret:** generate a long random string
- **Basic Auth Users:** one user (remember: all-or-nothing access, §5.3)
- **Additional Environment Variables:** `REGISTRY_STORAGE_DELETE_ENABLED` = `true` ← **do not skip**
- **API Port:** `30095`, Publish, Host IPs = the LAN address
- **Certificate ID:** the certificate you imported in step 2
- **Storage:** Host Path → `/mnt/tank/registry`, **Enable ACL**, grant uid/gid **568**
- **Resources:** 2 CPU / 4096 MB is fine

**4 — Client trust, on each Spark, the eval host and the Pi.**
```bash
sudo mkdir -p /etc/docker/certs.d/nas.lan:30095
sudo cp lan-ca.crt /etc/docker/certs.d/nas.lan:30095/ca.crt
docker login nas.lan:30095
# no restart, no reload
```

**5 — Prove it, in this order.**
```bash
curl -u user:pass https://nas.lan:30095/v2/_catalog                 # 200 {"repositories":[]}
docker pull alpine && docker tag alpine nas.lan:30095/smoke:v1
docker push nas.lan:30095/smoke:v1                                  # small push works
# multi-arch round trip — the thing you actually depend on
docker buildx imagetools create -t nas.lan:30095/smoke:multi \
  alpine:latest@sha256:<amd64-digest> alpine:latest@sha256:<arm64-digest>
docker buildx imagetools inspect nas.lan:30095/smoke:multi          # both platforms present
# then, on a Spark and on the amd64 host, pull the SAME tag and check you each got your own arch
# finally, the real test: push one 39 GB overlay end to end and time it
```

**6 — Push the real images, with a tag *and* record the digest.**
```bash
docker tag <local> nas.lan:30095/dgx-vllm-eugr-nightly:2026090501
docker push nas.lan:30095/dgx-vllm-eugr-nightly:2026090501
docker buildx imagetools inspect nas.lan:30095/dgx-vllm-eugr-nightly:2026090501 \
  --format '{{.Manifest.Digest}}'     # this digest goes in the recipe; the tag keeps GC off it (B2)
```

**7 — GC, as a weekly Cron Job** (System → Advanced → Cron Jobs, run as `root`):
```sh
midclt call -job app.stop registry-hosted
docker run --rm -v /mnt/tank/registry:/var/lib/registry \
  -e REGISTRY_STORAGE_FILESYSTEM_ROOTDIRECTORY=/var/lib/registry \
  registry:3.1.1 garbage-collect --delete-untagged /etc/distribution/config.yml
midclt call -job app.start registry-hosted
```
Run it **once by hand with `--dry-run`** first and read the output. Put the periodic snapshot task
*before* it in the schedule.

**8 — Optional: a pull-through cache.** Install **Distribution** a second time as `registry-cache-ghcr`
on port `30096`, Host Path `/mnt/tank/registry-cache`, with Additional Environment Variables
`REGISTRY_PROXY_REMOTEURL=https://ghcr.io`, `REGISTRY_PROXY_TTL=0`,
`REGISTRY_STORAGE_DELETE_ENABLED=true` (B9). Repeat per upstream — one instance each. Then either
rewrite references to `nas.lan:30096/…`, or, if your Sparks are on Engine ≥ 28 with the containerd
image store, drop the `hosts.toml` from §5.5 and change nothing else.

**9 — Data protection.** Periodic Snapshot Task on `tank/registry`, daily, keep 14. A Replication
Task if you have anywhere to send it.

### 8.2 If you are not on 25.10

- **25.04 Fangtooth / 24.10 Electric Eel** — identical, except step 2 is easier: the TrueNAS UI can
  still create a CA and sign a server certificate. Check `min_scale_version: 24.10.2.2`.
- **24.04 and earlier** — upgrade first. The chart there is `registry:2.8.3` and is on the wrong side
  of the multi-arch GC fix (B11).
- **CORE 13.x** — do not do this on CORE. Put the registry on a Linux host.
- **26 BETA** — should be identical (Docker Engine 29.0.4), but it is a beta and I have no evidence
  either way about the `distribution` app on it.

---

## §9 — What I could NOT determine

- **Which TrueNAS version you are actually on.** I was told "the latest", which in September 2026
  means **25.10 Goldeye** — but 26 is in BETA and someone running "latest" could mean that. §1 gives
  the commands; §8.2 gives the deltas. **Every version-dependent claim above is labelled.**
- **What your Sparks run.** Docker Engine version and whether the containerd image store is enabled
  decides §5.5 entirely. Three commands in §1.3.
- **Whether `distribution`'s proxy authenticates against `nvcr.io`.** The mechanism is a standard
  bearer challenge and NGC uses `$oauthtoken` + API key, so it should; no primary source confirms it,
  and I had no hardware to test on. Nexus is the only option with nvcr.io **in its documentation**.
- **Whether a 39 GB push survives end to end through the TrueNAS app.** The code path has no size
  limit and commits by rename, and I found no issue reporting a ceiling — but "no report of failure"
  is not "report of success". This is step 5's last line for a reason.
- **Whether #4939 (GC deleting live blobs on 3.1.1) is real.** One report, no comments, no repro,
  and the reporter was doing the thing the docs warn against. I could not confirm or refute it.
- **The actual deduplication ratio of your 61 images.** It is knowable — sum the unique blob digests
  across their manifests — but it needs the images, not the internet.
- **Whether zot handles a 39 GB blob.** Largest failure on record is 10 GB, unresolved, on a recent
  release.
- **Whether Harbor's generic `docker-registry` provider works against nvcr.io today.** One user said
  yes in 2023; the allow-list (`PERMITTED_REGISTRY_TYPES_FOR_PROXY_CACHE`) does include
  `docker-registry`, so it is at least permitted.
- **When Harbor v2.16.0 (the first multi-arch release) ships.** No branch, no dated milestone. Moot
  for TrueNAS, which is x86-64 only.
- **Nexus ticket status for anything not in a release-note table.** `issues.sonatype.org` was
  decommissioned in January 2024; there is no public Jira to check.

---

## Sources

**iXsystems (primary):**
[Software Status](https://www.truenas.com/docs/softwarestatus/) ·
[25.10 (Goldeye) Version Notes](https://www.truenas.com/docs/scale/25.10/gettingstarted/versionnotes/) ·
[26 Version Notes](https://www.truenas.com/docs/scale/26/gettingstarted/versionnotes/) ·
[24.10 (Electric Eel) Version Notes](https://www.truenas.com/docs/scale/24.10/gettingstarted/scalereleasenotes/) ·
[25.04 (Fangtooth) Version Notes](https://www.truenas.com/docs/scale/25.04/gettingstarted/scalereleasenotes/) ·
[Apps UI Reference (25.10)](https://www.truenas.com/docs/scale/25.10/scaleuireference/apps/) ·
[Custom App Screens](https://www.truenas.com/docs/scale/25.10/scaleuireference/apps/installcustomappscreens/) ·
[Installing Custom Apps](https://apps.truenas.com/managing-apps/installing-custom-apps/) ·
[App Storage](https://apps.truenas.com/getting-started/app-storage/) ·
[Initial Setup (registry mirrors)](https://apps.truenas.com/getting-started/initial-setup/) ·
[Managing Cron Jobs](https://www.truenas.com/docs/scale/25.10/scaletutorials/systemsettings/advanced/managecronjobsscale/) ·
[Certificates (25.10)](https://www.truenas.com/docs/scale/25.10/scaletutorials/credentials/certificates/certificatesscale/) ·
[Replication Tasks](https://www.truenas.com/docs/scale/dataprotection/replication/) ·
[Adding and Managing Datasets](https://www.truenas.com/docs/scale/datasets/managingdatasets/) ·
[ZFS Deduplication](https://www.truenas.com/docs/references/zfsdeduplication/) ·
[Hardware Guide](https://www.truenas.com/docs/scale/gettingstarted/scalehardwareguide/) ·
[Using the Shell](https://www.truenas.com/docs/scale/systemsettings/shell/usescaleshell/) ·
[Developer Mode (Unsupported)](https://www.truenas.com/docs/scale/systemsettings/advanced/developermode/)

**TrueNAS catalog source (read via the GitHub API):**
[`truenas/apps`](https://github.com/truenas/apps) → `ix-dev/community/distribution/{app.yaml,ix_values.yaml,questions.yaml,app_migrations.yaml,templates/docker-compose.yaml}`,
`ix-dev/community/nexus/{app.yaml,ix_values.yaml,questions.yaml,templates/docker-compose.yaml}`,
`library/2.3.11/functions.py`; full enumeration of `ix-dev/{stable,community,enterprise,test,dev}` (442 apps) ·
[`truenas/charts` → `community/distribution`](https://github.com/truenas/charts/tree/master/community/distribution) ·
[`truenas/middleware` → `etc_files/docker/daemon.json.py`](https://github.com/truenas/middleware/blob/master/src/middlewared/middlewared/etc_files/docker/daemon.json.py) ·
[truenas/apps #4876](https://github.com/truenas/apps/issues/4876)

**TrueCharts / trueforge (their own announcements):**
[scale-deprecation](https://truecharts.org/news/scale-deprecation/) ·
[scale-faq](https://truecharts.org/news/scale-faq/) ·
[leaving-scale](https://truecharts.org/news/leaving-scale/)

**CNCF `distribution`:** docs —
[configuration](https://distribution.github.io/distribution/about/configuration/),
[garbage collection](https://distribution.github.io/distribution/about/garbage-collection/),
[mirror recipe](https://distribution.github.io/distribution/recipes/mirror/);
source at `v3.1.1` — `registry/handlers/app.go`, `registry/proxy/{proxyregistry,proxymanifeststore,proxytagservice,proxyblobstore}.go`,
`registry/storage/{garbagecollect,linkedblobstore,paths}.go`, `registry/storage/driver/filesystem/driver.go`,
`registry/auth/htpasswd/{access,htpasswd}.go`, `registry/api/v2/routes.go`;
[v3.0.0 release notes](https://github.com/distribution/distribution/releases/tag/v3.0.0),
[v3.1.0 release notes](https://github.com/distribution/distribution/releases/tag/v3.1.0);
issues/PRs #2367, #3178, #3530, #3716, #3725, #3851, #4249, #4285, #4939, #4957 ·
[distribution-library-image Dockerfile + entrypoint.sh](https://github.com/distribution/distribution-library-image)

**Docker / moby:**
[dockerd reference](https://docs.docker.com/reference/cli/dockerd/) ·
[Mirror the Docker Hub library](https://docs.docker.com/docker-hub/image-library/mirror/) ·
[containerd image store](https://docs.docker.com/engine/storage/containerd/) ·
[live restore](https://docs.docker.com/engine/daemon/live-restore/) ·
[Engine 29 release notes](https://docs.docker.com/engine/release-notes/29/) ·
moby source — `daemon/hosts.go`, `daemon/pkg/registry/{service,service_v2,config}.go`, `daemon/reload.go`,
and the `daemon/images` vs `daemon/containerd` pull paths; commits `8b4cb6f5`, `b3569ebd`, `2aaae08a`;
issues #18818, #41456, #42433, #43794 ·
[containerd `docs/hosts.md`](https://github.com/containerd/containerd/blob/main/docs/hosts.md)

**Harbor:**
[installation prerequisites](https://goharbor.io/docs/latest/install-config/installation-prereqs/) ·
[garbage collection](https://goharbor.io/docs/latest/administration/garbage-collection/) ·
[configure proxy cache](https://goharbor.io/docs/latest/administration/configure-proxy-cache/) ·
[upgrade](https://goharbor.io/docs/latest/administration/upgrade/) ·
[compatibility list](https://goharbor.io/docs/latest/install-config/harbor-compatibility-list/) ·
source at `v2.15.2` — `make/photon/prepare/templates/docker_compose/docker-compose.yml.jinja`,
`make/photon/prepare/templates/core/env.jinja`, `src/server/v2.0/handler/project.go`, `Makefile`;
issues #3505, #11548, #15807, #16718, #16776, #17230, #18014, #18132, #20920, #21454, #22848, #23178,
#23185, #23199, #23466, #23570, #23753, #23803, #23882; PRs #22311, #23355

**Zot:**
[storage](https://zotregistry.dev/v2.1.21/articles/storage/) ·
[mirroring](https://zotregistry.dev/v2.1.21/articles/mirroring/) ·
[retention](https://zotregistry.dev/v2.1.21/articles/retention/) ·
[security posture](https://zotregistry.dev/v2.1.21/articles/security-posture/) ·
[high availability](https://zotregistry.dev/v2.1.21/articles/high-availability/) ·
[scale-out](https://zotregistry.dev/v2.1.21/articles/scaleout/) ·
source — `pkg/api/config/config.go`, `pkg/cli/server/root.go`, `pkg/storage/storage.go`,
`pkg/extensions/sync/destination.go`, `examples/config-popular-registries.json`;
issues/PRs #1866, #2117, #3846, #4062, #4079, #4140, #4149, #4271, #4323, #4337, #4349, #4399, #4404, #4410 ·
[CNCF Sandbox acceptance](https://lists.cncf.io/g/cncf-toc/message/7743)

**Sonatype Nexus:**
[Docker Registry](https://help.sonatype.com/en/docker-registry.html) (path-based routing, NVIDIA NIM) ·
[Proxy Repository for Docker](https://help.sonatype.com/en/proxy-repository-for-docker.html) ·
[Tasks](https://help.sonatype.com/en/tasks.html) ·
[Cleanup Policies](https://help.sonatype.com/en/cleanup-policies.html) ·
[Database options](https://help.sonatype.com/en/database-options.html) ·
[Usage Center](https://help.sonatype.com/en/usage-center.html) ·
[OCI Repositories](https://help.sonatype.com/en/oci-repositories.html) ·
[System Requirements](https://help.sonatype.com/en/sonatype-nexus-repository-system-requirements.html) ·
[3.96.0–3.96.1 release notes](https://help.sonatype.com/en/sonatype-nexus-repository-3-96-0-release-notes.html) ·
[issues.sonatype.org decommission FAQ](https://central.sonatype.org/faq/what-happened-to-issues-sonatype-org/) ·
`sonatype/nexus-public` issues #297, #468, #475, #613, #659, #967, #1040, #1059, #1061 ·
community threads 1753, 2719, 6871, 10004, 13831, 16311, 16322

**OpenZFS / NVIDIA:**
[OpenZFS Workload Tuning](https://openzfs.github.io/openzfs-docs/Performance%20and%20Tuning/Workload%20Tuning.html) ·
[NGC Private Registry User Guide](https://docs.nvidia.com/ngc/latest/ngc-private-registry-user-guide.html)
