# 3. Images are the product's concern, not a prerequisite satisfied by hand

Date: 2026-09-14

## Status

Proposed.

## Context

Deploying a recipe whose image nobody had built fails, and fails in the same
shape ADR 0002 describes for weights — except that nothing has been done about
it yet.

`dgxrun` checks the image is present locally and refuses otherwise:

```
[dgxrun] image "vllm-node-tf5-glm52-b12x:probe" not found locally.
v1 does not distribute images; build/load it on this node first
(docker load / registry pull).
```

That is an honest message and a dead end. The recipe does not say where the
image comes from; the build script that produces it
(`scripts/build-glm52-image.sh`) is committed but referenced from nowhere the
deploy path can see. So a fresh cluster asked to serve GLM-5.2 at TP4 fails on
all four ranks within seconds, and recovery means reading the recipe to learn
the image name, finding the build script by grep, building it, propagating the
result to four nodes by rebuild or `docker save`/`load`, and only then
discovering the second wall — that the weights are not staged either.

**Every image on the cluster today arrived by someone building or pulling it
over SSH, per node, by hand.** That is the same sentence ADR 0002 wrote about
weights, and it has the same consequences.

It also has one that weights do not. Because each node resolves a tag
independently, at whatever time someone happened to run `docker pull`, the same
tag can mean different bytes on different machines. This is not hypothetical:
the live `qwen3.8-27b-nvfp4` pool was found serving from two different images
under one **dated** tag —

```
ghcr.io/spark-arena/dgx-vllm-eugr-nightly:2026081501
  spark-03 -> sha256:d56faba2c44f...  (23.0 GB)
  spark-04 -> sha256:d2eb44d303ba...  (34.2 GB)
```

— and the hand-built `:probe` images that the 4x recipes reference drift the
same way, 38.7 GB on two nodes and 19.2 GB on the other two. A tensor-parallel
deploy can therefore run different code on different ranks, and nothing in the
product would say so. Dating the tag was the mitigation we adopted for this and
it is insufficient, because the tag is mutable upstream.

The cost of the absent concept is also plainly measurable. Across the four
Sparks there are 79 image instances, 61 unique, 1839 GB stored, of which 391 GB
is redundant copies of images that were each downloaded once per node. A further
1088 GB is hand-built images referenced by no recipe, which nobody deletes
because they exist nowhere else and rebuilding one is expensive.

## Decision 1 — an image is identified by digest; a tag is a convenience

Recipes name images by tag today. A tag is a mutable pointer, resolved
independently on every node, and we have direct evidence that it drifts — on a
dated tag, on production replicas, unnoticed.

An image is therefore identified by its **digest** (`sha256:...`) everywhere the
identity matters: what a recipe pins, what a deployment records, what ranks are
checked against. Tags remain for human use and for following an upstream stream
deliberately, but a deploy resolves a tag to a digest **once**, centrally, and
every rank is given that digest.

The alternative — keep tags and pull everywhere at the same moment — narrows the
window without closing it, and gives no way to answer afterwards which bytes a
benchmark ran against.

## Decision 2 — an image build is an entity, not a phase of a deployment

This mirrors ADR 0002 Decision 2 and for the same reasons, which are worth
restating because the temptation is the same: the deploy path already reports
progress, so a build could be modelled as something a deployment does.

It is not. An image outlives any deployment, is shared by several recipes, is
often wanted before anyone has chosen what to run, and takes tens of minutes —
long enough that a server restart must not lose it. So a **build job** is its
own persisted entity with its own `build:*` events.

A deployment may *wait on* a build. It never owns one. As with staging, a build
completing does not trigger a deployment: "deploy this once the image is ready"
is a durable wish that would need its own persisted reconciliation, and that is
a large cost to save one click.

## Decision 3 — the registry is the system of record; a node holds a cache

Today the node *is* the system of record: if an image is not on a node it does
not exist, and if it is on four nodes it exists four times with no guarantee
they agree.

A registry inverts that. It holds the authoritative, content-addressed copy; a
node's local image store is a cache that can be emptied without loss. This is
what makes node disk reclaimable — the 1088 GB of unreferenced hand-built images
is unreclaimable today purely because deleting one might be unrecoverable.

Two consequences follow deliberately. First, the registry must be at least as
available as deploys need to be, because once `dgxrun` pulls from it, registry
downtime is deploy downtime. Second, **promotion is explicit**: an image enters
the registry because someone or something decided it is worth keeping. Mirroring
every image ever built would move the hoarding problem rather than solve it.

## Decision 4 — "is the image present?" is answered by the digest, not by a name

ADR 0002 Decision 4 rejected a presence check for weights because a repo that is
two percent downloaded is a directory with the right name. The image equivalent
is worse, because a *complete* image with the right name can still be the wrong
image — that is exactly what #103 is.

So the check resolves the image and compares its **digest** against the one the
deploy pinned, on each target node. Its value, as with weights, is that it
cannot disagree with the launch: if it passes, every rank is running the same
bytes, and that is a claim `docker images | grep` cannot make.

For a multi-rank deploy this check is also a **gate between ranks**, not merely
per-rank: ranks resolving different digests is a failure even though each rank
individually has "an image with the right name".

## Decision 5 — a deploy preflights images and weights together, and returns a plan

ADR 0002 gave weights a 409 with a machine-readable body so the UI could offer
staging. That is right, but on a fresh cluster it solves half the problem and
the user meets the other half one failure later.

A deploy therefore preflights **both**, and when something is missing it returns
a *plan* rather than an error: which image must be built or pulled, which
weights staged, how large, roughly how long, and to which nodes. The user
approves once; the work then proceeds unattended as tracked jobs.

This is the "one click even if it takes hours" property. It is a UX decision
with an architectural requirement behind it — every step must be independently
resumable and idempotent, because the plan spans hours and this server restarts
routinely.

It does not extend to deciding *for* the user. A plan that would download 388 GB
and occupy four nodes for an afternoon is exactly the kind of thing ADR 0002
insisted a user should choose rather than discover.

## Consequences

- A new persisted entity (build job), a new `build:*` event family, and a new
  agent command for pulling a pinned digest. The image check adds one agent
  round-trip to the deploy path, alongside the staged-check ADR 0002 adds.
- **`dgxrun`'s "v1 does not distribute images" rule is retired.** ADR 0002
  already carved an exception for the hf-stager image on the grounds that a
  200 MB pull is seconds and a model is hours. That reasoning does not extend to
  a 34 GB vLLM image, so this is a genuine reversal rather than an extension:
  the rule existed to stop hours of invisible work, and a tracked job with
  progress is not invisible.
- A registry becomes infrastructure the cluster depends on, with the
  availability obligation that implies. Its host must be always-on and must not
  be a GPU node.
- Recipes referencing a digest are reproducible but no longer self-updating. A
  recipe that wants the newest nightly must say so explicitly, and something
  must re-resolve it — that is a deliberate trade of convenience for the ability
  to answer which bytes ran.
- Existing recipes pin tags. They keep working, and gain the drift they have
  today, until each is migrated. Migration is per-recipe and can be done
  opportunistically, but benchmark and multi-node recipes should go first.
- Build jobs are serialised per builder. Building two 34 GB images at once on
  one machine finishes no sooner and competes for the same disk.
- The build itself stays a committed script. The product orchestrates and
  records it; it does not own the Dockerfile's contents, which are hardware- and
  vendor-specific and change for reasons the product cannot model.
- Deleting an image from a node stops being a judgement call, which is the point.
  Reclaiming registry space in turn becomes a new, explicit chore — registries do
  not garbage-collect deleted layers on their own.

## Related

- ADR 0002 — staging weights; this ADR is deliberately its mirror, and the two
  meet in Decision 5
- #103 — the digest drift that motivates Decision 1, found on live replicas
- #102 — the registry that Decision 3 requires
- #101 — fleet disk reporting; Decision 3 is what makes the reported
  "reclaimable" figure actionable
