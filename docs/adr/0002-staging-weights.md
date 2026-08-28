# 2. Staging weights is its own concern, not a step inside deployment

Date: 2026-08-15

## Status

Accepted.

## Context

Deploying a model whose weights nobody had downloaded failed, and failed badly.

`dgxrun` bind-mounts the shared cache and sets `HF_HUB_OFFLINE=1`, on the
reasoning that cluster weights are pre-staged on NFS. Nothing checked that they
were. A deploy of an un-staged model therefore launched a container that ran for
a minute and died with a `huggingface_hub.errors.LocalEntryNotFoundError`
traceback, surfaced verbatim as the deployment's error string. Reading it and
concluding "nobody has downloaded this model yet" requires knowing what
`local_files_only` means.

Nor was there any remedy inside the product. The HF cache API could list,
rescan, and **delete** cached repos, but not fetch one — so a user could evict a
model from the models page and had no way to bring it back. Every model on the
cluster arrived by someone running a download over SSH by hand.

Meanwhile `sparkrun`, the other runner, has the opposite failure. It sets
`HF_HOME` and never sets `HF_HUB_OFFLINE`, so the same deploy *succeeds* — after
silently occupying a GPU node for the hours it takes to pull tens of gigabytes,
with no warning that this is what was about to happen and no progress anywhere.

So one runner failed opaquely and the other committed the cluster to hours of
invisible work. Both are the same missing concept.

## Decision 1 — weights belong to a cache group, not a node

A **cache group** is the set of nodes that see the same weights because they
share one storage location. The four Sparks mount the same NFS export and are
therefore one group; agenthost has no shared cache and is in none.

Staging targets the group. This is not a detail of addressing — asking whether a
*node* has a model is the wrong question, because the answer is a property of
its group, and "downloading it for the second node" would be the same bytes
written to the same path. Modelling it per-node would have forced a choice
between three redundant multi-hour downloads and three instant no-ops that mean
something the API cannot explain.

The group already existed in the code, discovered from a `.dgx-cache-id` marker
at the `HF_HOME` root, because inventories had to be de-duplicated for display.
This decision promotes it from a display concern to the thing staging addresses.

## Decision 2 — a staging job is an entity, not a phase of a deployment

The obvious cheap route was to reuse the Ollama pull path: it already reports
download progress to the dashboard. But that pipeline is keyed by
`deploymentId` and broadcasts `deployment:progress` with a phase — it models
downloading as something a deployment *does*.

Weights are not that. They outlive any particular deployment, are often wanted
before anyone has decided what to run, and their absence is a decision for the
user to make rather than an error to report. So a **staging job** is its own
entity, with its own persisted lifecycle and its own `staging:*` events.

A deployment may *wait on* a staging job. It never *owns* one. That is why a
deploy against un-staged weights is rejected outright rather than parked in an
`awaiting-weights` state: parking it would mean a deployment holding a node for
hours on the strength of a VRAM admission decision taken before the download
started. It is also why staging completing does not trigger the deployment. The
wish "deploy this once the weights land" is durable, outlives the browser tab,
and would have to become a second persisted entity with its own reconciliation —
a large cost to save one click.

The job is persisted rather than tracked in memory because a 52 GiB fetch takes
around half an hour at best and hours at worst, while this server restarts
routinely — a dashboard rebuild recreates it through `depends_on`, and a missed
NFS automount kills it on every host reboot. An in-memory job would leave a
download running on a node with nothing in the product aware of it.

## Decision 3 — dgxrun blocks, sparkrun warns

Un-staged weights are a guaranteed failure under dgxrun and a slow success under
sparkrun. One rule applied to both would either block sparkrun deploys that work
today, or permit dgxrun deploys that cannot.

So the gate follows each runner's actual behaviour: dgxrun rejects, sparkrun
warns that a download of a stated size is about to happen and asks whether to
continue. The asymmetry is deliberate and it is not a transitional state.

It is also honest about a limit. We know a recipe's model, but sparkrun fetches
whatever else its registry recipe decides to — a drafter, a tokenizer, something
we do not model. We can therefore inform accurately and cannot gate completely,
and a warning claims exactly as much as we can support.

## Decision 4 — "is it staged?" is answered by the runtime's own resolution

A repo that is two percent downloaded is a directory with the right name and a
plausible size; the cache inventory reports it as present. A gate built on
presence would pass, and the deploy would then fail with exactly the traceback
this work exists to remove.

The check is therefore the same call the runtime makes — resolve the snapshot
with `local_files_only` — run on a node in the group. Its value is not accuracy
in the abstract but that it *cannot disagree with the launch*: if the check
passes, the launch cannot fail for this reason. A cheaper check against the
last-pushed inventory would have been fast, stale, and unable to tell a
half-downloaded model from a whole one.

For the same reason there is no completeness marker written by the staging job.
A marker would be unambiguous for anything staged through this system and would
misreport every model on the cluster today, all of which were staged by hand.

## Consequences

- A new persisted entity, a new `staging:*` event family, and a new agent
  command. The staged-check adds one agent round-trip to the deploy path.
- The product gains an image it owns and builds: a small container carrying
  `huggingface_hub` and `hf_xet`. It exists so staging works before any recipe
  is chosen, which rules out reusing the recipe's own multi-gigabyte image.
- The agent must be able to `docker pull` that image. This is a deliberate
  exception to dgxrun's "v1 does not distribute images" rule, taken on the
  grounds that a 200 MB pull is seconds and a model is hours — the rule exists
  to stop the latter happening invisibly, not the former.
- Jobs are serialised per cache group. A group has one pipe and one filesystem,
  so concurrent large fetches finish no sooner and report meaningless ETAs.
- Cancelling keeps `.incomplete` blobs so re-staging resumes. Reclaiming the
  space stays an explicit act through the existing delete endpoint, which means
  a cancelled job leaves disk consumed until someone says otherwise.
- A staging job records the commit sha it resolved, so it is answerable later
  which weights a benchmark ran against. The deploy gate still asks the
  runtime's offline resolution, so a new upstream commit never silently
  un-stages a model already held.
