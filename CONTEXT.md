# Context

The domain glossary for DGX Manager. Terms only — no implementation detail, no
plans. If a term here disagrees with the code, one of them is wrong; say so.

## Inference reachability

### Gateway

The single network address through which clients reach models running on the
cluster. Routes a request to a deployment by the **published name** carried in
the request body, and aggregates the cluster's model list.

A gateway is *not* a load balancer: balancing across replicas is one behaviour
it exhibits when a **pool** has more than one member, not the reason it exists.
Its reason for existing is that clients should need one address and one name,
not a node IP and a port that changes when a deployment moves.

The gateway speaks the OpenAI API and nothing else. A backend runtime's own API
is never exposed through it, and which runtime serves a published name is not
observable to a client — a name backed by Ollama and a name backed by vLLM are
indistinguishable. Consequently the gateway serves a fixed set of operations
rather than forwarding whatever a runtime happens to support: an operation a
client cannot name cannot be reached, so a runtime gaining new endpoints never
widens what the cluster exposes.

This is why the cluster's model list is answered from the manager's own record
of what is deployed, never by asking a node what it holds. A node may hold
models nobody deployed; those are not part of the cluster's published surface
and must not be discoverable through it.

### Published name

The name a client puts in the `model` field to reach a deployment. It is
whatever name the deployment's runtime itself answers to.

It belongs to a *serving lifetime*, not to the deployment for all time: it is
established when the deployment starts serving, and a deployment that has
stopped or failed has no published name. A deployment that starts again
establishes it afresh, so a rename or a changed recipe cannot leave the old
name published.

An **eviction** does not end a serving lifetime. An allocation-inducing runtime
unloads a model it has not been asked for lately and loads it again on demand,
still answering to the same name — so the name survives, and the deployment is
merely not serving *at this moment*.

It is never rewritten in flight: the name a client sends is the name the
runtime receives. This is why it must be discovered rather than assumed. A
pinned runtime is authoritative about its own name and is asked; an
allocation-inducing runtime hosts many models behind one address, so the name
comes from the deployment's model tag instead.

### Pool

The set of running deployments sharing one published name. A request for that
name may be served by any member. A pool of one is the ordinary case; a pool
forms implicitly when a second deployment claims the same published name, and
dissolves when it stops.

### Runner

The mechanism that launches a deployment's runtime on a node and supervises it
for as long as it serves. A deployment names exactly one runner, and the runner
owns how the model is launched — the manager never does.

Two exist. **sparkrun** is a third-party tool the node invokes; the recipes it
launches come from registries it clones and refreshes on its own schedule, so
what a recipe means can change without anything in this repository changing.
**dgxrun** is ours: the recipe is a file here, and the launch is expressed
directly. That difference is the whole point of having a second runner — a
dgxrun deployment can be reproduced on a node that has never been touched.

### Mod

A named change applied to a runtime *before* it begins serving, giving it a
behaviour its own image does not have. A recipe declares the mods it needs, and
they are part of what the deployment *is*: the same recipe without its mods is a
different deployment, not a degraded one.

A mod is not configuration. Configuration selects among behaviours a runtime
already has; a mod adds one. That is why an unnamed or unrecognised mod is a
rejected deployment rather than a warning — a runtime that starts without a mod
it needed looks perfectly healthy, and fails much later somewhere unrelated.

### Allocation-inducing runtime

A runtime whose inference API can be made to allocate resources by an
unauthenticated caller — Ollama, which pulls and loads a model on demand when
asked for one it does not have resident.

Contrast a **pinned runtime** (vLLM / sparkrun), which serves one already
resident model and answers an unknown name with an error. The distinction is
the whole basis of the network policy: an allocation-inducing runtime is
reachable only from the manager, so the gateway is the only sanctioned path to
it, while a pinned runtime is reachable directly on the local network.

The rule is about *what a caller can cause*, not about which runtime is
trusted. A future runtime that loads on demand inherits Ollama's restriction.

### Cache group

The set of nodes that see the same model weights, because they share one
storage location. Membership is a property of the storage, not of the nodes: a
group forms wherever nodes mount the same shared filesystem, and a node with no
shared storage is simply not in one rather than being a group of its own.

The group, not the node, is what a body of weights belongs to. Asking whether a
node has a model is the wrong question — the answer is a property of its group,
and the same weights fetched a second time for a sibling node would be the same
bytes written to the same place.

### Staging job

The act of bringing a model's weights into a cache group, tracked as something
with a lifetime of its own. It exists because weights are large enough that
acquiring them is an event in its own right — hours long, worth watching, and
worth refusing — rather than a step that can hide inside something else.

A staging job belongs to a cache group and names one model. It is deliberately
*not* part of a deployment: the same weights outlive any particular deployment,
may be wanted before anyone has decided what to run, and their absence is the
user's decision to resolve rather than an error to report. A deployment may wait
on a staging job, but it never owns one.

The distinction that gives the concept its point is between weights that are
*absent* and weights that are *arriving*. Without it, a runner that cannot find
weights can only fail, and a user who wants them can only be told no.
