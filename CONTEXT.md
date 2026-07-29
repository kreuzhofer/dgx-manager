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
whatever name the deployment's runtime itself answers to — established when the
deployment starts serving, and thereafter a property of the deployment.

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
