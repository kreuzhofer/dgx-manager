# 1. An inference gateway, not a load balancer

Date: 2026-07-29

## Status

Accepted.

## Context

Models on the cluster were not reachable from the network in any consistent way.

Ollama deployments were not reachable at all. The agent installs a startup
firewall restricting `:11434` to the manager and loopback, because Ollama has no
auth and an unauthenticated caller once induced a 15 GB model load on a node
mid-deployment, killing a four-node cluster. The only workaround was an SSH
tunnel.

vLLM deployments were reachable, but only by knowing a node IP and a port that
changes when a deployment moves or has its port auto-bumped by a conflict — and
the name to send in the `model` field was not knowable from the manager, since
a recipe may set its own served-model-name.

A load balancer was supposed to solve this and never did. Its rules and
endpoints API was complete with zero rows, its inference proxy was written but
never mounted, and its page was a placeholder. Nothing routed.

Three decisions were taken whose reasoning would not be recoverable from the
resulting code. They are recorded here.

## Decision 1 — a gateway, not a load balancer

The system exposes one OpenAI-compatible address that fronts every running
deployment, routing by the **published name** in the request body. Balancing
across replicas is a behaviour it exhibits when a **pool** has more than one
member, not the reason it exists.

The load balancer's model — named rules, hand-attached endpoints, a configured
strategy — assumed the scarce thing was capacity across interchangeable
replicas. On this cluster it is not: a model spans four nodes as a single
deployment, and a second replica of anything is the exception. The scarce thing
is a stable address and a knowable name. Rules were therefore configuration
that had to be maintained to describe something the manager already knew.

Consequently: publication is automatic for any running deployment, pools form
implicitly when two deployments share a published name, and the rules and
endpoints tables, their API, and the unmounted proxy were deleted rather than
adapted.

**Rejected — path-based routing** (`/lb/<rule>/v1/...`): forces escaping of
names that legitimately contain colons and slashes, and cannot express several
models sharing one address, which is exactly how Ollama serves.

**Consequence accepted:** the manager sits in the inference data path. It is a
low-power host with a known failure mode (it does not start after a reboot until
its network mount is touched). This is tolerable because it was already the only
party permitted to reach Ollama, and because pinned-runtime traffic can still
address nodes directly — GPU inference never hard-depends on the gateway.

## Decision 2 — network policy follows what a caller can *cause*

Ollama stays restricted to the manager. vLLM stays directly reachable on the
local network. The asymmetry is deliberate and is not about which runtime is
trusted.

An **allocation-inducing runtime** can be made to allocate resources by an
unauthenticated caller: Ollama loads a model on demand when asked for one it has
resident but unloaded. A **pinned runtime** serves one already-resident model and
answers an unknown name with an error. The hazard is the induced allocation, so
the restriction follows the hazard, not the vendor. A future runtime that loads
on demand inherits Ollama's restriction.

The gateway does not weaken this, and this is the load-bearing part: it routes
**only by published name**, and a published name exists only where a deployment
record exists. A request for some other model resident on that node never
reaches Ollama — it is refused at the gateway. The cluster's model list is
likewise answered from the manager's own record of what is deployed, never by
asking a node what it holds, because an allocation-inducing runtime's own list
enumerates everything on disk.

The gateway therefore exposes a **strictly narrower** surface than the port it
fronts: the deployed subset, chosen deliberately, rather than everything present.

**Any change that makes the gateway forward an unrecognised name, or proxy a
runtime's own model list, reopens the incident that motivated the firewall.**

## Decision 3 — no authentication, deliberately

The gateway ships without authentication.

The manager has no authentication on any of its management routes — node
provisioning, deployment control, and token handling are all open on the local
network. A key on the inference surface alone would be theatre: anyone who could
reach it could equally stop a deployment through the management API.

The containment argument in Decision 2 is what makes this tolerable rather than
merely consistent — the gateway is narrower than the runtime it fronts, so
adding it does not widen the network's reach into the cluster beyond what a
deliberate deployment already granted.

Client-supplied authorization headers are accepted and discarded rather than
forwarded, so client credentials never reach a node.

**This is a decision, not an oversight.** Revisiting it means authenticating the
management API too; doing the gateway alone would buy very little.

## Consequences

- Two tables, an API, an unmounted proxy module, and a placeholder page were
  deleted. The rules/endpoints API was in the published OpenAPI spec, so its
  removal is a visible contract change despite having no consumers.
- Deployment records gain a published name, resolved once when the deployment
  starts serving, so routing never probes a node in the request path.
- The manager becomes the documented path for Ollama inference.
- A rule forbidding slashes in a display name outlived its original
  justification (path-segment collisions in the deleted router). The rule was
  kept and its rationale rewritten rather than being silently inherited.
