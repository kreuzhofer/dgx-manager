# 4. The manager measures nodes, never deployments

Date: 2026-09-24

## Status

Accepted.

## Context

Before a model is launched, the manager refuses the deploy if the node cannot
hold it. The check needs two numbers: how much memory is in use on the node,
and how much of that the pending action is about to free.

The first number is measured. An agent reports it every few seconds, and on
GB10 — where `nvidia-smi` reports `[N/A]` for both total and used, because the
GPU and the CPU share one pool — it is assembled by summing the memory of every
process holding a CUDA context.

The second number has no measurement behind it, and the product has twice now
tried to derive one anyway.

The first attempt (#1) noticed that a restart is refused for memory it is about
to release, and fixed it by defining that memory as *the node reading minus
everything attributable to other deployments*. The reasoning was sound as far as
it went: it avoided depending on whether a deployment's recorded memory is
stored per-node or summed across a cluster, which nothing in the schema settles.

The unconsidered case was that a node contains things the manager has no row
for. A fine-tune job holds tens of gigabytes and appears in no deployment query.
A container survives the offboarding that deleted its deployment row. Someone
starts a benchmark by hand. All of it lands in "not attributable to another
deployment" and is therefore credited to the restarting deployment as memory it
is about to hand back. With nothing else deployed on the node, *the entire node
reading* is credited away, the node computes as empty however full it is, and
the restart is admitted unconditionally. On a unified pool the admitted process
then contends with whatever was already there and something is OOM-killed (#92).

The second attempt was the fix first proposed for #118: bound the credit by the
deployment's own recorded memory. That field cannot bear the weight. For vLLM
and dgxrun the agent writes *the whole node's reading* into it — the same number
on every deployment sharing a node — while for Ollama it writes the model's own
size. The column's name says "this deployment's memory"; its contents say
"a sample of some node at some moment", and only for one of three runtimes does
it mean what it claims. Bounding the node reading by a copy of the node reading
is not a bound.

Underneath both attempts is one question nobody had answered: **is a
deployment's memory footprint something this system can know?**

## Decision 1 — a per-deployment memory figure is a claim, never a measurement

The manager measures nodes. It does not measure deployments, and no number it
holds about a deployment's memory is evidence of anything.

Per-process attribution is not technically out of reach — the per-process query
that already supplies the GB10 node total also reports a PID, and a PID can be
mapped to a container and thence to a deployment. We are deliberately not
building it, for a reason that will not go away with better engineering: a
CUDA-context census cannot see the memory that most often kills a deploy here.
The 126 GiB download that killed a serving replica in #92 held page cache, not a
CUDA context. On a unified pool, host pressure *is* GPU pressure, so any
attribution pass leaves a residue it cannot see by construction.

A design whose safety depends on attribution being complete therefore has #118's
bug with a smaller trigger, and a more confident tone. We would rather have a
system that knows it is working from claims.

## Decision 2 — the claim a deployment gets to make is its authorised share

When a deployment is created it is permitted to request a share of its node.
That share — the requested utilisation against the node's total — is the only
figure about a deployment's memory that means something, because it is a
*ceiling granted at admission time* rather than an observation after the fact.

A deployment may hold less than its share and frequently does. It may never
legitimately hold more. That asymmetry is what makes it safe to reason with: it
is an honest ceiling even when it is a loose one.

It is derived per node at the moment of the check rather than read from a stored
column, which also disposes of the cluster-attribution worry that sent #1 to the
node-metric trick. A share is a fraction of *a node*; there is nothing to
apportion and nothing to sum.

## Decision 3 — a restart is credited only up to its authorised share

Reclaim — the memory a restart releases before it needs it again — is bounded by
the restarting deployment's authorised share, and by nothing else. Memory the
manager cannot explain counts *against* the restart rather than for it.

The subtraction of other deployments' recorded memory is removed rather than
retained as a second bound. It adds no safety once the share caps the credit,
and it carries a live defect: because every vLLM deployment on a node records
that node's whole reading, two deployments over-count each other, the credit
clamps to zero, and the restart is refused for memory that *is* its own. That is
#1's symptom arriving from the opposite direction, and it was latent in the code
the whole time. After this decision, admission does not read that column at all.

The reclaim uses the share the deployment was created with; a restart that asks
for a *larger* share is sized by the new number and credited by the old one.
They are different quantities and the code names them separately.

## Decision 4 — the residual gap is accepted, and named

A deployment authorised for more than it takes still over-credits itself. An
image recipe that declares no utilisation at all falls back to the default and
claims most of a node for a workload taking a third of it.

We accept this. The gap is bounded by a number with a clear meaning — a
deployment can never credit itself more than a fresh deploy of the same recipe
would be permitted to request — where the behaviour it replaces was unbounded.
Closing it requires the measurement Decision 1 rejects.

## Decision 5 — a refusal that cannot name the holder has failed at half its job

Being told a node is full is not actionable. Being told which deployment or
training run fills it is.

Fine-tune jobs therefore appear in the list of holders a refusal reports, across
every phase in which they hold memory — training, merging, and quantizing are
three separate columns on the job and all three occupy a node. They contribute
nothing to the arithmetic; deploys were already *correct* about training memory,
since an unmodified node reading includes it. They were merely unable to say so,
which turned every refusal into an archaeology session (#101).

There is no override flag. A user who believes the check is wrong lowers the
requested share on the restart and is admitted on the merits, rather than by
disabling the check — which on a unified pool would deliver precisely the
OOM-kill the check exists to prevent.

## Consequences

Restarts can now be refused. Until now they effectively always succeeded, so the
first refusal on a node that is also training will look like a regression and is
not one; the refusal names the training job, and lowering the requested share is
the way through.

The field holding a deployment's recorded memory is now display-only. It remains
misleading — it is a node sample wearing a per-deployment name — and renaming it
is a separate cleanup. Nothing may reintroduce a dependency on it in an
admission path.

New deployments persist the share they were authorised for, so the claim stops
being implicit. Deployments predating this fall back to the recipe default; for
sparkrun recipes that default is read from a catalog the tool refreshes on its
own schedule, so an old deployment's recovered share can drift from the one it
launched with. The drift is bounded by the recipe and was not worth a migration.

Fine-tune jobs remain unchecked *on the way in*: starting a training run
performs no admission check at all. This ADR makes training visible to
deployment admission; it does not give training an admission check of its own.
That gap is real and separate.

## Related

- #118 — the restart that credited itself the whole node; the issue this
  decision resolves
- #1 — the first attempt, and the comment whose sound reasoning concealed the
  unconsidered case
- #92 — the download with no CUDA context that killed a serving replica;
  the concrete reason Decision 1 rejects attribution
- #101 — refusals that cannot name what is holding a resource; Decision 5 closes
  one instance of that complaint without generalising it
- `CONTEXT.md` § Node memory — the vocabulary this ADR fixes: admission, node
  reading, authorised share, unattributed memory, reclaim
