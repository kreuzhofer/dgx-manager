# Vendored dgxrun mods

A **mod** is a named change applied to a runtime container before it begins
serving — see the Mod entry in [`CONTEXT.md`](../CONTEXT.md). A dgxrun recipe
declares the mods it needs by name:

```yaml
mods:
  - instanttensor-hybrid-draft-loader
```

Each mod is a directory here containing a `run.sh`, which the agent executes
inside the container immediately before `exec`ing the serve command. The
directory is bind-mounted read-only at `/mods/<name>`; everything a mod writes
goes into the container's own Python tree.

## Why these are vendored rather than referenced

sparkrun resolves mods from registry clones under `~/.cache/sparkrun/registries/`,
which it refreshes on its own schedule. A deployment whose behaviour depends on
the contents of that cache is not reproducible: a routine `sparkrun registry
update` can change what a deploy patches, and a node that has never registered
the right registry simply fails. Copying the mod here pins it to an agent
version and puts it under review alongside the recipe that names it.

The cost is that upstream fixes do not reach us on their own. When re-pulling a
mod, record the new provenance below.

## Shipping and validation

The agent bundle carries `mods/` to `/opt/dgx-agent/mods/` on each node. The
manager validates that a declared mod name is a single path segment (it becomes
a bind-mount source); the agent checks that the directory actually exists and
**refuses the deploy** if it does not, because a runtime that starts without a
mod it needed looks healthy and fails much later somewhere unrelated.

## Provenance

| Mod | Upstream | Commit | Vendored |
|---|---|---|---|
| `instanttensor-hybrid-draft-loader` | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) `mods/` | `15b4f48` (2026-08-06) | 2026-08-08 |

Upstream is MIT licensed — Copyright (c) 2026 Eugene Rakhmatulin. The vendored
files are unmodified copies.
