# Missing Screenshots — capture checklist

Drop captured images into **`docs/screenshots/`** using the exact filenames below
(lowercase, kebab-case, `.png`). The README + docs refresh references these paths;
once the images are committed, the links resolve and render on GitHub.

Check each off as you capture it. **Hero** shots anchor the README's Screenshots
section and feature tour; **Optional** shots enrich the feature tour and the
self-hosting guide.

## Hero (README)

- [ ] `docs/screenshots/overview.png` — **Overview** page: cluster summary + live GPU sparklines + node cards
- [ ] `docs/screenshots/deployments.png` — **Deployments** list including a multi-node **cluster** deployment (status + cluster-node visualization)
- [ ] `docs/screenshots/deployment-logs.png` — **Deployment detail**: streaming vLLM startup log viewer
- [ ] `docs/screenshots/finetune-loss-curve.png` — **Fine-tune job detail**: live loss curve (train + eval overlay) + phase progress
- [ ] `docs/screenshots/benchmarks.png` — **Benchmarks**: results list / leaderboard

## Optional (feature tour / self-hosting guide)

- [ ] `docs/screenshots/benchmarks-compare.png` — **Benchmarks compare**: side-by-side run comparison
- [ ] `docs/screenshots/nodes.png` — **Nodes**: arch badge, GPU model, VRAM, provision health checks
- [ ] `docs/screenshots/finetune-create.png` — **Fine-tune create**: job creation form (recipe / node / dataset / hyperparameters)
- [ ] `docs/screenshots/datasets.png` — **Datasets**: dataset browser + preview rows
- [ ] `docs/screenshots/settings.png` — **Settings**: join-token management + agent bundle version / install command

## Not needed

Models and Load Balancer pages are intentionally **excluded** — their dashboard UIs
are still placeholders, so screenshots would misrepresent the project. (The
load-balancer *server API + inference proxy* are complete; only the UI is pending.)

---

_Source of truth: `docs/superpowers/specs/2026-06-11-docs-refresh-design.md` → Screenshot Manifest._
_Delete this file once all images are in place and the README links are verified._
