# DGX Manager — an open-source control plane for your DGX Spark cluster (looking for testers & feedback)

Hi all 👋

Over the past few months I've been building **DGX Manager**, a self-hosted control plane for running a DGX Spark cluster, and I'd love to get it in front of people who actually have this hardware. It's open source and I'm hoping to find a few folks willing to kick the tyres on their own nodes and tell me where it breaks.

## What it is

A single web dashboard to provision your nodes, deploy and load-balance inference, fine-tune models, and benchmark them — with real-time GPU telemetry and zero cloud dependencies. Everything runs on your own hardware. Under the hood it's a TypeScript monorepo: a dashboard that talks to a server over WebSocket, and a lightweight agent on each node that runs the runtimes and reports metrics.

## What it does today (all of this is working end-to-end)

- **Real-time GPU telemetry** — utilization, VRAM, temperature, network/RDMA across every node at 5-second resolution
- **One-click model deployment** — vLLM (recipe-driven) and Ollama
- **Multi-node inference clusters** — tensor/pipeline parallelism over Ray; I've served models up to Nemotron-3-Ultra 550B-A55B (NVFP4) across 4 nodes
- **End-to-end fine-tuning** — LoRA via DeepSpeed ZeRO-2/3, TRL+PEFT, or Unsloth; multi-node training; resume-from-checkpoint; merge → deploy in one loop
- **Live training observability** — phase-aware progress and a live loss curve streamed to the dashboard
- **Benchmarking** — llama-benchy presets with a compare view
- **Zero-touch onboarding** — single-use join tokens + a self-contained install script, with automatic agent updates
- **Heterogeneous hardware** — arm64 (DGX Spark / GB10) and amd64 nodes side by side

## Current state

Node management, deployment/inference, fine-tuning, agent bootstrap, and benchmarking are all functional. There's still plenty on the list — a Load Balancer UI and Models registry UI (the server APIs exist, the front-ends don't yet), auth/RBAC, and multi-cluster support are still to come.

## Where I could really use help

I'm pretty sure the **onboarding on fresh machines isn't as seamless as it should be** yet — NVIDIA drivers have to be in place, there are a few manager-host prerequisites, and I've only tested the install flow on my own boxes. That's exactly the kind of thing that's hard to get right without other people's hardware and setups.

So, two asks:

1. **Feedback** — tell me what's confusing, what's missing, what you'd want it to do.
2. **Run it on your hardware** — if you've got a DGX Spark (or even a single GB10 / amd64 GPU node) and a bit of patience for rough edges, I'd love a second set of eyes on the setup flow. I'll happily help you get it running and fix whatever trips you up.

Repo (setup guide in the README + `docs/SELF-HOSTING.md`): **https://github.com/kreuzhofer/dgx-manager**

Mostly I want to gauge interest and find out if this is useful to anyone besides me. Any feedback — even "this is the wrong approach" — is genuinely welcome. Thanks for reading! 🙏
