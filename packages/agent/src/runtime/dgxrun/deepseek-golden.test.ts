import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import {
  buildDgxrunDockerArgs,
  tokenizeCommand,
  type DgxrunRecipe,
} from "./dgxrun-args.js";

/**
 * Equivalence gate for the DeepSeek V4 Flash port.
 *
 * @dgxrun/deepseek-v4-flash-0731-2x claims to be a faithful port of the official
 * sparkrun recipe. The reference below is not hand-written: it is the launch
 * sparkrun actually performed on dgx-spark-01 on 2026-08-08 — the rendered
 * `/tmp/sparkrun_serve.sh` and the container's `Config.Env`, captured from the
 * running deployment that produced our GPQA-Diamond number.
 *
 * The point is that divergence must be *stated*. A previous attempt at this port
 * silently carried GLM-5.2's env block and dropped thirteen B12X variables; it
 * failed deep inside DeepGEMM with an assertion that named none of them. This
 * test would have caught that in milliseconds, so every difference from the
 * reference has to appear in one of the allowlists below or the test fails.
 */

const RECIPE = join(process.cwd(), "recipes/dgxrun/deepseek-v4-flash-0731-2x.yaml");

/** The serve command sparkrun rendered, verbatim. */
const SPARKRUN_SERVE = `vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
--host 0.0.0.0 --port 8000 --trust-remote-code --tensor-parallel-size 2 \
--kv-cache-dtype fp8 --block-size 256 --max-model-len auto --max-num-seqs 8 \
--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --enable-prefix-caching \
--tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
--reasoning-parser deepseek_v4 \
--reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}' \
--default-chat-template-kwargs.thinking=true \
--default-chat-template-kwargs.reasoning_effort=high \
--load-format instanttensor --moe-backend b12x --linear-backend b12x \
--attention-backend B12X_MLA_SPARSE --max-cudagraph-capture-size 64 \
--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
--speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","attention_backend":"B12X_MLA_SPARSE"}' \
--served-model-name deepseek-v4-flash \
--nnodes 2 --node-rank 0 --master-addr 192.168.44.36 --master-port 25000`;

/** Env sparkrun set for that container (exports + docker -e), minus image-baked vars. */
const SPARKRUN_ENV: Record<string, string> = {
  B12X_MLA_SM120_UNIFIED: "1", B12X_MOE_FORCE_A8: "1", CUTE_DSL_ARCH: "sm_121a",
  HF_HOME: "/cache/huggingface", HF_HUB_OFFLINE: "1", NCCL_CUMEM_ENABLE: "0",
  OMP_NUM_THREADS: "4", TRANSFORMERS_OFFLINE: "1", VLLM_MEMORY_PROFILE_INCLUDE_ATTN: "1",
  VLLM_USE_AOT_COMPILE: "1", VLLM_USE_B12X_FP8_GEMM: "1", VLLM_USE_B12X_MHC: "1",
  VLLM_USE_B12X_MOE: "1", VLLM_USE_B12X_SPARSE_INDEXER: "1", VLLM_USE_B12X_WO_PROJECTION: "1",
  VLLM_USE_BREAKABLE_CUDAGRAPH: "0", VLLM_USE_FLASHINFER_SAMPLER: "1",
  VLLM_USE_MEGA_AOT_ARTIFACT: "-1", VLLM_USE_V2_MODEL_RUNNER: "1",
  NCCL_NET: "IB", NCCL_IB_DISABLE: "0", NCCL_IB_HCA: "rocep1s0f0,roceP2p1s0f0",
  NCCL_SOCKET_IFNAME: "enP7s7,enp1s0f0np0,enP2p1s0f0np0", NCCL_IB_GID_INDEX: "3",
  NCCL_CROSS_NIC: "1", NCCL_IGNORE_CPU_AFFINITY: "1", GLOO_SOCKET_IFNAME: "enP7s7",
};

/** Flags dgxrun adds that sparkrun did not set, each with its reason. */
const DELIBERATE_EXTRA_FLAGS: Record<string, string> = {
  "--distributed-executor-backend": "dgxrun forces mp; ray is broken on this build",
};

/** Env keys we deliberately do not carry over from sparkrun's launch. */
const DELIBERATE_ENV_OMISSIONS: Record<string, string> = {
  UCX_NET_DEVICES: "unused by the mp executor; six working dgxrun recipes omit it",
  MN_IF_NAME: "sparkrun-internal, consumed by its own launcher",
  NODE_IP: "sparkrun-internal, consumed by its own launcher",
  TP_SOCKET_IFNAME: "unused; six working dgxrun recipes omit it",
  OMPI_MCA_btl_tcp_if_include: "OpenMPI is not in the mp execution path",
};

/** Env values we deliberately set differently from sparkrun's launch. */
const DELIBERATE_ENV_DIFFERENCES: Record<string, string> = {
  GLOO_SOCKET_IFNAME: "our six working recipes use the mgmt NIC for the gloo store",
};

/** Split an argv into flag→value, treating `--x=y` and `--x y` alike. */
function flagMap(argv: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (!t.startsWith("-")) continue;
    const eq = t.indexOf("=");
    if (eq > 0) { out.set(t.slice(0, eq), t.slice(eq + 1)); continue; }
    const next = argv[i + 1];
    if (next != null && !next.startsWith("--")) { out.set(t, next); i++; }
    else out.set(t, "");
  }
  return out;
}

function envMap(dockerArgs: string[]): Map<string, string> {
  const out = new Map<string, string>();
  for (let i = 0; i < dockerArgs.length; i++) {
    if (dockerArgs[i] !== "-e") continue;
    const kv = dockerArgs[i + 1];
    const eq = kv.indexOf("=");
    out.set(kv.slice(0, eq), kv.slice(eq + 1)); // later -e wins, as docker does
  }
  return out;
}

const recipe = parse(readFileSync(RECIPE, "utf-8")) as DgxrunRecipe;

// Launch options chosen to match the captured sparkrun deployment exactly, so
// rendezvous flags compare equal rather than showing as spurious differences.
const dockerArgs = buildDgxrunDockerArgs(recipe, {
  containerName: "dgxrun_test",
  weightsDir: "/mnt/tank/models",
  rank: 0,
  nnodes: 2,
  masterAddr: "192.168.44.36",
  masterPort: 25000,
});

const script = dockerArgs[dockerArgs.length - 1];
const ourServe = tokenizeCommand(script.slice(script.indexOf(" exec ") + " exec ".length));

describe("@dgxrun/deepseek-v4-flash-0731-2x vs the sparkrun launch", () => {
  const ours = flagMap(ourServe);
  const theirs = flagMap(tokenizeCommand(SPARKRUN_SERVE));

  it("passes every flag sparkrun passed, with the same value", () => {
    const wrong: string[] = [];
    for (const [flag, value] of theirs) {
      if (!ours.has(flag)) wrong.push(`missing ${flag}`);
      else if (ours.get(flag) !== value) wrong.push(`${flag}: ours=${ours.get(flag)} theirs=${value}`);
    }
    expect(wrong).toEqual([]);
  });

  it("adds no flag beyond the deliberate ones", () => {
    const extra = [...ours.keys()].filter((f) => !theirs.has(f) && !(f in DELIBERATE_EXTRA_FLAGS));
    expect(extra).toEqual([]);
  });

  it("keeps the JSON arguments intact as single tokens", () => {
    expect(ours.get("--speculative-config")).toBe(
      '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic",' +
      '"attention_backend":"B12X_MLA_SPARSE"}',
    );
    expect(ours.get("--compilation-config")).toBe(
      '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}',
    );
  });

  it("substitutes the placeholder nested inside the speculative JSON", () => {
    expect(ourServe.join(" ")).not.toContain("{num_speculative_tokens}");
    expect(ourServe.join(" ")).not.toContain("{{");
  });

  it("sets every env var sparkrun set, with the same value", () => {
    const env = envMap(dockerArgs);
    const wrong: string[] = [];
    for (const [k, v] of Object.entries(SPARKRUN_ENV)) {
      if (k in DELIBERATE_ENV_DIFFERENCES) { expect(env.has(k)).toBe(true); continue; }
      if (env.get(k) !== v) wrong.push(`${k}: ours=${env.get(k)} theirs=${v}`);
    }
    expect(wrong).toEqual([]);
  });

  it("omits only the sparkrun-internal env we chose to drop", () => {
    const env = envMap(dockerArgs);
    for (const k of Object.keys(DELIBERATE_ENV_OMISSIONS)) expect(env.has(k)).toBe(false);
  });

  it("applies the instanttensor loader mod", () => {
    expect(recipe.mods).toEqual(["instanttensor-hybrid-draft-loader"]);
    expect(script).toMatch(/^bash \/mods\/instanttensor-hybrid-draft-loader\/run\.sh && exec /);
  });
});
