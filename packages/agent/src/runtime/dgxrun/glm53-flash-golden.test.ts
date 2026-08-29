import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import { buildDgxrunDockerArgs, tokenizeCommand, type DgxrunRecipe } from "./dgxrun-args.js";

/**
 * Equivalence gate for the GLM-5.3-Flash port.
 *
 * @dgxrun/glm-5.3-flash-nvfp4-2x claims to be a faithful port of
 * barrydeen/glm53-flash-dgx-spark @ c267f9f — the launcher that validated this
 * model on 2x GB10. The reference below is that repo's
 * `scripts/launch-glm53-vllm-dflash2-tp2.sh`, rank 0.
 *
 * The point, as with the DeepSeek gate, is that divergence must be *stated*.
 * This model is unusually punishing about silent divergence: drop
 * `--moe-backend marlin` and vLLM auto-selects an NVFP4 MoE backend that emits
 * repeated-token garbage from the first token with no error, passing /health the
 * whole time. A config that "looks fine" is exactly the failure mode here, so
 * every difference from the reference has to appear in an allowlist below.
 */

const RECIPE = join(process.cwd(), "recipes/dgxrun/glm-5.3-flash-nvfp4-2x.yaml");

/** Upstream's serve argv for rank 0, verbatim apart from the model path. */
const UPSTREAM_SERVE = `vllm serve LibertAIDAI/GLM-5.3-Flash-NVFP4 \
--served-model-name glm-5.3-flash --host 0.0.0.0 --port 8000 --trust-remote-code \
--tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 262144 \
--max-num-seqs 6 --block-size 2304 --moe-backend marlin --kv-cache-dtype fp8_e4m3 \
--enforce-eager --tool-call-parser glm47 --enable-auto-tool-choice \
--reasoning-parser glm45 --distributed-executor-backend mp \
--nnodes 2 --node-rank 0 --master-addr 192.168.44.37 --master-port 25000`;

/** Env upstream set, minus its own fleet's NIC names (see DELIBERATE_ENV_*). */
const UPSTREAM_ENV: Record<string, string> = {
  HF_HOME: "/cache/huggingface",
  HF_HUB_OFFLINE: "1",
  TRANSFORMERS_OFFLINE: "1",
  VLLM_ENGINE_READY_TIMEOUT_S: "3600",
  PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True",
  TORCH_CUDA_ARCH_LIST: "12.1a",
  FLASHINFER_CUDA_ARCH_LIST: "12.1a",
  FLASHINFER_DISABLE_VERSION_CHECK: "1",
  NCCL_NET: "IB",
  NCCL_IB_DISABLE: "0",
  NCCL_NVLS_ENABLE: "0",
  NCCL_CUMEM_ENABLE: "0",
  NCCL_IGNORE_CPU_AFFINITY: "1",
  NCCL_DEBUG: "WARN",
  TORCH_NCCL_ASYNC_ERROR_HANDLING: "1",
};

/** Flags dgxrun adds that upstream's launcher did not, each with its reason. */
const DELIBERATE_EXTRA_FLAGS: Record<string, string> = {};

/**
 * Flags upstream passed that we deliberately do not. Each entry is a decision
 * recorded in the recipe, not an oversight — deleting one from here without
 * changing the recipe fails the test.
 */
const DELIBERATE_FLAG_OMISSIONS: Record<string, string> = {
  "--speculative-config":
    "no spec decode on first bring-up: native MTP is unvalidated at TP2 and the " +
    "DFlash2 draft is CC-BY-NC-ND against an MIT target. See the recipe.",
};

/** Env keys we deliberately do not carry over from upstream's launcher. */
const DELIBERATE_ENV_OMISSIONS: Record<string, string> = {
  VLLM_HOST_IP: "dgxrun rendezvouses via --master-addr; six working recipes omit it",
  TP_SOCKET_IFNAME: "unused by the mp executor; every working dgxrun recipe omits it",
  MN_IF_NAME: "launcher-internal, consumed by upstream's own script",
};

/** Env we deliberately set to a different value than upstream's launcher. */
const DELIBERATE_ENV_DIFFERENCES: Record<string, string> = {
  NCCL_IB_HCA: "upstream names its own fleet's HCA; dgxrun injects ours",
  NCCL_SOCKET_IFNAME: "same — upstream's NIC names are not ours",
  GLOO_SOCKET_IFNAME: "same",
  NCCL_CROSS_NIC: "our fleet default is 1 and is proven at TP4; upstream's 2-node link uses 0",
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

const dockerArgs = buildDgxrunDockerArgs(recipe, {
  containerName: "dgxrun_test",
  weightsDir: "/mnt/tank/models",
  rank: 0,
  nnodes: 2,
  masterAddr: "192.168.44.37",
  masterPort: 25000,
});

// No mods on this recipe, so the serve argv is the container command directly —
// everything after the image reference.
const imageIdx = dockerArgs.lastIndexOf(recipe.container);
const ourServe = dockerArgs.slice(imageIdx + 1);

describe("@dgxrun/glm-5.3-flash-nvfp4-2x vs the validated 2x-Spark launcher", () => {
  const ours = flagMap(ourServe);
  const theirs = flagMap(tokenizeCommand(UPSTREAM_SERVE));

  it("passes every flag upstream passed, with the same value", () => {
    const wrong: string[] = [];
    for (const [flag, value] of theirs) {
      if (flag in DELIBERATE_FLAG_OMISSIONS) continue;
      if (!ours.has(flag)) wrong.push(`missing ${flag}`);
      else if (ours.get(flag) !== value) wrong.push(`${flag}: ours=${ours.get(flag)} theirs=${value}`);
    }
    expect(wrong).toEqual([]);
  });

  it("adds no flag beyond the deliberate ones", () => {
    const extra = [...ours.keys()].filter((f) => !theirs.has(f) && !(f in DELIBERATE_EXTRA_FLAGS));
    expect(extra).toEqual([]);
  });

  it("omits only the upstream flags we chose to drop", () => {
    for (const f of Object.keys(DELIBERATE_FLAG_OMISSIONS)) expect(ours.has(f)).toBe(false);
  });

  /**
   * The single highest-consequence flag in the recipe. Auto-selection picks
   * FLASHINFER_CUTLASS for the NVFP4 MoE on GB10, which produces degenerate
   * output with zero errors while /health keeps passing — so its absence is
   * invisible at runtime and must be caught here instead.
   */
  it("pins the NVFP4 MoE backend to marlin", () => {
    expect(ours.get("--moe-backend")).toBe("marlin");
  });

  /** fp8 KV only works because overlay layer v8 caps EFF_CTA_TILE_KV for GB10. */
  it("keeps fp8 KV and eager execution, which the overlay is built around", () => {
    expect(ours.get("--kv-cache-dtype")).toBe("fp8_e4m3");
    expect(ours.has("--enforce-eager")).toBe(true);
  });

  /**
   * block_size must be a multiple of BOTH 256 (index pool alignment) and 128
   * (MLA alignment). 2304 = 9 x 256 satisfies both; a "rounder" value like 2048
   * or 4096 silently breaks the index pool.
   */
  it("uses a block size legal for both the index pool and MLA", () => {
    const block = Number(ours.get("--block-size"));
    expect(block % 256).toBe(0);
    expect(block % 128).toBe(0);
  });

  it("sets every env var upstream set, with the same value", () => {
    const env = envMap(dockerArgs);
    const wrong: string[] = [];
    for (const [k, v] of Object.entries(UPSTREAM_ENV)) {
      if (k in DELIBERATE_ENV_DIFFERENCES) { expect(env.has(k)).toBe(true); continue; }
      if (env.get(k) !== v) wrong.push(`${k}: ours=${env.get(k)} theirs=${v}`);
    }
    expect(wrong).toEqual([]);
  });

  it("omits only the launcher-internal env we chose to drop", () => {
    const env = envMap(dockerArgs);
    for (const k of Object.keys(DELIBERATE_ENV_OMISSIONS)) expect(env.has(k)).toBe(false);
  });

  /**
   * This model needs no mod: the whole sm_121 patch stack is baked into the
   * image, because three of its layers are pip pins that cannot be applied at
   * serve time. A mod appearing here means someone split the stack in two.
   */
  it("declares no mods — the patch stack lives in the image", () => {
    expect(recipe.mods).toBeUndefined();
    expect(recipe.container).toBe("vllm-glm53-flash-sm121:probe");
  });

  /** Guards against someone "adapting" a GLM-5.2 recipe onto glm5_next. */
  it("carries no GLM-5.2 machinery", () => {
    const joined = ourServe.join(" ");
    for (const token of [
      "B12X_MLA_SPARSE",
      "index_topk_pattern",
      "--decode-context-parallel-size",
      "instanttensor",
    ]) {
      expect(joined).not.toContain(token);
    }
  });
});
