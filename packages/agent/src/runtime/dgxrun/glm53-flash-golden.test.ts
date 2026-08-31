import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { parse } from "yaml";
import { buildDgxrunDockerArgs, tokenizeCommand, type DgxrunRecipe } from "./dgxrun-args.js";

/**
 * Equivalence gate for the GLM-5.3-Flash port.
 *
 * @dgxrun/glm-5.3-flash-libertai-nvfp4-2x claims to be a faithful port of
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

const RECIPE = join(process.cwd(), "recipes/dgxrun/glm-5.3-flash-libertai-nvfp4-2x.yaml");

/**
 * Upstream's serve argv for rank 0, verbatim apart from the model path.
 *
 * Transcribed from `scripts/launch-glm53-vllm-dflash2-tp2.sh` in
 * barrydeen/glm53-flash-dgx-spark @ c267f9f. Only that repo's `docker/` tree is
 * vendored here (scripts/glm53-flash-overlay/), so the launcher itself is NOT
 * in this repo — check it upstream before trusting this line.
 */
const UPSTREAM_SERVE = `vllm serve LibertAIDAI/GLM-5.3-Flash-NVFP4 \
--served-model-name glm-5.3-flash --host 0.0.0.0 --port 8000 --trust-remote-code \
--tensor-parallel-size 2 --gpu-memory-utilization 0.85 --max-model-len 262144 \
--max-num-seqs 6 --block-size 2304 --moe-backend marlin --kv-cache-dtype fp8_e4m3 \
--enforce-eager --tool-call-parser glm47 --enable-auto-tool-choice \
--reasoning-parser glm45 --distributed-executor-backend mp \
--speculative-config '{"method":"dflash","model":"/models/glm-5.3-flash-dflash2","num_speculative_tokens":7}' \
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
const DELIBERATE_EXTRA_FLAGS: Record<string, string> = {
  "--default-chat-template-kwargs":
    "pins reasoning_effort — upstream leaves it unset and therefore at the " +
    "template's `max` fallback, which nobody chose. See the recipe.",
};

/**
 * Flags we deliberately set to a DIFFERENT value than upstream's launcher.
 * Each entry states why, because a silent value drift here is exactly how a
 * validated config rots into an unvalidated one.
 */
const DELIBERATE_FLAG_DIFFERENCES: Record<string, string> = {
  "--gpu-memory-utilization":
    "0.87, not upstream's 0.85 — issue #24. At 0.85 roughly 5 GiB of the device " +
    "went unused; 0.89 cleared vLLM's startup guard and was then OOM-killed " +
    "during multimodal warmup. The recipe carries the measurements.",
  "--max-model-len":
    "327680 (320K), not upstream's 262144 — issue #24. Validated by needle probe " +
    "at 294,828 prompt tokens; 491520 was tried and could not serve its own " +
    "window. The recipe carries the measurements.",
  "--speculative-config":
    "native MTP, not upstream's DFlash2 — issue #26. The checkpoint carries the " +
    "MTP head in layer 45 (eh_proj/enorm/hnorm/shared_head), so drafting needs " +
    "no extra weights and no extra licence. DFlash2 would need a ninth overlay " +
    "layer porting an unmerged vLLM PR plus a CC-BY-NC-ND draft against an MIT " +
    "target.",
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

describe("@dgxrun/glm-5.3-flash-libertai-nvfp4-2x vs the validated 2x-Spark launcher", () => {
  const ours = flagMap(ourServe);
  const theirs = flagMap(tokenizeCommand(UPSTREAM_SERVE));

  it("passes every flag upstream passed, with the same value", () => {
    const wrong: string[] = [];
    for (const [flag, value] of theirs) {
      if (!ours.has(flag)) wrong.push(`missing ${flag}`);
      else if (flag in DELIBERATE_FLAG_DIFFERENCES) {
        // Stated divergence — the exact values are pinned by their own test, so
        // this only records that the flag is still passed and still differs.
        // An allowlist entry that has drifted back to upstream's value is stale,
        // which is its own kind of rot.
        expect(ours.get(flag), `${flag} is allowlisted as different but matches upstream`)
          .not.toBe(value);
      } else if (ours.get(flag) !== value) {
        wrong.push(`${flag}: ours=${ours.get(flag)} theirs=${value}`);
      }
    }
    expect(wrong).toEqual([]);
  });

  it("adds no flag beyond the deliberate ones", () => {
    const extra = [...ours.keys()].filter((f) => !theirs.has(f) && !(f in DELIBERATE_EXTRA_FLAGS));
    expect(extra).toEqual([]);
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

  /**
   * gmu and max_model_len are EMPIRICALLY VALIDATED values, not derived ones.
   * They are pinned here so they cannot drift silently; the measurements and
   * the four separate walls behind them live in ONE place — the recipe's own
   * comments — rather than being restated (and then diverging) here.
   *
   * Do not change either without a fresh long-prefill needle probe. This is not
   * ceremony: at max_model_len 491520 the deploy booted, profiled, allocated a
   * 927,955-token pool, passed /health and answered short prompts correctly —
   * and still could not serve its own window, dying on the first long prefill
   * with a GB10 indexer top-k kernel limit. Nothing short of a real long prompt
   * distinguishes a working window from a broken one. See issue #24.
   */
  /**
   * reasoning_effort must be PINNED, and to a value the template accepts.
   *
   * This checkpoint's chat_template.jinja opens with:
   *
   *   {%- set effective_reasoning_effort = reasoning_effort
   *         if reasoning_effort is defined and reasoning_effort in ['low','high']
   *         else 'max' -%}
   *
   * Two consequences, both traps. The default is `max` — the most expensive
   * setting, reachable only as a fallback and never chosen. And the accepted
   * set is ONLY {low, high}: unlike Qwen, whose template raise_exception()s on
   * an unknown value, this one SILENTLY falls back to `max`. So "medium" — the
   * value both qwen3.8 recipes pin — would look applied here and give you max.
   *
   * That is why this asserts membership in the accepted set rather than just
   * "is set": a typo or a copied-across "medium" is invisible at runtime.
   */
  it("pins reasoning_effort to a value this template actually accepts", () => {
    const raw = ours.get("--default-chat-template-kwargs");
    expect(raw, "--default-chat-template-kwargs must be set").toBeDefined();
    const kw = JSON.parse(raw as string);
    expect(["low", "high"]).toContain(kw.reasoning_effort);
  });

  it("pins the empirically validated memory and window settings", () => {
    expect(ours.get("--gpu-memory-utilization")).toBe("0.87");
    expect(ours.get("--max-model-len")).toBe("327680");
  });

  /**
   * 4 is a measured setting, not a default: it halves the KV pool. Pinned for
   * the same reason as the two above — see the recipe for the numbers.
   */
  it("drafts with the checkpoint's own MTP head, not an external draft model", () => {
    expect(ours.has("--speculative-config")).toBe(true);
    const spec = JSON.parse(ours.get("--speculative-config") as string);
    expect(spec.method).toBe("mtp");
    // No `model` key: an external drafter would mean extra weights, and the only
    // published option carries a licence mismatch. See the allowlist entry.
    expect(spec.model).toBeUndefined();
    expect(spec.num_speculative_tokens).toBe(4);
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
