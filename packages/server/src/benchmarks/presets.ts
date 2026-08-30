export type LatencyMode = "api" | "generation" | "none";

export type BenchmarkConfig = {
  pp: number[];
  tg: number[];
  depth: number[];
  runs: number;
  concurrency: number[];
  latencyMode: LatencyMode;
  enablePrefixCaching: boolean;
  skipCoherence: boolean;
};

export type ToolEvalConfig = {
  short: boolean;          // --short (15 core scenarios) vs full 63
  hardmode: boolean;       // --hardmode (adds the hard scenario tier)
  contextPressure: number | null; // --context-pressure R (0-1); null = omit
  seed: number;            // --seed N, for reproducible runs
};

export type AccuracyConfig = {
  tasks: string[];
  primaryTask: string;
  primaryMetric: string;
  limit: number | null;
  numFewshot: number | null;
  maxGenToks: number;
  applyChatTemplate: boolean;
  reasoning: boolean;
  seed: number;
  /** lm-eval concurrent requests. Match the deployment's --max-num-seqs: 1 for the
   *  latency-tuned serving recipe, higher (e.g. 16) for the batched eval recipe.
   *  Optional/omitted => 1. */
  numConcurrent?: number;
  /** Prepended as a system message on every request (lm-eval's
   *  --system_instruction). Exists because the GPQA prompt never states an
   *  answer format: it ends "Let's think step by step: " and assumes the model
   *  will spontaneously write "The answer is (C)". A model that phrases it
   *  differently scores zero however correct it is. Optional/omitted => the flag
   *  is not passed at all, so existing presets and their baselines are
   *  unaffected. Setting it CHANGES WHAT THE NUMBER MEANS - a run with an
   *  instruction is not comparable with one without. */
  systemInstruction?: string;
  /** Per-request timeout in SECONDS, bounding the whole request including
   *  generation (lm-eval applies it as aiohttp ClientTimeout(total=...)).
   *  Optional/omitted => DEFAULT_TIMEOUT_S. Raise it when maxGenToks divided by
   *  the model's realistic per-request tokens/sec exceeds the default. */
  timeout?: number;
};

export type BenchmarkKind = "throughput" | "tool-eval" | "accuracy";

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig | AccuracyConfig;
};

type AccuracyBench = {
  idBase: string;
  label: string;
  task: string;
  primaryMetric: string;
  quickLimit: number;
  maxGenToks: number;
  blurb: string;
  /**
   * When set, also emit an `acc-<idBase>-full-longgen` preset with this
   * generation cap instead of {@link maxGenToks}.
   *
   * Exists because a reasoning model that thinks past max_gen_toks returns
   * EMPTY content; lm-eval substitutes LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER and
   * scores the item WRONG, silently, with no error in the results. Measured on
   * Qwen3.8-27B (2026-08-29): 33% of GPQA-Diamond items came back null at the
   * standard 4096 cap, which would have produced a badly depressed score that
   * looked like a real result.
   *
   * This is ADDITIVE rather than a raise of the shared cap on purpose: GLM-5.2
   * ran the standard preset with 0 errors and its published 69.2% is a
   * comparison baseline, so changing that preset's cap would silently
   * invalidate every number already measured with it.
   */
  longGenToks?: number;
};

// The v1 lineup: HF Open LLM Leaderboard v2 minus MuSR, all generative/CoT so
// they run over an OpenAI chat endpoint. Task ids are pinned against the
// LM_EVAL_VERSION set in orchestrator.ts — verify them when bumping that pin.
const ACCURACY_BENCHES: AccuracyBench[] = [
  { idBase: "ifeval", label: "IFEval", task: "ifeval", primaryMetric: "prompt_level_strict_acc", quickLimit: 100, maxGenToks: 2048, blurb: "Instruction-following adherence." },
  { idBase: "mmlu-pro", label: "MMLU-Pro (CoT)", task: "mmlu_pro", primaryMetric: "exact_match", quickLimit: 200, maxGenToks: 4096, blurb: "Knowledge/reasoning tail, chain-of-thought." },
  { idBase: "gpqa-diamond", label: "GPQA-Diamond (CoT)", task: "gpqa_diamond_cot_zeroshot", primaryMetric: "exact_match", quickLimit: 50, maxGenToks: 4096, longGenToks: 32768, blurb: "Hard graduate-level Q&A, chain-of-thought." },
  { idBase: "gsm8k", label: "GSM8K", task: "gsm8k_cot", primaryMetric: "exact_match", quickLimit: 200, maxGenToks: 2048, blurb: "Grade-school math word problems." },
  { idBase: "bbh", label: "BBH", task: "bbh_cot_zeroshot", primaryMetric: "exact_match", quickLimit: 40, maxGenToks: 4096, blurb: "Big-Bench-Hard reasoning suite, chain-of-thought." },
  { idBase: "math-hard", label: "MATH-hard", task: "leaderboard_math_hard", primaryMetric: "exact_match", quickLimit: 100, maxGenToks: 4096, blurb: "Competition-level MATH (level-5)." },
];

/**
 * Slowest per-request generation rate we plan for, in tokens/sec, used to derive a
 * long-generation preset's timeout from its own cap.
 *
 * Not a measurement of any one model — a floor. Qwen3.8-27B measures 10.75 tok/s
 * single-stream on one Spark and Muse Glimmer ~16.8 at concurrency 5, but presets run at
 * numConcurrent 8 where per-request throughput is a fraction of that. Lower this if a
 * slower endpoint starts timing out; the derivation then widens every longgen timeout.
 */
const SLOWEST_PER_REQUEST_TOKS_PER_SEC = 5;

function accuracyPresets(): BenchmarkPreset[] {
  const out: BenchmarkPreset[] = [];
  for (const b of ACCURACY_BENCHES) {
    const base: AccuracyConfig = {
      tasks: [b.task],
      primaryTask: b.task,
      primaryMetric: b.primaryMetric,
      limit: null,
      numFewshot: null,
      maxGenToks: b.maxGenToks,
      applyChatTemplate: true,
      reasoning: true,
      seed: 42,
    };
    out.push({
      id: `acc-${b.idBase}-quick`,
      label: `${b.label} — quick (${b.quickLimit})`,
      description: `${b.blurb} Sampled to ${b.quickLimit} items for a fast quality probe.`,
      kind: "accuracy",
      config: { ...base, limit: b.quickLimit },
    });
    out.push({
      id: `acc-${b.idBase}-full`,
      label: `${b.label} — full`,
      description: `${b.blurb} Complete dataset — can run for hours on a slow endpoint.`,
      kind: "accuracy",
      config: { ...base },
    });
    if (b.longGenToks) {
      out.push({
        id: `acc-${b.idBase}-full-longgen`,
        label: `${b.label} — full, long generation (${Math.round(b.longGenToks / 1024)}k)`,
        description:
          `${b.blurb} Complete dataset with a ${b.longGenToks}-token generation cap, for ` +
          `reasoning models that overrun the standard ${b.maxGenToks}. Use this when the run log ` +
          `shows "API returned null content" — those items score WRONG rather than erroring.`,
        kind: "accuracy",
        config: {
          ...base,
          maxGenToks: b.longGenToks,
          // DERIVED, never hardcoded: the timeout bounds the WHOLE request including
          // generation, so a cap the timeout cannot reach is a guaranteed failure rather
          // than a bigger budget. Deriving it from the cap is what stops the two drifting
          // apart — which is exactly how the 2026-08-30 GPQA run died at 137/198 after
          // 4h34m, needing >18 tok/s per request to fit 32768 tokens in the 1800s default.
          timeout: Math.ceil(b.longGenToks / SLOWEST_PER_REQUEST_TOKS_PER_SEC),
        },
      });
    }
  }
  return out;
}

export const BENCHMARK_PRESETS: BenchmarkPreset[] = [
  {
    id: "quick-smoke",
    label: "Quick smoke",
    description: "30-second sanity check: one short prompt, one generation.",
    kind: "throughput",
    config: {
      pp: [128],
      tg: [32],
      depth: [0],
      runs: 1,
      concurrency: [1],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "chat-short",
    label: "Chat (short)",
    description: "Typical chatbot turn: 512-token prompt, 128 generated.",
    kind: "throughput",
    config: {
      pp: [512],
      tg: [128],
      depth: [0],
      runs: 3,
      concurrency: [1, 4],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "chat-long",
    label: "Chat (long context)",
    description: "Long conversation: 2k prompt, 128 generated, swept across 0/4k context.",
    kind: "throughput",
    config: {
      pp: [2048],
      tg: [128],
      depth: [0, 4096],
      runs: 3,
      concurrency: [1, 4],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "code-32k",
    label: "Code (32k context)",
    description: "Repo-scale codegen: 8k prompt, 512 generated, swept up to 32k context.",
    kind: "throughput",
    config: {
      pp: [8192],
      tg: [512],
      depth: [0, 16384, 32000],
      runs: 2,
      concurrency: [1],
      latencyMode: "api",
      enablePrefixCaching: false,
      skipCoherence: true,
    },
  },
  {
    id: "throughput",
    label: "Throughput sweep",
    description: "Concurrency ramp: same prompt at 1/4/16/32/64 in-flight requests.",
    kind: "throughput",
    config: {
      pp: [512],
      tg: [128],
      depth: [0],
      runs: 2,
      concurrency: [1, 4, 16, 32, 64],
      latencyMode: "none",
      enablePrefixCaching: false,
      skipCoherence: false,
    },
  },
  {
    id: "tool-eval-quick",
    label: "Tool eval — quick (15)",
    description: "15 core tool-calling scenarios: a fast tool-use sanity check.",
    kind: "tool-eval",
    config: { short: true, hardmode: false, contextPressure: null, seed: 42 },
  },
  {
    id: "tool-eval-full",
    label: "Tool eval — full (63)",
    description: "Full 63-scenario tool-calling suite across all categories.",
    kind: "tool-eval",
    config: { short: false, hardmode: false, contextPressure: null, seed: 42 },
  },
  {
    id: "tool-eval-hardmode",
    label: "Tool eval — hard mode",
    description: "Full suite plus the harder scenario tier (Category P).",
    kind: "tool-eval",
    config: { short: false, hardmode: true, contextPressure: null, seed: 42 },
  },
  {
    id: "tool-eval-pressure",
    label: "Tool eval — context pressure",
    description: "Full suite with context filled to 75% to stress long-context tool use.",
    kind: "tool-eval",
    config: { short: false, hardmode: false, contextPressure: 0.75, seed: 42 },
  },
  ...accuracyPresets(),
];

export function listPresets(): BenchmarkPreset[] {
  return BENCHMARK_PRESETS;
}

export function getPreset(id: string): BenchmarkPreset | undefined {
  return BENCHMARK_PRESETS.find((p) => p.id === id);
}
