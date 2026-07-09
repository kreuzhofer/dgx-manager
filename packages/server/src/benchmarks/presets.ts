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
};

// The v1 lineup: HF Open LLM Leaderboard v2 minus MuSR, all generative/CoT so
// they run over an OpenAI chat endpoint. Task ids are pinned against the
// LM_EVAL_VERSION set in orchestrator.ts — verify them when bumping that pin.
const ACCURACY_BENCHES: AccuracyBench[] = [
  { idBase: "ifeval", label: "IFEval", task: "ifeval", primaryMetric: "prompt_level_strict_acc", quickLimit: 100, maxGenToks: 2048, blurb: "Instruction-following adherence." },
  { idBase: "mmlu-pro", label: "MMLU-Pro (CoT)", task: "mmlu_pro", primaryMetric: "exact_match", quickLimit: 200, maxGenToks: 4096, blurb: "Knowledge/reasoning tail, chain-of-thought." },
  { idBase: "gpqa-diamond", label: "GPQA-Diamond (CoT)", task: "gpqa_diamond_cot_zeroshot", primaryMetric: "exact_match", quickLimit: 50, maxGenToks: 4096, blurb: "Hard graduate-level Q&A, chain-of-thought." },
  { idBase: "gsm8k", label: "GSM8K", task: "gsm8k_cot", primaryMetric: "exact_match", quickLimit: 200, maxGenToks: 2048, blurb: "Grade-school math word problems." },
  { idBase: "bbh", label: "BBH", task: "bbh_cot_zeroshot", primaryMetric: "exact_match", quickLimit: 40, maxGenToks: 4096, blurb: "Big-Bench-Hard reasoning suite, chain-of-thought." },
  { idBase: "math-hard", label: "MATH-hard", task: "leaderboard_math_hard", primaryMetric: "exact_match", quickLimit: 100, maxGenToks: 4096, blurb: "Competition-level MATH (level-5)." },
];

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
