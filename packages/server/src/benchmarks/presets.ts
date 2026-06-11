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

export type BenchmarkKind = "throughput" | "tool-eval";

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  kind: BenchmarkKind;
  config: BenchmarkConfig | ToolEvalConfig;
};

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
];

export function listPresets(): BenchmarkPreset[] {
  return BENCHMARK_PRESETS;
}

export function getPreset(id: string): BenchmarkPreset | undefined {
  return BENCHMARK_PRESETS.find((p) => p.id === id);
}
