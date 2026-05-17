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

export type BenchmarkPreset = {
  id: string;
  label: string;
  description: string;
  config: BenchmarkConfig;
};

export const BENCHMARK_PRESETS: BenchmarkPreset[] = [
  {
    id: "quick-smoke",
    label: "Quick smoke",
    description: "30-second sanity check: one short prompt, one generation.",
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
];

export function listPresets(): BenchmarkPreset[] {
  return BENCHMARK_PRESETS;
}

export function getPreset(id: string): BenchmarkPreset | undefined {
  return BENCHMARK_PRESETS.find((p) => p.id === id);
}
