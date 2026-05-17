import type { BenchmarkConfig } from "./presets.js";

export type BenchyTarget = {
  baseUrl: string;
  modelName: string;
  outputPath: string;
};

// llama-benchy uses argparse nargs='+' for its list flags (--pp, --tg,
// --depth, --concurrency). That means each value must be its own argv
// token, e.g. ["--pp", "128", "512"]. A comma-joined string like "128,512"
// would be parsed as a single token and fail with an integer-conversion
// error.
export function buildBenchyArgs(
  config: BenchmarkConfig,
  target: BenchyTarget,
): string[] {
  const args: string[] = [
    "--base-url", target.baseUrl,
    "--model", target.modelName,
    "--format", "json",
    "--save-result", target.outputPath,
    "--pp", ...config.pp.map(String),
    "--tg", ...config.tg.map(String),
    "--depth", ...config.depth.map(String),
    "--concurrency", ...config.concurrency.map(String),
    "--runs", String(config.runs),
    "--latency-mode", config.latencyMode,
  ];
  if (config.enablePrefixCaching) args.push("--enable-prefix-caching");
  if (config.skipCoherence) args.push("--skip-coherence");
  return args;
}
