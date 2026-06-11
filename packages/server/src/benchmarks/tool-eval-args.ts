import type { ToolEvalConfig } from "./presets.js";

export type ToolEvalTarget = {
  baseUrl: string;   // OpenAI-compatible base URL, already including /v1
  modelName: string; // passed explicitly to skip tool-eval-bench's picker
  outputPath: string;
};

// tool-eval-bench is an OpenAI-compatible CLI. We always pass --model so the
// interactive /v1/models picker can never block a headless run, and
// --json-file so results land at our conventional result.json path
// (--json-file implies --json). Variant flags are boolean/optional toggles.
export function buildToolEvalArgs(
  config: ToolEvalConfig,
  target: ToolEvalTarget,
): string[] {
  const args: string[] = [
    "--base-url", target.baseUrl,
    "--model", target.modelName,
    "--json-file", target.outputPath,
    "--seed", String(config.seed),
  ];
  if (config.short) args.push("--short");
  if (config.hardmode) args.push("--hardmode");
  if (config.contextPressure !== null) {
    args.push("--context-pressure", String(config.contextPressure));
  }
  return args;
}
