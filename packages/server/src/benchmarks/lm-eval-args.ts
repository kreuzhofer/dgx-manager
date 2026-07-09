import type { AccuracyConfig } from "./presets.js";

export type LmEvalTarget = {
  baseUrl: string;   // OpenAI base including /v1 (deployment or strip proxy)
  modelName: string; // vLLM served model id
  outputDir: string; // lm-eval writes results_*.json under here
};

// lm-eval's local-chat-completions model wants base_url to be the FULL
// /v1/chat/completions path. tokenized_requests=False keeps it from loading a
// local HF tokenizer for the served model; num_concurrent=1 suits one slow
// endpoint. --gen_kwargs / --limit / --num_fewshot are single-token key=val or
// scalar flags (not nargs), unlike llama-benchy's list flags.
export function buildLmEvalArgs(config: AccuracyConfig, target: LmEvalTarget): string[] {
  // NOTE: modelName is the vLLM served-model id (operator-controlled, from the
  // DB). It's interpolated into the comma/`=`-delimited model_args; a name
  // containing `,` or `=` could inject extra model_args. Real served ids are HF
  // repo ids so this is acceptable for now — revisit if names become arbitrary.
  const modelArgs = [
    `base_url=${target.baseUrl}/chat/completions`,
    `model=${target.modelName}`,
    "num_concurrent=1",
    "tokenized_requests=False",
  ].join(",");

  const args: string[] = [
    "--model", "local-chat-completions",
    "--model_args", modelArgs,
    "--tasks", config.tasks.join(","),
    "--gen_kwargs", `max_gen_toks=${config.maxGenToks}`,
    "--seed", String(config.seed),
    "--output_path", target.outputDir,
  ];
  if (config.applyChatTemplate) args.push("--apply_chat_template");
  if (config.limit !== null) args.push("--limit", String(config.limit));
  if (config.numFewshot !== null) args.push("--num_fewshot", String(config.numFewshot));
  return args;
}
