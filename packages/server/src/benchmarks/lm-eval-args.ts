import type { AccuracyConfig } from "./presets.js";

/**
 * Per-request timeout in seconds, always emitted so the effective value is
 * visible in the stored args rather than silently inherited.
 *
 * lm-eval defaults to 300 (`api_models.py: timeout: int = 300`), applied as
 * aiohttp `ClientTimeout(total=...)` — it bounds the ENTIRE request including
 * generation. That makes it a function of maxGenToks divided by the model's
 * per-request throughput, not of endpoint health, and 300 is far too low for a
 * slow model under concurrency.
 *
 * 1800 is sized from measurement: Muse Glimmer (BF16, GB10) does ~17.6 tok/s
 * single-stream, so at num_concurrent=8 a 4096-token item needs well over 300s.
 * On 2026-08-29 that livelocked a GPQA run at 119/198 with 132 timeouts — every
 * retry hit the identical wall, and with lm-eval's max_retries=3 the items would
 * eventually have been scored WRONG. Deliberately a flat default rather than one
 * derived from maxGenToks: the per-request rate depends on the model and the
 * hardware, which this module cannot know. Override per run when it matters.
 */
export const DEFAULT_TIMEOUT_S = 1800;

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
  // Concurrent requests. Sanitize to a positive int (it's interpolated into the
  // comma-delimited model_args) and default to 1. Set higher (matching the eval
  // deployment's --max-num-seqs) to batch requests against the eval recipe.
  const nc = Number.isInteger(config.numConcurrent) && (config.numConcurrent as number) > 0
    ? (config.numConcurrent as number)
    : 1;
  // Same sanitising as numConcurrent: it is interpolated into the comma-delimited
  // model_args, so a non-positive or fractional value falls back to the default.
  const to = Number.isInteger(config.timeout) && (config.timeout as number) > 0
    ? (config.timeout as number)
    : DEFAULT_TIMEOUT_S;
  const modelArgs = [
    `base_url=${target.baseUrl}/chat/completions`,
    `model=${target.modelName}`,
    `num_concurrent=${nc}`,
    `timeout=${to}`,
    "tokenized_requests=False",
  ].join(",");

  const args: string[] = [
    "--model", "local-chat-completions",
    "--model_args", modelArgs,
    "--tasks", config.tasks.join(","),
    "--gen_kwargs", `max_gen_toks=${config.maxGenToks}`,
    "--seed", String(config.seed),
    "--output_path", target.outputDir,
    // Per-sample outputs, always. Without them lm-eval writes only aggregate
    // numbers, and an item that was mis-scored rather than errored leaves no
    // trace to find it by.
    //
    // Concretely: the 2026-08-29 Qwen3.8-27B GPQA run had 33 of 198 items
    // return EMPTY content, because the model reasons past max_gen_toks and the
    // chat template defaults `reasoning_effort` to `xhigh` (its maximum) with no
    // way for lm-eval to override it per request. lm-eval substitutes
    // LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER and scores those WRONG — silently.
    // The reported 70.7% was therefore a floor, not a score, and without sample
    // logs the failures could not be identified, re-run, or audited; the only
    // evidence was a count of warnings in the run log.
    //
    // Costs disk (samples embed prompts + full generations) and nothing else.
    // That is a good trade against publishing a number no one can check.
    "--log_samples",
  ];
  // Only emitted when set, so a preset that does not ask for it produces the
  // exact argv it produced before this existed.
  const instruction = typeof config.systemInstruction === "string" ? config.systemInstruction.trim() : "";
  if (instruction !== "") args.push("--system_instruction", instruction);
  if (config.applyChatTemplate) args.push("--apply_chat_template");
  if (config.limit !== null) args.push("--limit", String(config.limit));
  if (config.numFewshot !== null) args.push("--num_fewshot", String(config.numFewshot));
  return args;
}
