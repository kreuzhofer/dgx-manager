import { describe, expect, it } from "vitest";
import type { AccuracyConfig } from "./presets.js";
import { buildLmEvalArgs, DEFAULT_TIMEOUT_S } from "./lm-eval-args.js";

const base: AccuracyConfig = {
  tasks: ["ifeval"],
  primaryTask: "ifeval",
  primaryMetric: "prompt_level_strict_acc",
  limit: 100,
  numFewshot: null,
  maxGenToks: 2048,
  applyChatTemplate: true,
  reasoning: true,
  seed: 42,
};
const target = { baseUrl: "http://10.0.0.1:8000/v1", modelName: "m", outputDir: "/out" };

function valueAfter(args: string[], flag: string): string | undefined {
  const i = args.indexOf(flag);
  return i < 0 ? undefined : args[i + 1];
}

describe("buildLmEvalArgs", () => {
  it("targets local-chat-completions with the /chat/completions base_url and headless args", () => {
    const args = buildLmEvalArgs(base, target);
    expect(valueAfter(args, "--model")).toBe("local-chat-completions");
    expect(valueAfter(args, "--model_args")).toBe(
      `base_url=http://10.0.0.1:8000/v1/chat/completions,model=m,num_concurrent=1,timeout=${DEFAULT_TIMEOUT_S},tokenized_requests=False`,
    );
    expect(valueAfter(args, "--tasks")).toBe("ifeval");
    expect(valueAfter(args, "--gen_kwargs")).toBe("max_gen_toks=2048");
    expect(valueAfter(args, "--seed")).toBe("42");
    expect(valueAfter(args, "--output_path")).toBe("/out");
  });

  it("joins multiple tasks with commas", () => {
    const args = buildLmEvalArgs({ ...base, tasks: ["a", "b"] }, target);
    expect(valueAfter(args, "--tasks")).toBe("a,b");
  });

  it("includes --apply_chat_template only when applyChatTemplate is set", () => {
    expect(buildLmEvalArgs(base, target)).toContain("--apply_chat_template");
    expect(buildLmEvalArgs({ ...base, applyChatTemplate: false }, target)).not.toContain("--apply_chat_template");
  });

  it("includes --limit only when limit is non-null", () => {
    expect(valueAfter(buildLmEvalArgs(base, target), "--limit")).toBe("100");
    expect(buildLmEvalArgs({ ...base, limit: null }, target)).not.toContain("--limit");
  });

  it("includes --num_fewshot only when numFewshot is non-null (0 is valid)", () => {
    expect(buildLmEvalArgs({ ...base, numFewshot: null }, target)).not.toContain("--num_fewshot");
    expect(valueAfter(buildLmEvalArgs({ ...base, numFewshot: 0 }, target), "--num_fewshot")).toBe("0");
    expect(valueAfter(buildLmEvalArgs({ ...base, numFewshot: 5 }, target), "--num_fewshot")).toBe("5");
  });
});

describe("num_concurrent plumbing", () => {
  const base: AccuracyConfig = { tasks:["ifeval"], primaryTask:"ifeval", primaryMetric:"x", limit:null, numFewshot:null, maxGenToks:2048, applyChatTemplate:true, reasoning:false, seed:1 };
  const tgt = { baseUrl:"http://h/v1", modelName:"glm-5.2", outputDir:"/o" };
  const ma = (cfg: AccuracyConfig) => { const a=buildLmEvalArgs(cfg, tgt); return a[a.indexOf("--model_args")+1]; };
  it("defaults to num_concurrent=1 when unset", () => {
    expect(ma(base)).toContain("num_concurrent=1");
  });
  it("uses the configured numConcurrent", () => {
    expect(ma({...base, numConcurrent:16})).toContain("num_concurrent=16");
  });
  it("ignores a bogus numConcurrent (falls back to 1)", () => {
    expect(ma({...base, numConcurrent:0})).toContain("num_concurrent=1");
    expect(ma({...base, numConcurrent:-4})).toContain("num_concurrent=1");
    expect(ma({...base, numConcurrent:2.5})).toContain("num_concurrent=1");
  });
});

describe("timeout plumbing", () => {
  const base: AccuracyConfig = { tasks:["ifeval"], primaryTask:"ifeval", primaryMetric:"x", limit:null, numFewshot:null, maxGenToks:2048, applyChatTemplate:true, reasoning:false, seed:1 };
  const tgt = { baseUrl:"http://h/v1", modelName:"glm-5.2", outputDir:"/o" };
  const ma = (cfg: AccuracyConfig) => { const a=buildLmEvalArgs(cfg, tgt); return a[a.indexOf("--model_args")+1]; };

  // lm-eval's own default is 300s (api_models.py: `timeout: int = 300`, applied as
  // aiohttp ClientTimeout(total=...)). That bounds the WHOLE request including
  // generation, so it is a function of maxGenToks / tokens-per-second, not of the
  // endpoint being healthy. We always emit an explicit value so the effective
  // timeout is visible in the stored args rather than inherited invisibly.
  it("always emits an explicit timeout, so the effective value is never hidden", () => {
    expect(ma(base)).toContain(`timeout=${DEFAULT_TIMEOUT_S}`);
  });

  it("uses the configured timeout", () => {
    expect(ma({...base, timeout:1800})).toContain("timeout=1800");
  });

  it("ignores a bogus timeout (falls back to the default)", () => {
    expect(ma({...base, timeout:0})).toContain(`timeout=${DEFAULT_TIMEOUT_S}`);
    expect(ma({...base, timeout:-30})).toContain(`timeout=${DEFAULT_TIMEOUT_S}`);
    expect(ma({...base, timeout:12.5})).toContain(`timeout=${DEFAULT_TIMEOUT_S}`);
  });

  // Regression guard for the 2026-08-29 livelock: Muse Glimmer at BF16 on a GB10
  // does ~17.6 tok/s single-stream; at num_concurrent=8 each request gets a
  // fraction of that, so a 4096-token item cannot finish inside 300s. Every
  // attempt hit the same wall and the run stalled at 119/198 with 132 timeouts.
  // The default must leave room for maxGenToks at a realistic per-request rate.
  it("defaults high enough for a slow model to finish a large generation", () => {
    expect(DEFAULT_TIMEOUT_S).toBeGreaterThanOrEqual(1200);
  });
});

describe("--log_samples", () => {
  // Without per-sample logs, lm-eval writes ONLY aggregate results, so a run
  // that silently mis-scored items leaves no way to find out which ones. That
  // is not hypothetical: the 2026-08-29 Qwen3.8 GPQA run had 33/198 items come
  // back with empty content (the model reasoned past max_gen_toks). Each was
  // scored WRONG rather than erroring, and because samples were not logged the
  // only evidence was a count of warnings in the run log — the failing items
  // could not be identified, re-run, or audited.
  it("always logs per-sample outputs so mis-scored items can be identified", () => {
    expect(buildLmEvalArgs(base, target)).toContain("--log_samples");
  });

  // --log_samples writes alongside the results file, so it needs the same
  // --output_path. Asserted together because dropping either one silently
  // disables the diagnostics.
  it("keeps --output_path, which --log_samples writes alongside", () => {
    const args = buildLmEvalArgs(base, target);
    expect(args).toContain("--output_path");
    expect(args[args.indexOf("--output_path") + 1]).toBe(target.outputDir);
  });
});
