import { describe, expect, it } from "vitest";
import type { AccuracyConfig } from "./presets.js";
import { buildLmEvalArgs } from "./lm-eval-args.js";

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
      "base_url=http://10.0.0.1:8000/v1/chat/completions,model=m,num_concurrent=1,tokenized_requests=False",
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
