import { describe, expect, it } from "vitest";
import { buildLmEvalArgs } from "./lm-eval-args.js";
import type { AccuracyConfig } from "./presets.js";

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
