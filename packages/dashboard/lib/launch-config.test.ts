import { describe, expect, it } from "vitest";
import { parseLaunchConfig, LAUNCH_CONFIG_LABELS } from "./launch-config";

describe("parseLaunchConfig", () => {
  it("returns an empty array when config is null", () => {
    expect(parseLaunchConfig(null)).toEqual([]);
  });

  it("returns an empty array when config is an empty string", () => {
    expect(parseLaunchConfig("")).toEqual([]);
  });

  it("returns an empty array when config is invalid JSON", () => {
    expect(parseLaunchConfig("{not json")).toEqual([]);
  });

  it("returns an empty array when config parses to an empty object", () => {
    expect(parseLaunchConfig("{}")).toEqual([]);
  });

  it("returns an empty array when config parses to a JSON array", () => {
    // Defensive: typeof [] === "object" so the type guard must check
    // Array.isArray explicitly; otherwise array indices would be
    // emitted as bogus entries.
    expect(parseLaunchConfig("[1,2,3]")).toEqual([]);
  });

  it("returns labeled key-value pairs in canonical order", () => {
    const cfg = JSON.stringify({
      lora_alpha: 32,
      learning_rate: 0.0002,
      max_seq_length: 16384,
      lora_r: 16,
    });
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "learning_rate", label: "Learning rate", value: 0.0002 },
      { key: "max_seq_length", label: "Max seq length", value: 16384 },
      { key: "lora_r", label: "LoRA rank (r)", value: 16 },
      { key: "lora_alpha", label: "LoRA alpha", value: 32 },
    ]);
  });

  it("skips keys whose value is null", () => {
    const cfg = JSON.stringify({ learning_rate: null, batch_size: 1 });
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "batch_size", label: "Batch size", value: 1 },
    ]);
  });

  it("preserves unknown keys at the end with their raw key as label", () => {
    const cfg = JSON.stringify({ learning_rate: 1e-4, save_steps: 50 });
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "learning_rate", label: "Learning rate", value: 1e-4 },
      { key: "save_steps", label: "save_steps", value: 50 },
    ]);
  });

  it("emits only unknown keys when no canonical keys are present", () => {
    const cfg = JSON.stringify({ save_steps: 50, warmup_ratio: 0.1 });
    expect(parseLaunchConfig(cfg)).toEqual([
      { key: "save_steps", label: "save_steps", value: 50 },
      { key: "warmup_ratio", label: "warmup_ratio", value: 0.1 },
    ]);
  });

  it("LAUNCH_CONFIG_LABELS exposes the canonical label order", () => {
    expect(Object.keys(LAUNCH_CONFIG_LABELS)).toEqual([
      "learning_rate",
      "batch_size",
      "max_seq_length",
      "lora_r",
      "lora_alpha",
      "num_train_epochs",
      "max_steps",
    ]);
  });
});
