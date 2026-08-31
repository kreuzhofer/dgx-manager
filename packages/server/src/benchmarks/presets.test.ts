import { describe, expect, it } from "vitest";
import { AccuracyConfig, BENCHMARK_PRESETS, BenchmarkConfig, getPreset, listPresets } from "./presets.js";

describe("BENCHMARK_PRESETS", () => {
  it("exposes the five throughput presets plus four tool-eval presets by id", () => {
    expect(listPresets().map((p) => p.id).sort()).toEqual([
      "acc-bbh-full",
      "acc-bbh-quick",
      "acc-gpqa-diamond-full",
      "acc-gpqa-diamond-full-longgen",
      "acc-gpqa-diamond-full-longgen-formatted",
      "acc-gpqa-diamond-quick",
      "acc-gsm8k-full",
      "acc-gsm8k-quick",
      "acc-ifeval-full",
      "acc-ifeval-quick",
      "acc-math-hard-full",
      "acc-math-hard-quick",
      "acc-mmlu-pro-full",
      "acc-mmlu-pro-quick",
      "chat-long",
      "chat-short",
      "code-32k",
      "quick-smoke",
      "throughput",
      "tool-eval-full",
      "tool-eval-hardmode",
      "tool-eval-pressure",
      "tool-eval-quick",
    ]);
  });

  it("every throughput preset has at least one pp value, one tg value, and runs>=1", () => {
    for (const p of listPresets().filter((p) => p.kind === "throughput")) {
      const cfg = p.config as BenchmarkConfig;
      expect(cfg.pp.length).toBeGreaterThan(0);
      expect(cfg.tg.length).toBeGreaterThan(0);
      expect(cfg.runs).toBeGreaterThanOrEqual(1);
    }
  });

  it("getPreset returns the preset by id", () => {
    const p = getPreset("quick-smoke");
    expect(p?.id).toBe("quick-smoke");
  });

  it("getPreset returns undefined for unknown ids", () => {
    expect(getPreset("does-not-exist")).toBeUndefined();
  });

  it("quick-smoke is small enough to finish in under a minute on a single GPU", () => {
    const p = getPreset("quick-smoke")!;
    const cfg = p.config as BenchmarkConfig;
    // Heuristic: a single short prompt × generation × 1 run.
    expect(cfg.pp).toEqual([128]);
    expect(cfg.tg).toEqual([32]);
    expect(cfg.runs).toBe(1);
    expect(cfg.concurrency).toEqual([1]);
  });
});

// The presets themselves are kept narrow; if you add or rename one, update
// this list-level test too — that's intentional, presets are part of the
// product surface.

describe("tool-eval presets", () => {
  const ids = ["tool-eval-quick", "tool-eval-full", "tool-eval-hardmode", "tool-eval-pressure"];

  it("registers all four tool-eval presets with kind 'tool-eval'", () => {
    for (const id of ids) {
      const p = getPreset(id);
      expect(p, `preset ${id} should exist`).toBeDefined();
      expect(p!.kind).toBe("tool-eval");
    }
  });

  it("keeps the five throughput presets tagged kind 'throughput'", () => {
    const throughputIds = ["quick-smoke", "chat-short", "chat-long", "code-32k", "throughput"];
    for (const id of throughputIds) {
      expect(getPreset(id)!.kind).toBe("throughput");
    }
  });

  it("maps each tool-eval preset to the documented flag combination", () => {
    const cfg = (id: string) => getPreset(id)!.config as {
      short: boolean; hardmode: boolean; contextPressure: number | null; seed: number;
    };
    expect(cfg("tool-eval-quick")).toMatchObject({ short: true, hardmode: false, contextPressure: null });
    expect(cfg("tool-eval-full")).toMatchObject({ short: false, hardmode: false, contextPressure: null });
    expect(cfg("tool-eval-hardmode")).toMatchObject({ short: false, hardmode: true, contextPressure: null });
    expect(cfg("tool-eval-pressure")).toMatchObject({ short: false, hardmode: false, contextPressure: 0.75 });
  });

  it("every preset carries a kind field", () => {
    for (const p of BENCHMARK_PRESETS) {
      expect(["throughput", "tool-eval", "accuracy"]).toContain(p.kind);
    }
  });
});

describe("accuracy presets", () => {
  const benches = ["ifeval", "mmlu-pro", "gpqa-diamond", "gsm8k", "bbh", "math-hard"];

  it("registers a quick and a full variant per benchmark, all kind 'accuracy'", () => {
    for (const b of benches) {
      const quick = getPreset(`acc-${b}-quick`);
      const full = getPreset(`acc-${b}-full`);
      expect(quick, `acc-${b}-quick should exist`).toBeDefined();
      expect(full, `acc-${b}-full should exist`).toBeDefined();
      expect(quick!.kind).toBe("accuracy");
      expect(full!.kind).toBe("accuracy");
    }
  });

  it("quick variants set a numeric limit; full variants set limit null", () => {
    for (const b of benches) {
      const quick = getPreset(`acc-${b}-quick`)!.config as import("./presets.js").AccuracyConfig;
      const full = getPreset(`acc-${b}-full`)!.config as import("./presets.js").AccuracyConfig;
      expect(typeof quick.limit).toBe("number");
      expect(full.limit).toBeNull();
    }
  });

  it("every accuracy preset has primaryTask ∈ tasks and a non-empty primaryMetric", () => {
    for (const p of listPresets().filter((p) => p.kind === "accuracy")) {
      const cfg = p.config as import("./presets.js").AccuracyConfig;
      expect(cfg.tasks.length).toBeGreaterThan(0);
      expect(cfg.tasks).toContain(cfg.primaryTask);
      expect(cfg.primaryMetric.length).toBeGreaterThan(0);
      expect(cfg.reasoning).toBe(true);
      expect(cfg.applyChatTemplate).toBe(true);
      expect(cfg.maxGenToks).toBeGreaterThan(0);
    }
  });
});

describe("long-generation accuracy variants", () => {
  // A reasoning model that thinks past max_gen_toks returns EMPTY content, which
  // lm-eval fills with LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER and scores WRONG —
  // silently, with no error anywhere. Measured on Qwen3.8-27B 2026-08-29: 33% of
  // GPQA-Diamond items came back null at the standard 4096 cap. GLM-5.2 ran the
  // same preset with 0 errors, so this is per-model, not a broken default —
  // hence an ADDITIVE variant rather than raising the shared cap, which would
  // silently invalidate comparisons against already-published numbers.
  it("offers a GPQA variant with a generation cap big enough for long reasoning", () => {
    const p = getPreset("acc-gpqa-diamond-full-longgen")!;
    expect(p).toBeDefined();
    expect(p.kind).toBe("accuracy");
    const cfg = p.config as AccuracyConfig;
    expect(cfg.maxGenToks).toBeGreaterThanOrEqual(32768);
  });

  it("differs from the standard full preset ONLY in the cap and its matching timeout", () => {
    const base = getPreset("acc-gpqa-diamond-full")!.config as AccuracyConfig;
    const long = getPreset("acc-gpqa-diamond-full-longgen")!.config as AccuracyConfig;
    expect(long.maxGenToks).toBeGreaterThan(base.maxGenToks);
    const strip = (c: AccuracyConfig) => ({ ...c, maxGenToks: 0, timeout: 0 });
    expect(strip(long)).toEqual(strip(base));
  });

  // The timeout bounds the WHOLE request, generation included — lm-eval applies it as
  // aiohttp ClientTimeout(total=...). A cap the timeout cannot reach is not a larger
  // budget, it is a guaranteed failure. The 2026-08-30 GPQA run died at 137/198 after
  // 4h34m with ServerDisconnectedError for exactly this reason: 32768 tokens inside the
  // 1800s default needs >18 tok/s per request, and this fleet serves roughly 5 at
  // numConcurrent 8 (Qwen3.8-27B measures 10.75 tok/s single-stream). Encode the rule
  // that lm-eval-args.ts already states in prose.
  const SLOWEST_PER_REQUEST_TOKS_PER_SEC = 5;
  it("gives every long-generation variant a timeout its own cap can actually reach", () => {
    const longGen = listPresets().filter((x) => x.id.endsWith("-full-longgen"));
    expect(longGen.length).toBeGreaterThan(0);
    for (const p of longGen) {
      const cfg = p.config as AccuracyConfig;
      expect(cfg.timeout, `${p.id} must set an explicit timeout`).toBeDefined();
      expect(
        cfg.timeout! * SLOWEST_PER_REQUEST_TOKS_PER_SEC,
        `${p.id}: timeout ${cfg.timeout}s cannot reach ${cfg.maxGenToks} tokens`,
      ).toBeGreaterThanOrEqual(cfg.maxGenToks);
    }
  });

  it("leaves the standard preset's cap untouched, so old numbers stay comparable", () => {
    expect((getPreset("acc-gpqa-diamond-full")!.config as AccuracyConfig).maxGenToks).toBe(4096);
  });
});

describe("answer-format preset variant", () => {
  // GPQA's prompt never states an answer format: doc_to_text ends "Let's think
  // step by step: " and the filters then demand either "The answer is " or a
  // parenthesised capital. Muse Glimmer reasoned correctly, wrote "This is
  // choice B", matched neither, and scored 22.73 against a published 83.5.
  // Instructing the format lifted the same model to 82.83 on the same 198 items.
  //
  // ADDITIVE, exactly as longGenToks is: an instruction changes what the number
  // MEANS, so it must never appear on the presets whose values are already
  // baselines (GLM-5.2 67.68, DeepSeek 55.05, Qwen 76.77, GLM-5.3 81.31 were all
  // measured WITHOUT one). A separate id is also what makes an instructed number
  // reproducible from the dashboard instead of requiring a remembered string.
  it("emits a formatted variant alongside the plain longgen preset", () => {
    const ids = listPresets().map((p) => p.id);
    expect(ids).toContain("acc-gpqa-diamond-full-longgen");
    expect(ids).toContain("acc-gpqa-diamond-full-longgen-formatted");
  });

  it("carries an instruction that satisfies the strict-match filter", () => {
    const p = getPreset("acc-gpqa-diamond-full-longgen-formatted")!;
    const cfg = p.config as AccuracyConfig;
    // strict-match is regex "(?<=The answer is )(.*)(?=.)" — the instruction is
    // worthless unless it asks for that literal phrasing.
    expect(cfg.systemInstruction).toContain("The answer is");
    expect(cfg.maxGenToks).toBe(32768);
  });

  it("leaves the baseline presets free of any instruction", () => {
    for (const id of ["acc-gpqa-diamond-quick", "acc-gpqa-diamond-full", "acc-gpqa-diamond-full-longgen"]) {
      const p = getPreset(id)!;
      expect((p.config as AccuracyConfig).systemInstruction).toBeUndefined();
    }
  });

  it("adds no formatted variant for benches without an instruction", () => {
    const ids = listPresets().map((p) => p.id);
    expect(ids).not.toContain("acc-ifeval-full-longgen-formatted");
    expect(ids).not.toContain("acc-gsm8k-full-longgen-formatted");
  });
});

