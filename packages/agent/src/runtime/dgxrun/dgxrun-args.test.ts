import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import {
  buildDgxrunDockerArgs,
  tokenizeCommand,
  fillPlaceholders,
  forceMpExecutor,
  type DgxrunRecipe,
} from "./dgxrun-args.js";

// A trimmed GLM-5.2-shaped recipe (the validation target): env block, a
// command template with JSON-bearing single-quoted args, and defaults.
const glmRecipe: DgxrunRecipe = {
  model: "CosmicRaisins/GLM-5.2-AWQ-INT4-15pct",
  container: "vllm-node-tf5-glm52-b12x:probe",
  env: {
    LD_PRELOAD: "/cache/huggingface/nccl-2.30.4/libnccl.so.2",
    VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS: "5400",
    NCCL_NET: "IB",
    NCCL_IB_DISABLE: "0",
  },
  command:
    "vllm serve {model} --served-model-name {served_model_name} --host {host} --port {port} " +
    "--trust-remote-code --enable-prefix-caching " +
    "--speculative-config '{\"model\":\"CosmicRaisins/GLM-5.2-MTP-INT4-aligned\",\"method\":\"mtp\",\"num_speculative_tokens\":3}' " +
    "-tp {tensor_parallel} --pipeline-parallel-size 1 --distributed-executor-backend mp " +
    "--max-model-len {max_model_len} --gpu-memory-utilization {gpu_memory_utilization} " +
    "--compilation-config '{\"cudagraph_mode\":\"FULL\"}'",
  defaults: {
    port: 8000,
    host: "0.0.0.0",
    tensor_parallel: 4,
    gpu_memory_utilization: 0.88,
    max_model_len: 87040,
    served_model_name: "glm-5.2",
  },
};

const baseOpts = {
  containerName: "dgxrun_dep123",
  weightsDir: "/mnt/tank/models",
  nnodes: 4,
  masterAddr: "192.168.44.36",
  masterPort: 29500,
};

describe("tokenizeCommand", () => {
  it("keeps single-quoted JSON as one token with inner double-quotes intact", () => {
    const argv = tokenizeCommand("a --cfg '{\"k\":\"v\",\"n\":3}' b");
    expect(argv).toEqual(["a", "--cfg", '{"k":"v","n":3}', "b"]);
  });

  it("collapses runs of whitespace and ignores leading/trailing space", () => {
    expect(tokenizeCommand("  vllm   serve  x  ")).toEqual(["vllm", "serve", "x"]);
  });
});

describe("fillPlaceholders", () => {
  it("fills {word} tokens but leaves inline JSON braces untouched", () => {
    const out = fillPlaceholders("--port {port} --cfg '{\"model\":\"x\"}'", { port: "8000" });
    expect(out).toBe("--port 8000 --cfg '{\"model\":\"x\"}'");
  });

  it("leaves an unknown placeholder verbatim (fail-loud, not blank)", () => {
    expect(fillPlaceholders("--x {missing}", {})).toBe("--x {missing}");
  });
});

describe("forceMpExecutor", () => {
  it("rewrites an existing ray backend to mp", () => {
    expect(forceMpExecutor(["a", "--distributed-executor-backend", "ray", "b"]))
      .toEqual(["a", "--distributed-executor-backend", "mp", "b"]);
  });
  it("appends the flag when absent", () => {
    expect(forceMpExecutor(["vllm", "serve"]))
      .toEqual(["vllm", "serve", "--distributed-executor-backend", "mp"]);
  });
});

describe("buildDgxrunDockerArgs — rank 0 (head)", () => {
  const argv = buildDgxrunDockerArgs(glmRecipe, { ...baseOpts, rank: 0 });
  const s = argv.join(" ");

  it("emits detached run with the deployment-scoped container name", () => {
    expect(argv.slice(0, 4)).toEqual(["run", "-d", "--name", "dgxrun_dep123"]);
  });

  it("sets the key container flags incl. --ipc host and IB passthrough", () => {
    expect(s).toContain("--network host");
    expect(s).toContain("--ipc host");
    expect(s).toContain("--gpus all");
    expect(s).toContain("--device /dev/infiniband:/dev/infiniband");
    expect(s).toContain("--cap-add IPC_LOCK");
    expect(s).toContain("--ulimit memlock=-1:-1");
    expect(s).toContain("--shm-size 10gb");
  });

  it("bind-mounts the weights dir to /cache/huggingface", () => {
    expect(s).toContain("-v /mnt/tank/models:/cache/huggingface");
  });

  it("passes every recipe env var as -e KEY=VALUE", () => {
    expect(s).toContain("-e LD_PRELOAD=/cache/huggingface/nccl-2.30.4/libnccl.so.2");
    expect(s).toContain("-e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=5400");
    expect(s).toContain("-e NCCL_NET=IB");
    expect(s).toContain("-e NCCL_IB_DISABLE=0");
  });

  it("places the image immediately before the serve command", () => {
    const imgIdx = argv.indexOf("vllm-node-tf5-glm52-b12x:probe");
    expect(imgIdx).toBeGreaterThan(0);
    expect(argv[imgIdx + 1]).toBe("vllm");
    expect(argv[imgIdx + 2]).toBe("serve");
  });

  it("fills placeholders from defaults", () => {
    expect(argv[argv.indexOf("--served-model-name") + 1]).toBe("glm-5.2");
    expect(argv[argv.indexOf("--host") + 1]).toBe("0.0.0.0");
    expect(argv[argv.indexOf("--port") + 1]).toBe("8000");
    expect(argv[argv.indexOf("-tp") + 1]).toBe("4");
    expect(argv[argv.indexOf("--max-model-len") + 1]).toBe("87040");
    expect(argv[argv.indexOf("--gpu-memory-utilization") + 1]).toBe("0.88");
    expect(argv).toContain("CosmicRaisins/GLM-5.2-AWQ-INT4-15pct");
  });

  it("keeps JSON args intact as single tokens", () => {
    expect(argv).toContain('{"cudagraph_mode":"FULL"}');
    expect(argv).toContain(
      '{"model":"CosmicRaisins/GLM-5.2-MTP-INT4-aligned","method":"mtp","num_speculative_tokens":3}',
    );
  });

  it("forces the mp executor backend", () => {
    expect(argv[argv.indexOf("--distributed-executor-backend") + 1]).toBe("mp");
  });

  it("appends the distributed args with rank 0 and NO --headless", () => {
    expect(argv[argv.indexOf("--nnodes") + 1]).toBe("4");
    expect(argv[argv.indexOf("--node-rank") + 1]).toBe("0");
    expect(argv[argv.indexOf("--master-addr") + 1]).toBe("192.168.44.36");
    expect(argv[argv.indexOf("--master-port") + 1]).toBe("29500");
    expect(argv).not.toContain("--headless");
  });
});

describe("buildDgxrunDockerArgs — rank > 0 (worker)", () => {
  const argv = buildDgxrunDockerArgs(glmRecipe, { ...baseOpts, rank: 2 });

  it("sets the correct node-rank and appends --headless", () => {
    expect(argv[argv.indexOf("--node-rank") + 1]).toBe("2");
    expect(argv[argv.indexOf("--nnodes") + 1]).toBe("4");
    expect(argv[argv.indexOf("--master-addr") + 1]).toBe("192.168.44.36");
    expect(argv).toContain("--headless");
  });

  it("still points --master-addr at the head, not itself", () => {
    expect(argv[argv.indexOf("--master-addr") + 1]).toBe("192.168.44.36");
  });
});

describe("buildDgxrunDockerArgs — overrides + validation", () => {
  it("lets camelCase params override snake_case placeholders", () => {
    const argv = buildDgxrunDockerArgs(glmRecipe, {
      ...baseOpts, rank: 0,
      params: { tensorParallel: 2, gpuMem: 0.9, maxModelLen: 40000, port: 8001 },
    });
    expect(argv[argv.indexOf("-tp") + 1]).toBe("2");
    expect(argv[argv.indexOf("--gpu-memory-utilization") + 1]).toBe("0.9");
    expect(argv[argv.indexOf("--max-model-len") + 1]).toBe("40000");
    expect(argv[argv.indexOf("--port") + 1]).toBe("8001");
  });

  it("throws when the recipe has no container image", () => {
    expect(() => buildDgxrunDockerArgs({ ...glmRecipe, container: "" }, { ...baseOpts, rank: 0 }))
      .toThrow(/container/i);
  });

  it("throws when the recipe has no command", () => {
    expect(() => buildDgxrunDockerArgs({ ...glmRecipe, command: "  " }, { ...baseOpts, rank: 0 }))
      .toThrow(/command/i);
  });

  /** Invariant: --headless appears iff the rank is a worker (rank > 0), and the
   *  emitted --node-rank always equals the requested rank, for any cluster size. */
  itProp.prop([fc.integer({ min: 0, max: 15 }), fc.integer({ min: 1, max: 16 })])(
    "headless iff rank>0, and --node-rank matches",
    (rank, nnodes) => {
      const argv = buildDgxrunDockerArgs(glmRecipe, { ...baseOpts, rank, nnodes });
      expect(argv[argv.indexOf("--node-rank") + 1]).toBe(String(rank));
      expect(argv.includes("--headless")).toBe(rank > 0);
    },
  );
});
