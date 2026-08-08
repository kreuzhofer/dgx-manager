import { describe, it, expect } from "vitest";
import { it as itProp, fc } from "@fast-check/vitest";
import {
  buildDgxrunDockerArgs,
  tokenizeCommand,
  fillPlaceholders,
  forceMpExecutor,
  shellQuote,
  missingModDirs,
  FABRIC_ENV_DEFAULTS,
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

  // Every upstream sparkrun recipe writes its command with backslash line
  // continuations. Treating `\<newline>` as an escaped literal emitted a bare
  // "\n" as its own argv element, so vLLM received a phantom argument between
  // every flag — silent corruption, since the recipe reads perfectly fine.
  it("drops backslash line continuations instead of emitting a newline token", () => {
    expect(tokenizeCommand("vllm serve m \\\n    --host 0.0.0.0 \\\n    --port 8000"))
      .toEqual(["vllm", "serve", "m", "--host", "0.0.0.0", "--port", "8000"]);
  });

  it("keeps a literal backslash-n (not a continuation) as an escaped character", () => {
    expect(tokenizeCommand("a b\\nc")).toEqual(["a", "bnc"]);
  });

  it("treats a bare newline as ordinary whitespace", () => {
    expect(tokenizeCommand("a\nb")).toEqual(["a", "b"]);
  });

  it("preserves a backslash inside single quotes", () => {
    expect(tokenizeCommand("'a\\\nb'")).toEqual(["a\\\nb"]);
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
    expect(s).toContain("--ulimit stack=67108864:67108864");
    expect(s).toContain("--shm-size 32gb");
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

describe("shellQuote", () => {
  it("wraps a plain token in single quotes", () => {
    expect(shellQuote(["vllm", "serve"])).toBe("'vllm' 'serve'");
  });

  it("survives a token containing a single quote", () => {
    expect(tokenizeCommand(shellQuote(["it's"]))).toEqual(["it's"]);
  });

  /**
   * Invariant: quoting an argv and tokenizing it back yields the original argv,
   * for ANY tokens. This is the load-bearing property of the mod wrapper — once
   * the serve command is embedded inside `bash -c '…'`, a quoting bug silently
   * reshapes vLLM's arguments instead of failing. The JSON-bearing args
   * (`--speculative-config '{"method":"dspark",…}'`) are exactly the shape that
   * has broken before, so they must survive a round trip untouched.
   */
  itProp.prop([fc.array(fc.string(), { minLength: 1, maxLength: 12 })])(
    "tokenizeCommand(shellQuote(argv)) === argv",
    (argv) => {
      expect(tokenizeCommand(shellQuote(argv))).toEqual(argv);
    },
  );
});

describe("missingModDirs", () => {
  const exists = (p: string) => p === "/opt/dgx-agent/mods/present";

  it("reports a mod whose directory is absent on this node", () => {
    expect(missingModDirs(["present", "absent"], "/opt/dgx-agent/mods", exists))
      .toEqual(["absent"]);
  });

  it("reports nothing when every mod is present, or none are declared", () => {
    expect(missingModDirs(["present"], "/opt/dgx-agent/mods", exists)).toEqual([]);
    expect(missingModDirs(undefined, "/opt/dgx-agent/mods", exists)).toEqual([]);
  });
});

describe("buildDgxrunDockerArgs — mods", () => {
  const modded: DgxrunRecipe = { ...glmRecipe, mods: ["instanttensor-hybrid-draft-loader"] };
  const argv = buildDgxrunDockerArgs(modded, { ...baseOpts, rank: 1 });

  it("bind-mounts each mod read-only under /mods", () => {
    expect(argv.join(" ")).toContain(
      "-v /opt/dgx-agent/mods/instanttensor-hybrid-draft-loader:" +
      "/mods/instanttensor-hybrid-draft-loader:ro",
    );
  });

  it("runs the mod before serve, and execs serve so it stays the main process", () => {
    const script = argv[argv.length - 1];
    expect(argv[argv.length - 3]).toBe("bash");
    expect(argv[argv.length - 2]).toBe("-c");
    expect(script).toMatch(
      /^bash \/mods\/instanttensor-hybrid-draft-loader\/run\.sh && exec /,
    );
  });

  // The wrapper is only safe if the serve command it embeds is the SAME argv the
  // unwrapped path would have run — same placeholders, same JSON, same
  // distributed flags, same --headless. Anything less and mods silently change
  // how the model is launched.
  it("embeds exactly the argv the unwrapped launch would have used", () => {
    const script = argv[argv.length - 1];
    const embedded = tokenizeCommand(script.slice(script.indexOf(" exec ") + " exec ".length));

    const plain = buildDgxrunDockerArgs(glmRecipe, { ...baseOpts, rank: 1 });
    const serveStart = plain.indexOf(glmRecipe.container) + 1;
    expect(embedded).toEqual(plain.slice(serveStart));
  });

  it("keeps the distributed flags and --headless INSIDE the wrapper, not after it", () => {
    const script = argv[argv.length - 1];
    expect(script).toContain("--node-rank");
    expect(script).toContain("--headless");
    expect(argv.slice(argv.indexOf("-c"))).not.toContain("--nnodes");
  });

  it("chains multiple mods in declaration order, failing fast on the first", () => {
    const two = buildDgxrunDockerArgs({ ...glmRecipe, mods: ["a", "b"] }, { ...baseOpts, rank: 0 });
    expect(two[two.length - 1]).toMatch(
      /^bash \/mods\/a\/run\.sh && bash \/mods\/b\/run\.sh && exec /,
    );
  });

  // Regression guard for the six existing dgxrun recipes: no mods must mean no
  // wrapper at all, so the serve argv is still the container's command.
  it("does NOT wrap when the recipe declares no mods", () => {
    const plain = buildDgxrunDockerArgs(glmRecipe, { ...baseOpts, rank: 0 });
    expect(plain).not.toContain("-c");
    expect(plain[plain.indexOf(glmRecipe.container) + 1]).toBe("vllm");
  });

  it("rejects a mod name that could escape the mods directory", () => {
    for (const bad of ["../evil", "a/b", "", "/abs"]) {
      expect(() => buildDgxrunDockerArgs({ ...glmRecipe, mods: [bad] }, { ...baseOpts, rank: 0 }))
        .toThrow(/mod name/i);
    }
  });
});

describe("buildDgxrunDockerArgs — fabric env defaults", () => {
  // The NCCL/GLOO block was copy-pasted identically into all six dgxrun recipes,
  // and the official upstream recipes omit it entirely because sparkrun injects
  // it per node. Without it NCCL silently falls back to TCP over the management
  // NIC — a working-but-slow deploy with no error to notice.
  it("injects the fabric env so a recipe need not carry it", () => {
    const bare: DgxrunRecipe = { ...glmRecipe, env: {} };
    const s = buildDgxrunDockerArgs(bare, { ...baseOpts, rank: 0 }).join(" ");
    expect(s).toContain("-e NCCL_NET=IB");
    expect(s).toContain("-e NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0");
    expect(s).toContain("-e NCCL_SOCKET_IFNAME=enP7s7,enp1s0f0np0,enP2p1s0f0np0");
    expect(s).toContain("-e GLOO_SOCKET_IFNAME=enp1s0f0np0");
    expect(s).toContain("-e OMP_NUM_THREADS=4");
    expect(s).toContain("-e TRANSFORMERS_OFFLINE=1");
  });

  it("lets a recipe override any fabric default (recipe env comes last)", () => {
    const r: DgxrunRecipe = { ...glmRecipe, env: { NCCL_DEBUG: "INFO" } };
    const argv = buildDgxrunDockerArgs(r, { ...baseOpts, rank: 0 });
    const defIdx = argv.findIndex((x, i) => argv[i - 1] === "-e" && x === "NCCL_DEBUG=WARN");
    const ovrIdx = argv.findIndex((x, i) => argv[i - 1] === "-e" && x === "NCCL_DEBUG=INFO");
    expect(defIdx).toBeGreaterThanOrEqual(0);
    expect(ovrIdx).toBeGreaterThan(defIdx);
  });

  it("matches the block the existing recipes hardcode", () => {
    expect(FABRIC_ENV_DEFAULTS.NCCL_IB_GID_INDEX).toBe("3");
    expect(FABRIC_ENV_DEFAULTS.NCCL_CROSS_NIC).toBe("1");
    expect(FABRIC_ENV_DEFAULTS.NCCL_CUMEM_ENABLE).toBe("0");
    expect(FABRIC_ENV_DEFAULTS.NCCL_IGNORE_CPU_AFFINITY).toBe("1");
    expect(FABRIC_ENV_DEFAULTS.NCCL_IB_DISABLE).toBe("0");
  });
});

describe("buildDgxrunDockerArgs — HF cache defaults", () => {
  // dgxrun owns the /cache/huggingface bind-mount, so it must default HF_HOME
  // there + offline; else a recipe that omits them (like the registry recipe,
  // which relies on sparkrun) re-downloads the weights from the HF Hub.
  it("defaults HF_HOME=/cache/huggingface and HF_HUB_OFFLINE=1 when the recipe omits them", () => {
    const s = buildDgxrunDockerArgs(glmRecipe, { ...baseOpts, rank: 0 }).join(" ");
    expect(s).toContain("-e HF_HOME=/cache/huggingface");
    expect(s).toContain("-e HF_HUB_OFFLINE=1");
  });

  it("lets a recipe env override the HF default (recipe value comes after → docker uses it)", () => {
    const r: DgxrunRecipe = { ...glmRecipe, env: { ...glmRecipe.env, HF_HOME: "/custom/hf" } };
    const argv = buildDgxrunDockerArgs(r, { ...baseOpts, rank: 0 });
    const defIdx = argv.findIndex((x, i) => argv[i - 1] === "-e" && x === "HF_HOME=/cache/huggingface");
    const ovrIdx = argv.findIndex((x, i) => argv[i - 1] === "-e" && x === "HF_HOME=/custom/hf");
    expect(defIdx).toBeGreaterThanOrEqual(0);
    expect(ovrIdx).toBeGreaterThan(defIdx);
  });
});
