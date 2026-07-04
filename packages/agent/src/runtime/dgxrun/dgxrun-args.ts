/**
 * Pure argv builder for the dgxrun runtime — our own multi-node vLLM launcher
 * that expresses the `mp` (multiproc) executor config sparkrun cannot (per-node
 * `--nnodes/--node-rank/--master-addr` + `--ipc host`). Generalises the
 * validated `scratchpad/glm52-mp-launch.sh` reference launcher.
 *
 * Deterministic + IO-free so it can be unit-tested exactly like
 * `sparkrun-args.ts`. `dgxrun.ts` (the effectful lifecycle) feeds it resolved
 * values (weights dir, container name) and runs the returned `docker` argv.
 */

/** The subset of a resolved recipe dgxrun needs to launch one rank. */
export interface DgxrunRecipe {
  /** HF model id — fills the `{model}` placeholder. */
  model?: string;
  /** Container image ref (recipe `container:`). */
  container: string;
  /** Recipe `env:` block — emitted verbatim as `-e KEY=VALUE`. */
  env?: Record<string, string | number | boolean>;
  /** Recipe `command:` template with `{placeholder}` tokens. */
  command: string;
  /** Recipe `defaults:` — the placeholder source (port, tensor_parallel, …). */
  defaults?: Record<string, unknown>;
}

export interface DgxrunLaunchOptions {
  /** Container name to create (`dgxrun_<deploymentId>`). */
  containerName: string;
  /** Host HF cache dir bind-mounted to `/cache/huggingface`. */
  weightsDir: string;
  /** This node's rank; 0 = head. */
  rank: number;
  /** Total node count → `--nnodes`. */
  nnodes: number;
  /** Head node's management IP → `--master-addr`. */
  masterAddr: string;
  /** torch TCPStore rendezvous port → `--master-port`. */
  masterPort: number;
  /** `--headless` is appended when true; defaults to `rank > 0`. */
  headless?: boolean;
  /** Per-deploy overrides for command placeholders (port, tensorParallel, …).
   *  Keys are matched against both the raw name and its snake_case default
   *  key, so `tensorParallel` overrides the `{tensor_parallel}` placeholder. */
  params?: Record<string, string | number | undefined>;
  /** `--shm-size` value (belt-and-suspenders with `--ipc host`). Default 10gb. */
  shmSize?: string;
}

/** Map a camelCase override key onto the recipe's snake_case placeholder name. */
const OVERRIDE_ALIASES: Record<string, string> = {
  tensorParallel: "tensor_parallel",
  pipelineParallel: "pipeline_parallel",
  gpuMem: "gpu_memory_utilization",
  maxModelLen: "max_model_len",
  servedModelName: "served_model_name",
};

/**
 * Build the substitution map for `{placeholder}` tokens: recipe defaults first,
 * then `{model}`, then per-deploy overrides (which win). camelCase override
 * keys are also mapped onto their snake_case placeholder names so a config
 * `tensorParallel: 4` fills `{tensor_parallel}`.
 */
function buildSubstitutions(recipe: DgxrunRecipe, params: DgxrunLaunchOptions["params"]): Record<string, string> {
  const subs: Record<string, string> = {};
  for (const [k, v] of Object.entries(recipe.defaults ?? {})) {
    if (v != null && v !== "") subs[k] = String(v);
  }
  if (recipe.model) subs.model = recipe.model;
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v == null || v === "") continue;
    subs[k] = String(v);
    const alias = OVERRIDE_ALIASES[k];
    if (alias) subs[alias] = String(v);
  }
  return subs;
}

/**
 * Fill `{identifier}` placeholders from the substitution map. Only tokens
 * matching `{word}` are touched — inline JSON like `'{"model":"x"}'` (which
 * starts with `{"`, not `{word}`) is left intact. An unknown placeholder is
 * left verbatim (fail-loud in the launched command rather than silently blank).
 */
export function fillPlaceholders(template: string, subs: Record<string, string>): string {
  return template.replace(/\{([A-Za-z_][A-Za-z0-9_]*)\}/g, (whole, key: string) =>
    Object.prototype.hasOwnProperty.call(subs, key) ? subs[key] : whole,
  );
}

/**
 * Shell-aware tokenizer: split a command string into argv, honoring single
 * quotes (fully literal), double quotes (literal, `\"`/`\\` escapes), and
 * backslash escapes outside quotes. Needed because recipe `command:` templates
 * carry single-quoted JSON (`--speculative-config '{"model":…}'`) that must
 * survive as ONE argv element with its inner double-quotes intact.
 */
export function tokenizeCommand(cmd: string): string[] {
  const tokens: string[] = [];
  let cur = "";
  let has = false; // did the current token get any (even empty-quoted) content?
  let i = 0;
  const n = cmd.length;
  while (i < n) {
    const c = cmd[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      if (has) { tokens.push(cur); cur = ""; has = false; }
      i++;
      continue;
    }
    has = true;
    if (c === "'") {
      i++;
      while (i < n && cmd[i] !== "'") { cur += cmd[i]; i++; }
      i++; // skip closing quote
    } else if (c === '"') {
      i++;
      while (i < n && cmd[i] !== '"') {
        if (cmd[i] === "\\" && i + 1 < n && (cmd[i + 1] === '"' || cmd[i + 1] === "\\")) {
          cur += cmd[i + 1]; i += 2;
        } else { cur += cmd[i]; i++; }
      }
      i++; // skip closing quote
    } else if (c === "\\" && i + 1 < n) {
      cur += cmd[i + 1]; i += 2;
    } else {
      cur += c; i++;
    }
  }
  if (has) tokens.push(cur);
  return tokens;
}

/**
 * Force the distributed executor backend to `mp`. If the tokenized serve
 * command already carries `--distributed-executor-backend <x>`, rewrite `<x>`
 * to `mp`; otherwise append the flag. dgxrun ONLY supports mp multi-node —
 * `ray` is broken on our vLLM build (see the runner spec).
 */
export function forceMpExecutor(argv: string[]): string[] {
  const out = [...argv];
  const idx = out.indexOf("--distributed-executor-backend");
  if (idx >= 0 && idx + 1 < out.length) {
    out[idx + 1] = "mp";
  } else {
    out.push("--distributed-executor-backend", "mp");
  }
  return out;
}

/**
 * Build the full `docker run` argv (starting at `run`) for ONE rank.
 *
 * Layout (per the runner spec):
 *   run -d --name <name>
 *   --network host --ipc host --gpus all
 *   --device /dev/infiniband:/dev/infiniband
 *   --cap-add IPC_LOCK --ulimit memlock=-1:-1 --shm-size <shm>
 *   -v <weightsDir>:/cache/huggingface
 *   -e KEY=VALUE ...            (recipe env, verbatim)
 *   <image>
 *   <serve argv, executor forced to mp>
 *   --nnodes <n> --node-rank <rank> --master-addr <ip> --master-port <port>
 *   [--headless]               (rank > 0)
 */
export function buildDgxrunDockerArgs(recipe: DgxrunRecipe, opts: DgxrunLaunchOptions): string[] {
  if (!recipe.container) throw new Error("dgxrun recipe missing container image");
  if (!recipe.command || !recipe.command.trim()) throw new Error("dgxrun recipe missing command");

  const subs = buildSubstitutions(recipe, opts.params);
  const filled = fillPlaceholders(recipe.command, subs);
  const serve = forceMpExecutor(tokenizeCommand(filled));

  const headless = opts.headless ?? opts.rank > 0;
  const shmSize = opts.shmSize ?? "10gb";

  const args: string[] = [
    "run", "-d", "--name", opts.containerName,
    // Container flags — --ipc host is THE fix sparkrun couldn't express; IB
    // passthrough + IPC_LOCK + memlock are required or NCCL silently drops to TCP.
    "--network", "host",
    "--ipc", "host",
    "--gpus", "all",
    "--device", "/dev/infiniband:/dev/infiniband",
    "--cap-add", "IPC_LOCK",
    "--ulimit", "memlock=-1:-1",
    "--shm-size", shmSize,
    "-v", `${opts.weightsDir}:/cache/huggingface`,
  ];

  for (const [k, v] of Object.entries(recipe.env ?? {})) {
    args.push("-e", `${k}=${String(v)}`);
  }

  args.push(recipe.container, ...serve);

  args.push(
    "--nnodes", String(opts.nnodes),
    "--node-rank", String(opts.rank),
    "--master-addr", opts.masterAddr,
    "--master-port", String(opts.masterPort),
  );
  if (headless) args.push("--headless");

  return args;
}
