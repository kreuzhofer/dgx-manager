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

import { shQuote } from "../../jobs/sh-quote.js";

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
  /** Recipe `mods:` — names of vendored mods to apply before serving. */
  mods?: string[];
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
  /** `--shm-size` value (belt-and-suspenders with `--ipc host`). Default 32gb. */
  shmSize?: string;
  /** Host directory holding the vendored mods. Default `/opt/dgx-agent/mods`. */
  modsDir?: string;
  /**
   * Whether this host has the RoCE/InfiniBand fabric (`/dev/infiniband`).
   * Defaults to `true` — every DGX Spark has it, so the Spark argv is
   * unchanged. Set false on a host without it (the amd64 RTX-5090 box): the
   * device passthrough would make `docker run` fail outright, and the fabric
   * env names NICs that do not exist there.
   */
  hasInfiniband?: boolean;
}

/** Where the agent bundle installs the vendored mods on a node. */
export const DEFAULT_MODS_DIR = "/opt/dgx-agent/mods";

/** Where a mod is mounted inside the container. */
const CONTAINER_MODS_ROOT = "/mods";

/**
 * Fabric + process env every dgxrun launch needs, injected BEFORE the recipe's
 * own env so a recipe can still override any of it.
 *
 * All six dgxrun recipes carried this block byte-identically, and the upstream
 * sparkrun recipes carry none of it — sparkrun injects the equivalent per node
 * at `docker run` time. Leaving it to the recipe means every new recipe must
 * remember ten variables, and the failure when it doesn't is silent: NCCL falls
 * back to TCP over the management NIC and the deploy is merely slow.
 */
export const FABRIC_ENV_DEFAULTS: Record<string, string> = {
  NCCL_NET: "IB",
  NCCL_IB_DISABLE: "0",
  NCCL_IB_HCA: "rocep1s0f0,roceP2p1s0f0",
  NCCL_SOCKET_IFNAME: "enP7s7,enp1s0f0np0,enP2p1s0f0np0",
  NCCL_IB_GID_INDEX: "3",
  NCCL_CROSS_NIC: "1",
  NCCL_CUMEM_ENABLE: "0",
  NCCL_IGNORE_CPU_AFFINITY: "1",
  NCCL_DEBUG: "WARN",
  GLOO_SOCKET_IFNAME: "enp1s0f0np0",
  OMP_NUM_THREADS: "4",
  TRANSFORMERS_OFFLINE: "1",
};

/**
 * The subset of {@link FABRIC_ENV_DEFAULTS} that names physical fabric — IB
 * HCAs and the GB10 NIC interfaces. On a host without `/dev/infiniband` these
 * point at hardware that isn't there, and NCCL then fails to initialise rather
 * than quietly falling back, so they are dropped instead of passed through.
 *
 * The rest of the block (`NCCL_CUMEM_ENABLE`, `NCCL_IGNORE_CPU_AFFINITY`,
 * `NCCL_DEBUG`, `OMP_NUM_THREADS`, `TRANSFORMERS_OFFLINE`) is hardware-
 * independent and is injected either way. Filtering the single ordered map —
 * rather than composing two — keeps `-e` emission order identical on the
 * Sparks, which the golden argv tests pin.
 */
export const IB_FABRIC_ENV_KEYS: ReadonlySet<string> = new Set([
  "NCCL_NET",
  "NCCL_IB_DISABLE",
  "NCCL_IB_HCA",
  "NCCL_SOCKET_IFNAME",
  "NCCL_IB_GID_INDEX",
  "NCCL_CROSS_NIC",
  "GLOO_SOCKET_IFNAME",
]);

/** A mod name must be a single path segment — it becomes a bind-mount source. */
const MOD_NAME_RE = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

/**
 * Quote an argv into a single shell string that tokenizes back to the same
 * argv. Needed only for the mod wrapper, where the serve command is embedded in
 * `bash -c '…'` instead of being the container's argv directly.
 *
 * Per-token quoting is {@link shQuote}, deliberately shared with the benchmark
 * job wrapper rather than reimplemented — a second copy of a shell-quoting rule
 * is how the two drift and one of them becomes an injection bug.
 */
export function shellQuote(argv: string[]): string {
  return argv.map(shQuote).join(" ");
}

/**
 * Which of the declared mods have no directory on this node. A recipe can name
 * a mod newer than the agent bundle installed here, and a runtime that starts
 * without a mod it needed looks healthy and fails much later somewhere
 * unrelated — so the caller refuses the deploy instead.
 */
export function missingModDirs(
  mods: string[] | undefined,
  modsDir: string,
  exists: (path: string) => boolean,
): string[] {
  return (mods ?? []).filter((m) => !exists(`${modsDir}/${m}`));
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
 *
 * `\<newline>` is a line continuation and disappears, as in sh. Upstream recipes
 * write their command across many continued lines, and treating the backslash as
 * an ordinary escape emitted a bare newline as its own argv element between
 * every flag.
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
        if (cmd[i] === "\\" && cmd[i + 1] === "\n") {
          i += 2; // line continuation inside double quotes
        } else if (cmd[i] === "\\" && i + 1 < n && (cmd[i + 1] === '"' || cmd[i + 1] === "\\")) {
          cur += cmd[i + 1]; i += 2;
        } else { cur += cmd[i]; i++; }
      }
      i++; // skip closing quote
    } else if (c === "\\" && cmd[i + 1] === "\n") {
      // Line continuation: both characters vanish. If this token has no other
      // content the trailing `has` flag keeps it from becoming an empty argv
      // element, since the next character is whitespace or end-of-string.
      i += 2;
      if (!cur) has = false;
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
 *   --cap-add IPC_LOCK --ulimit memlock=-1:-1 --ulimit stack=64m --shm-size <shm>
 *   -v <weightsDir>:/cache/huggingface
 *   -v <modsDir>/<mod>:/mods/<mod>:ro ...   (one per declared mod)
 *   -e KEY=VALUE ...            (HF + fabric defaults, then recipe env)
 *   <image>
 *   <serve argv, executor forced to mp>
 *   --nnodes <n> --node-rank <rank> --master-addr <ip> --master-port <port>
 *   [--headless]               (rank > 0)
 *
 * With mods declared, the trailing serve argv is replaced by
 * `bash -c 'bash /mods/<mod>/run.sh && … && exec <same serve argv>'`. The `exec`
 * matters: serve stays the container's main process, so the exit code, log
 * stream and teardown contract are identical either way.
 */
export function buildDgxrunDockerArgs(recipe: DgxrunRecipe, opts: DgxrunLaunchOptions): string[] {
  if (!recipe.container) throw new Error("dgxrun recipe missing container image");
  if (!recipe.command || !recipe.command.trim()) throw new Error("dgxrun recipe missing command");

  const mods = recipe.mods ?? [];
  for (const m of mods) {
    if (!MOD_NAME_RE.test(m)) {
      throw new Error(`dgxrun: invalid mod name ${JSON.stringify(m)} — must be a single path segment`);
    }
  }

  const subs = buildSubstitutions(recipe, opts.params);
  const filled = fillPlaceholders(recipe.command, subs);
  const serve = forceMpExecutor(tokenizeCommand(filled));

  const headless = opts.headless ?? opts.rank > 0;
  const shmSize = opts.shmSize ?? "32gb";
  const modsDir = opts.modsDir ?? DEFAULT_MODS_DIR;
  const hasInfiniband = opts.hasInfiniband ?? true;

  const args: string[] = [
    "run", "-d", "--name", opts.containerName,
    // Container flags — --ipc host is THE fix sparkrun couldn't express; IB
    // passthrough + IPC_LOCK + memlock are required or NCCL silently drops to TCP.
    // The 64 MB stack matches sparkrun's launch; the JIT's template
    // instantiation recurses deeper than the 8 MB default.
    "--network", "host",
    "--ipc", "host",
    "--gpus", "all",
    ...(hasInfiniband ? ["--device", "/dev/infiniband:/dev/infiniband"] : []),
    "--cap-add", "IPC_LOCK",
    "--ulimit", "memlock=-1:-1",
    "--ulimit", "stack=67108864:67108864",
    "--shm-size", shmSize,
    "-v", `${opts.weightsDir}:/cache/huggingface`,
  ];

  // Mods are read-only: everything a mod writes goes into the container's own
  // Python tree, never back into the mount.
  for (const m of mods) {
    args.push("-v", `${modsDir}/${m}:${CONTAINER_MODS_ROOT}/${m}:ro`);
  }

  // On a host with no RoCE fabric, drop the env that names IB HCAs and GB10
  // NICs; the rest of the block is hardware-independent and still applies.
  const fabricEnv = hasInfiniband
    ? FABRIC_ENV_DEFAULTS
    : Object.fromEntries(
        Object.entries(FABRIC_ENV_DEFAULTS).filter(([k]) => !IB_FABRIC_ENV_KEYS.has(k)),
      );

  // dgxrun OWNS the `/cache/huggingface` bind-mount (weightsDir), so default HF
  // there and go offline — cluster weights are pre-staged on NFS. Pushed BEFORE
  // the recipe env so a recipe can still override (docker uses the last -e for a
  // repeated key). Without this, a recipe that omits HF_HOME (e.g. the registry
  // recipe, which relies on sparkrun to set it) re-downloads from the HF Hub
  // instead of using the mounted cache.
  const hfDefaults: Record<string, string> = {
    HF_HOME: "/cache/huggingface",
    HF_HUB_OFFLINE: "1",
    ...fabricEnv,
  };
  for (const [k, v] of Object.entries(hfDefaults)) {
    args.push("-e", `${k}=${v}`);
  }
  for (const [k, v] of Object.entries(recipe.env ?? {})) {
    args.push("-e", `${k}=${String(v)}`);
  }

  args.push(recipe.container);

  const serveArgv = [
    ...serve,
    "--nnodes", String(opts.nnodes),
    "--node-rank", String(opts.rank),
    "--master-addr", opts.masterAddr,
    "--master-port", String(opts.masterPort),
  ];
  if (headless) serveArgv.push("--headless");

  // No mods → serve IS the container's argv, exactly as before. With mods, the
  // command becomes a short shell script that applies each mod and then `exec`s
  // the same argv, so serve remains the container's main process and the exit
  // code, log stream and teardown contract are all unchanged.
  if (mods.length === 0) {
    args.push(...serveArgv);
    return args;
  }

  const preamble = mods.map((m) => `bash ${CONTAINER_MODS_ROOT}/${m}/run.sh && `).join("");
  args.push("bash", "-c", `${preamble}exec ${shellQuote(serveArgv)}`);
  return args;
}
