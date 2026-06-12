export interface SparkrunLaunchOptions {
  recipeRef: string;            // registry name | @spark-arena/id | URL | absolute path
  hosts: string[];              // head first; length drives --tp by default
  tp?: number;                  // explicit tensor-parallel override
  pp?: number;
  port?: number;
  gpuMem?: number;
  maxModelLen?: number;
  servedModelName?: string;
  options?: Record<string, string | number>; // -o key=value passthrough
}

/** Build the argv for `uvx --from <pkg> sparkrun <argv>`. Pure + deterministic. */
export function buildSparkrunArgs(o: SparkrunLaunchOptions): string[] {
  const args: string[] = ["run", o.recipeRef, "--no-follow"];
  if (o.hosts.length > 1) args.push("-H", o.hosts.join(","));
  const tp = o.tp ?? o.hosts.length;
  args.push("--tp", String(tp));
  if (o.pp != null) args.push("--pp", String(o.pp));
  if (o.port != null) args.push("--port", String(o.port));
  if (o.gpuMem != null) args.push("--gpu-mem", String(o.gpuMem));
  if (o.maxModelLen != null) args.push("--max-model-len", String(o.maxModelLen));
  if (o.servedModelName) args.push("--served-model-name", o.servedModelName);
  for (const [k, v] of Object.entries(o.options ?? {})) {
    args.push("-o", `${k}=${v}`);
  }
  return args;
}
