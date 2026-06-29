import type { PrismaClient } from "../generated/prisma/client.js";

export const DEFAULT_REGISTRIES = [
  { name: "sparkrun-testing", url: "https://github.com/dbotwinick/sparkrun-recipe-registry.git", subpath: "testing/recipes",
    description: "Sparkrun testing registry for recipes, tuning configs, and benchmark profiles", visible: false,
    tuningSubpath: "testing/tuning", benchmarkSubpath: "testing/benchmarking", modsSubpath: null, sortOrder: 0 },
  { name: "sparkrun-transitional", url: "https://github.com/dbotwinick/sparkrun-recipe-registry.git", subpath: "transitional/recipes",
    description: "Transitional registry for recipes", visible: true,
    tuningSubpath: "testing/tuning", benchmarkSubpath: null, modsSubpath: null, sortOrder: 1 },
  { name: "official", url: "https://github.com/spark-arena/recipe-registry.git", subpath: "official-recipes",
    description: "Official Spark Arena registry for recipes, tuning configs, and benchmark profiles", visible: true,
    tuningSubpath: "tuning", benchmarkSubpath: "benchmarking", modsSubpath: "official-mods", sortOrder: 2 },
  { name: "experimental", url: "https://github.com/spark-arena/recipe-registry.git", subpath: "experimental-recipes",
    description: "Spark Arena registry for experimental recipes", visible: false,
    tuningSubpath: null, benchmarkSubpath: null, modsSubpath: "experimental-mods", sortOrder: 3 },
  { name: "community", url: "https://github.com/spark-arena/community-recipe-registry.git", subpath: "recipes",
    description: "Community registry for sparkrun", visible: false,
    tuningSubpath: "tuning", benchmarkSubpath: "benchmarking", modsSubpath: null, sortOrder: 4 },
  { name: "eugr", url: "https://github.com/eugr/spark-vllm-docker", subpath: "recipes",
    description: "Official eugr/spark-vllm-docker repo recipes", visible: true,
    tuningSubpath: null, benchmarkSubpath: null, modsSubpath: "mods", sortOrder: 5 },
  { name: "atlas", url: "https://github.com/Avarok-Cybersecurity/atlas-recipes.git", subpath: "recipes",
    description: "Atlas recipes", visible: false,
    tuningSubpath: null, benchmarkSubpath: null, modsSubpath: null, sortOrder: 6 },
] as const;

/** Insert the standard registries only when the table is empty. Returns rows inserted. */
export async function seedDefaultRegistries(prisma: PrismaClient): Promise<number> {
  const count = await prisma.sparkrunRegistry.count();
  if (count > 0) return 0;
  await prisma.sparkrunRegistry.createMany({ data: DEFAULT_REGISTRIES.map((r) => ({ ...r })) });
  return DEFAULT_REGISTRIES.length;
}
