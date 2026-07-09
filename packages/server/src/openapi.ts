import swaggerJsdoc from "swagger-jsdoc";
import swaggerUi from "swagger-ui-express";
import type { Express } from "express";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const SYSTEM_OVERVIEW = `
# DGX Manager API

Full-stack control plane for a DGX Spark cluster. Domains and how they relate:

- **Nodes** — register/provision DGX Spark nodes over SSH; agents on each node report GPU metrics.
- **Recipes** — inference recipes discovered from sparkrun registries; the catalog the deploy form uses.
- **Deployments** — launch a recipe on one or more nodes via **sparkrun** (the head-node agent runs
  \`sparkrun run\`). Accepts a registry recipe (\`recipeFile\`), an NFS path (\`recipePath\`), or an
  **inline recipe body** (\`recipeYaml\`) for remote recipe development. Stopped via DELETE.
- **Load Balancer** — round-robin inference proxy routing to running deployments.
- **Fine-tune** — training jobs (train -> merge -> quantize -> deploy the merged model via sparkrun).
- **Benchmarks** — server-side benchmark runs (llama-benchy / tool-eval-bench / lm-eval) against a deployment.
- **Datasets / Models / Settings / Tokens / Agent bundle** — supporting resources.

Typical flow: register a Node -> it provisions (incl. sparkrun) -> pick a Recipe -> create a
Deployment -> route traffic through the Load Balancer -> optionally Benchmark it.
`;

// Build an absolute glob so swagger-jsdoc resolves correctly both at runtime
// (CWD = repo root inside Docker) and in vitest (CWD = repo root).
// Using import.meta.url gives us the directory of *this* compiled file, from
// which we can reliably reach the routes directory regardless of CWD.
const __dirname = dirname(fileURLToPath(import.meta.url));
const ROUTES_GLOB = join(__dirname, "routes", "*.ts");
// In production (compiled to dist/) the source *.ts files are gone; fall back
// to *.js siblings. swagger-jsdoc reads the raw text for JSDoc parsing, so
// either extension works as long as the file exists.
const ROUTES_GLOB_JS = join(__dirname, "routes", "*.js");

export function buildOpenApiSpec() {
  return swaggerJsdoc({
    definition: {
      openapi: "3.0.3",
      info: { title: "DGX Manager API", version: "1.0.0", description: SYSTEM_OVERVIEW },
      tags: [
        { name: "Nodes", description: "Register, audit, and provision DGX Spark nodes." },
        { name: "Cluster", description: "Cluster-wide operations (e.g. reseed cross-node SSH known_hosts trust)." },
        { name: "Recipes", description: "Inference recipe catalog from sparkrun registries." },
        { name: "Deployments", description: "Launch/stop inference workloads via sparkrun (registry ref, NFS path, or inline YAML)." },
        { name: "Load Balancer", description: "Round-robin inference proxy over running deployments." },
        { name: "Fine-tune", description: "Training jobs and deploying fine-tuned models." },
        { name: "Benchmarks", description: "Server-side benchmark runs against deployments." },
        { name: "Datasets", description: "Training/eval dataset management." },
        { name: "Models", description: "Model records." },
        { name: "Settings", description: "Server settings." },
        { name: "Tokens", description: "HF / API tokens." },
        { name: "Agent bundle", description: "Per-arch agent tarballs + install script." },
        { name: "HF Cache", description: "Hugging Face cache inventory, scan, and guarded delete." },
      ],
    },
    apis: [ROUTES_GLOB, ROUTES_GLOB_JS],
  });
}

/** Mount GET /api/openapi.json (spec) and GET /api/docs (Swagger UI). */
export function mountOpenApi(app: Express) {
  const spec = buildOpenApiSpec();
  app.get("/api/openapi.json", (_req, res) => res.json(spec));
  app.use("/api/docs", swaggerUi.serve, swaggerUi.setup(spec));
}
