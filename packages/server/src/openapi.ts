import swaggerJsdoc from "swagger-jsdoc";
import swaggerUi from "swagger-ui-express";
import type { Express } from "express";

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
- **Benchmarks** — server-side benchmark runs (llama-benchy / tool-eval-bench) against a deployment.
- **Datasets / Models / Settings / Tokens / Agent bundle** — supporting resources.

Typical flow: register a Node -> it provisions (incl. sparkrun) -> pick a Recipe -> create a
Deployment -> route traffic through the Load Balancer -> optionally Benchmark it.
`;

export function buildOpenApiSpec() {
  return swaggerJsdoc({
    definition: {
      openapi: "3.0.3",
      info: { title: "DGX Manager API", version: "1.0.0", description: SYSTEM_OVERVIEW },
      tags: [
        { name: "Nodes", description: "Register, audit, and provision DGX Spark nodes." },
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
      ],
    },
    apis: ["packages/server/src/routes/*.ts", "packages/server/src/routes/*.js"],
  });
}

/** Mount GET /api/openapi.json (spec) and GET /api/docs (Swagger UI). */
export function mountOpenApi(app: Express) {
  const spec = buildOpenApiSpec();
  app.get("/api/openapi.json", (_req, res) => res.json(spec));
  app.use("/api/docs", swaggerUi.serve, swaggerUi.setup(spec));
}
