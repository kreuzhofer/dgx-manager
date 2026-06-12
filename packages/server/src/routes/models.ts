import { Router } from "express";
import { prisma } from "../prisma.js";

export const modelsRouter = Router();

/**
 * @openapi
 * /api/models:
 *   get:
 *     tags: [Models]
 *     summary: List all model records
 *     description: >
 *       Returns every Model row ordered by creation date descending, each including
 *       its associated Deployment records. Models are created automatically when a
 *       deployment is launched (vLLM or Ollama) or when a fine-tune job is deployed.
 *       The `name` field is the recipe key (e.g. `llama3-8b-instruct`) for vLLM,
 *       the Ollama tag for Ollama models, and a stable id-derived name for fine-tunes.
 *     responses:
 *       '200':
 *         description: Array of model objects with deployments included
 */
modelsRouter.get("/", async (_req, res) => {
  const models = await prisma.model.findMany({
    orderBy: { createdAt: "desc" },
    include: { deployments: true },
  });
  res.json(models);
});

/**
 * @openapi
 * /api/models:
 *   post:
 *     tags: [Models]
 *     summary: Create a model record manually
 *     description: >
 *       Creates a Model row explicitly. Normally models are auto-created by the
 *       deployment and fine-tune routes. Use this endpoint to pre-register a model
 *       that will be referenced by a deployment, or to create a model entry for
 *       tracking purposes. The `runtime` field determines which inference engine
 *       will serve it: `vllm` or `ollama`.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             required: [name, runtime]
 *             properties:
 *               name: { type: string, description: "Unique model identifier / recipe key." }
 *               runtime: { type: string, enum: [vllm, ollama], description: "Inference engine." }
 *               parameters: { type: string, description: "Optional parameter count label (e.g. '8B')." }
 *     responses:
 *       '201':
 *         description: Created model record
 *       '400':
 *         description: name and runtime required
 */
modelsRouter.post("/", async (req, res) => {
  const { name, runtime, parameters } = req.body;
  if (!name || !runtime) {
    return res.status(400).json({ error: "name and runtime required" });
  }
  const model = await prisma.model.create({
    data: { name, runtime, parameters },
  });
  res.status(201).json(model);
});

/**
 * @openapi
 * /api/models/{id}:
 *   delete:
 *     tags: [Models]
 *     summary: Delete a model record
 *     description: >
 *       Removes the Model row. Does not stop any running deployments that reference
 *       this model — callers should stop deployments first. Used by the dashboard
 *       cleanup flow to remove stale model entries after fine-tune jobs are deleted.
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ deleted: true }'
 */
modelsRouter.delete("/:id", async (req, res) => {
  await prisma.model.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});
