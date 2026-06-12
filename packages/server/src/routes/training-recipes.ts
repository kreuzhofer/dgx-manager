import { Router } from "express";
import type { AgentHub } from "../ws/agent-hub.js";

export const trainingRecipesRouter = Router();

/**
 * @openapi
 * /api/training-recipes:
 *   get:
 *     tags: [Fine-tune]
 *     summary: List available fine-tuning training recipes
 *     description: >
 *       Returns the cached list of training recipes collected from connected agents.
 *       Each recipe specifies the base model, LoRA method, dataset format, training
 *       hyperparameters, deploy configuration (container, gpu_memory_utilization),
 *       and optional custom scripts (merge, quantize_fp8). The `file` field is the
 *       registry reference used as `recipeFile` in POST /api/finetune. The list is
 *       populated when an agent sends its recipe scan (same `cmd:rescan-recipes`
 *       that refreshes inference recipes).
 *     responses:
 *       '200':
 *         description: Array of training recipe objects
 */
trainingRecipesRouter.get("/", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getTrainingRecipes());
});
