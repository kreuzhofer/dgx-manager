import { Router } from "express";
import type { AgentHub } from "../ws/agent-hub.js";

export const recipesRouter = Router();

/**
 * @openapi
 * /api/recipes:
 *   get:
 *     tags: [Recipes]
 *     summary: List available sparkrun inference recipes
 *     description: >
 *       Returns the cached list of sparkrun inference recipes collected from all
 *       connected agents (via `sparkrun list`). Each recipe entry includes its
 *       `file` (the registry reference used as `recipeFile` in POST /api/deployments),
 *       `defaults` (gpu_memory_utilization, tensor_parallel, pipeline_parallel, etc.),
 *       and metadata. The list is refreshed whenever an agent reconnects or when
 *       POST /api/recipes/refresh is called. Used by the deploy form to populate
 *       the recipe dropdown.
 *     responses:
 *       '200':
 *         description: Array of recipe objects from the sparkrun registry
 */
recipesRouter.get("/", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getRecipes());
});

/**
 * @openapi
 * /api/recipes/ollama-models:
 *   get:
 *     tags: [Recipes]
 *     summary: List Ollama models available on connected agents
 *     description: >
 *       Returns the list of Ollama models reported by connected agents (from
 *       `ollama list`). Each entry includes `name`, `size`, and `modified`.
 *       Used by the Ollama deploy form to populate the model selector. The list
 *       is refreshed automatically whenever an agent reconnects or sends a metrics
 *       update that includes the Ollama model list.
 *     responses:
 *       '200':
 *         description: Array of Ollama model objects
 */
recipesRouter.get("/ollama-models", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getOllamaModels());
});

/**
 * @openapi
 * /api/recipes/refresh:
 *   post:
 *     tags: [Recipes]
 *     summary: Trigger agents to rescan recipe directories
 *     description: >
 *       Sends `cmd:rescan-recipes` to every connected agent, causing each to rescan
 *       its local recipe directories (sparkrun vLLM recipes + training recipes) and
 *       re-send the updated lists to the server. Without this, new YAML files added
 *       to the NFS share after the agent started won't appear in GET /api/recipes
 *       until the agent restarts. Returns the number of agents that received the
 *       command.
 *     responses:
 *       '200':
 *         description: '{ refreshed: N } — number of agents that received the command'
 */
recipesRouter.post("/refresh", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const nodeIds = agentHub.getConnectedNodeIds();
  for (const nodeId of nodeIds) {
    agentHub.sendToAgent(nodeId, { type: "cmd:rescan-recipes", payload: {} });
  }
  res.json({ refreshed: nodeIds.length });
});
