import { Router } from "express";
import type { AgentHub } from "../ws/agent-hub.js";

export const recipesRouter = Router();

recipesRouter.get("/", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getRecipes());
});

recipesRouter.get("/ollama-models", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getOllamaModels());
});

// Trigger every connected agent to re-scan its local recipe directories
// (vLLM + training) and re-send the lists. Without this, the only way to
// pick up new recipes added to the NFS share is to restart the agent.
recipesRouter.post("/refresh", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  const nodeIds = agentHub.getConnectedNodeIds();
  for (const nodeId of nodeIds) {
    agentHub.sendToAgent(nodeId, { type: "cmd:rescan-recipes", payload: {} });
  }
  res.json({ refreshed: nodeIds.length });
});
