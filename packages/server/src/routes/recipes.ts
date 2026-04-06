import { Router } from "express";
import type { AgentHub } from "../ws/agent-hub.js";

export const recipesRouter = Router();

recipesRouter.get("/", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getRecipes());
});
