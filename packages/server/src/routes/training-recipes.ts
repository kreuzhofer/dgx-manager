import { Router } from "express";
import type { AgentHub } from "../ws/agent-hub.js";

export const trainingRecipesRouter = Router();

trainingRecipesRouter.get("/", (req, res) => {
  const agentHub: AgentHub = req.app.get("agentHub");
  res.json(agentHub.getTrainingRecipes());
});
