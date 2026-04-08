import "./env.js";
import express from "express";
import cors from "cors";
import { createServer } from "http";
import { AgentHub } from "./ws/agent-hub.js";
import { DashboardHub } from "./ws/dashboard-hub.js";
import { nodesRouter } from "./routes/nodes.js";
import { modelsRouter } from "./routes/models.js";
import { deploymentsRouter } from "./routes/deployments.js";
import { finetuneRouter } from "./routes/finetune.js";
import { loadbalancerRouter } from "./routes/loadbalancer.js";
import { recipesRouter } from "./routes/recipes.js";
import { trainingRecipesRouter } from "./routes/training-recipes.js";
import { sseHandler } from "./sse.js";

const app = express();
const server = createServer(app);

app.use(cors());
app.use(express.json());

// WebSocket hubs
const agentHub = new AgentHub();
const dashboardHub = new DashboardHub();

server.on("upgrade", (request, socket, head) => {
  const { pathname } = new URL(request.url!, `http://${request.headers.host}`);
  if (pathname === "/ws/agent") {
    agentHub.handleUpgrade(request, socket, head);
  } else if (pathname === "/ws/dashboard") {
    dashboardHub.handleUpgrade(request, socket, head);
  } else {
    socket.destroy();
  }
});

// Make hubs available to routes
app.set("agentHub", agentHub);
app.set("dashboardHub", dashboardHub);

// REST API routes
app.use("/api/nodes", nodesRouter);
app.use("/api/models", modelsRouter);
app.use("/api/deployments", deploymentsRouter);
app.use("/api/finetune", finetuneRouter);
app.use("/api/lb", loadbalancerRouter);
app.use("/api/recipes", recipesRouter);
app.use("/api/training-recipes", trainingRecipesRouter);

// Broadcast recipe updates to dashboard
agentHub.setRecipesHandler((recipes) => {
  dashboardHub.broadcast("update:recipes", recipes);
});
agentHub.setTrainingRecipesHandler((recipes) => {
  dashboardHub.broadcast("update:training-recipes", recipes);
});

// SSE endpoint for real-time dashboard updates
app.get("/api/events", sseHandler);

// Health check
app.get("/api/health", (_req, res) => {
  res.json({ status: "ok" });
});

const PORT = process.env.PORT || 4000;

server.listen(PORT, () => {
  console.log(`DGX Manager server listening on port ${PORT}`);
  console.log(`Agent WebSocket:     ws://localhost:${PORT}/ws/agent`);
  console.log(`Dashboard WebSocket: ws://localhost:${PORT}/ws/dashboard`);
});
