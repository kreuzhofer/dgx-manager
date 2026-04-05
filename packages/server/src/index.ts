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

const app = express();
const server = createServer(app);

app.use(cors());
app.use(express.json());

// WebSocket hubs
const agentHub = new AgentHub(server);
const dashboardHub = new DashboardHub(server);

// Make hubs available to routes
app.set("agentHub", agentHub);
app.set("dashboardHub", dashboardHub);

// REST API routes
app.use("/api/nodes", nodesRouter);
app.use("/api/models", modelsRouter);
app.use("/api/deployments", deploymentsRouter);
app.use("/api/finetune", finetuneRouter);
app.use("/api/lb", loadbalancerRouter);

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
