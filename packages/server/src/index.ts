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
import { tokensRouter } from "./routes/tokens.js";
import { settingsRouter } from "./routes/settings.js";
import { ollamaCatalogRouter } from "./routes/ollama-catalog.js";
import { agentBundleRouter } from "./routes/agent-bundle.js";
import { datasetsRouter } from "./routes/datasets.js";
import { benchmarksRouter } from "./routes/benchmarks.js";
import { hfCacheRouter } from "./routes/hf-cache.js";
import { clusterRouter } from "./routes/cluster.js";
import { registriesRouter } from "./routes/registries.js";
import { seedDefaultRegistries } from "./registries/seed.js";
import { mountOpenApi } from "./openapi.js";
import { prisma } from "./prisma.js";
import { sseHandler } from "./sse.js";
import { startMetricRetention } from "./metric-retention.js";
import { sshExec } from "./ssh/executor.js";
import { sendMagicPacket } from "./nodes/wol.js";
import { reconcileStaleRuns } from "./benchmarks/boot-reconcile.js";

const app = express();
const server = createServer(app);

app.use(cors());
app.use(express.json());

// WebSocket hubs
const agentHub = new AgentHub();
agentHub.start();
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
app.set("capClient", agentHub.capClient);
app.set("dashboardHub", dashboardHub);
app.set("sshExec", sshExec);
app.set("wolSend", sendMagicPacket);

// REST API routes
app.use("/api/nodes", nodesRouter);
app.use("/api/models", modelsRouter);
app.use("/api/deployments", deploymentsRouter);
app.use("/api/finetune", finetuneRouter);
app.use("/api/lb", loadbalancerRouter);
app.use("/api/recipes", recipesRouter);
app.use("/api/training-recipes", trainingRecipesRouter);
app.use("/api/tokens", tokensRouter);
app.use("/api/settings", settingsRouter);
app.use("/api/ollama-catalog", ollamaCatalogRouter);
app.use("/api/agent", agentBundleRouter);
app.use("/api/datasets", datasetsRouter);
app.use("/api/benchmarks", benchmarksRouter);
app.use("/api/hf-cache", hfCacheRouter);
app.use("/api/cluster", clusterRouter);
app.use("/api/registries", registriesRouter);

// OpenAPI spec + Swagger UI
mountOpenApi(app);

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

async function main() {
  // Reconcile benchmark runs across a restart. Legacy local runs are failed;
  // remote systemd jobs on the eval node are resumed. Fire-and-forget so an
  // offline eval agent can't block startup. See benchmarks/boot-reconcile.ts.
  void reconcileStaleRuns((nodeId, name, input) => agentHub.capClient.invoke(nodeId, name, input));

  try {
    const seeded = await seedDefaultRegistries(prisma);
    if (seeded > 0) console.log(`Seeded ${seeded} default sparkrun registries`);
  } catch (err) {
    console.error("Skipping registry seed (table not ready — run `npm run db:push`):", err);
  }

  const retentionDays = Number(process.env.METRIC_RETENTION_DAYS ?? 7);
  startMetricRetention({
    retentionDays,
    intervalMs: 60 * 60 * 1000,
  });

  server.listen(PORT, () => {
    console.log(`DGX Manager server listening on port ${PORT}`);
    console.log(`Agent WebSocket:     ws://localhost:${PORT}/ws/agent`);
    console.log(`Dashboard WebSocket: ws://localhost:${PORT}/ws/dashboard`);
  });
}

main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
