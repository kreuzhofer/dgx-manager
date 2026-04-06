import WebSocket from "ws";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { collectMetrics } from "./metrics.js";
import { discoverRecipes } from "./recipes.js";
import { launchRecipe, stopRecipe, checkDeployments, forceStopVllm, isVllmContainerRunning } from "./runtime/vllm.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const AGENT_VERSION: string = JSON.parse(
  readFileSync(join(__dirname, "../package.json"), "utf-8")
).version;

const MANAGER_URL = process.env.MANAGER_URL || "ws://localhost:4000/ws/agent";
const NODE_ID = process.env.NODE_ID || "unknown";
const METRICS_INTERVAL = 5_000;
const HEALTH_CHECK_INTERVAL = 15_000;
const RECONNECT_BASE = 1_000;
const RECONNECT_MAX = 30_000;

let ws: WebSocket | null = null;
let reconnectDelay = RECONNECT_BASE;
let metricsTimer: ReturnType<typeof setInterval> | null = null;
let healthTimer: ReturnType<typeof setInterval> | null = null;

function connect() {
  console.log(`Connecting to ${MANAGER_URL}...`);
  ws = new WebSocket(MANAGER_URL, { perMessageDeflate: false });

  ws.on("open", async () => {
    console.log("Connected to manager");
    reconnectDelay = RECONNECT_BASE;

    // Register
    const metrics = await collectMetrics();
    ws!.send(JSON.stringify({
      type: "agent:register",
      payload: {
        nodeId: NODE_ID,
        hostname: process.env.HOSTNAME || "unknown",
        gpuModel: metrics.gpuModel,
        vramTotal: metrics.vramTotal,
        agentVersion: AGENT_VERSION,
      },
    }));

    // Discover and report available vLLM recipes
    const recipes = discoverRecipes();
    if (recipes.length > 0) {
      ws!.send(JSON.stringify({
        type: "agent:recipes",
        payload: { recipes },
      }));
    }

    // Start metrics loop — includes vLLM stats when available
    if (metricsTimer) clearInterval(metricsTimer);
    metricsTimer = setInterval(async () => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      const m = await collectMetrics();

      // Enrich with vLLM deployment metrics
      let activeRequests: number | null = null;
      try {
        const statuses = await checkDeployments();
        const running = statuses.filter((s) => s.containerRunning);
        if (running.length > 0) {
          activeRequests = running.reduce((sum, s) => sum + (s.requestsRunning ?? 0) + (s.requestsWaiting ?? 0), 0);
        }
      } catch { /* ignore */ }

      ws.send(JSON.stringify({
        type: "agent:metrics",
        payload: {
          gpuUtil: m.gpuUtil,
          vramUsed: m.vramUsed,
          tps: null,
          activeRequests,
          temp: m.temperature,
        },
      }));
    }, METRICS_INTERVAL);

    // Start deployment health check loop
    if (healthTimer) clearInterval(healthTimer);
    healthTimer = setInterval(async () => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      try {
        const statuses = await checkDeployments();
        for (const status of statuses) {
          // Report if container died or has errors
          if (!status.containerRunning && !status.alive) {
            sendMsg("agent:deployment:status", {
              deploymentId: status.deploymentId,
              status: "failed",
              error: status.error || "Container stopped unexpectedly",
            });
          } else if (status.error) {
            sendMsg("agent:deployment:log", {
              deploymentId: status.deploymentId,
              log: `[HEALTH] ${status.error}\n`,
            });
          }
        }
      } catch { /* ignore */ }
    }, HEALTH_CHECK_INTERVAL);
  });

  ws.on("message", (data) => {
    try {
      const msg = JSON.parse(data.toString());
      console.log(`Received command: ${msg.type}`);
      handleCommand(msg);
    } catch (err) {
      console.error("Message parse error:", err);
    }
  });

  ws.on("close", () => {
    console.log(`Disconnected. Reconnecting in ${reconnectDelay}ms...`);
    if (metricsTimer) clearInterval(metricsTimer);
    if (healthTimer) clearInterval(healthTimer);
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX);
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err.message);
    ws?.close();
  });
}

function sendMsg(type: string, payload: Record<string, unknown>) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type, payload }));
  }
}

function handleCommand(msg: { type: string; payload: Record<string, unknown> }) {
  switch (msg.type) {
    case "cmd:deploy": {
      const { deploymentId, recipeFile, config } = msg.payload as {
        deploymentId: string;
        recipeFile?: string;
        config?: Record<string, unknown>;
      };
      if (!recipeFile) {
        sendMsg("agent:deployment:status", {
          deploymentId,
          status: "failed",
          error: "No recipeFile specified",
        });
        return;
      }
      try {
        sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
        const port = launchRecipe(
          deploymentId,
          recipeFile,
          {
            port: (config?.port as number) ?? 8000,
            gpuMem: config?.gpuMem as number,
            maxModelLen: config?.maxModelLen as number,
          },
          (line) => {
            // Stream logs back
            sendMsg("agent:deployment:log", { deploymentId, log: line });
          },
          (code) => {
            sendMsg("agent:deployment:status", {
              deploymentId,
              status: code === 0 ? "stopped" : "failed",
              error: code !== 0 ? `Process exited with code ${code}` : undefined,
            });
          }
        );
        sendMsg("agent:deployment:status", { deploymentId, status: "running", port });
      } catch (err) {
        sendMsg("agent:deployment:status", {
          deploymentId,
          status: "failed",
          error: String(err),
        });
      }
      break;
    }

    case "cmd:undeploy": {
      const { deploymentId } = msg.payload as { deploymentId: string };
      stopRecipe(deploymentId);
      // Always try to stop any vLLM containers (handles orphaned containers after agent restart)
      forceStopVllm();
      sendMsg("agent:deployment:status", { deploymentId, status: "stopped" });
      break;
    }

    default:
      console.log(`Unknown command: ${msg.type}`);
  }
}

connect();
