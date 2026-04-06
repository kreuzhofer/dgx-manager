import WebSocket from "ws";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { collectMetrics } from "./metrics.js";
import { discoverRecipes } from "./recipes.js";
import { launchRecipe, stopRecipe } from "./runtime/vllm.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const AGENT_VERSION: string = JSON.parse(
  readFileSync(join(__dirname, "../package.json"), "utf-8")
).version;

const MANAGER_URL = process.env.MANAGER_URL || "ws://localhost:4000/ws/agent";
const NODE_ID = process.env.NODE_ID || "unknown";
const METRICS_INTERVAL = 5_000;
const RECONNECT_BASE = 1_000;
const RECONNECT_MAX = 30_000;

let ws: WebSocket | null = null;
let reconnectDelay = RECONNECT_BASE;
let metricsTimer: ReturnType<typeof setInterval> | null = null;

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

    // Start metrics loop
    if (metricsTimer) clearInterval(metricsTimer);
    metricsTimer = setInterval(async () => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      const m = await collectMetrics();
      ws.send(JSON.stringify({
        type: "agent:metrics",
        payload: {
          gpuUtil: m.gpuUtil,
          vramUsed: m.vramUsed,
          tps: null,
          activeRequests: null,
          temp: m.temperature,
        },
      }));
    }, METRICS_INTERVAL);
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
      sendMsg("agent:deployment:status", { deploymentId, status: "stopped" });
      break;
    }

    default:
      console.log(`Unknown command: ${msg.type}`);
  }
}

connect();
