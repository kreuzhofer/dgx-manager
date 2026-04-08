import WebSocket from "ws";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { collectMetrics } from "./metrics.js";
import { discoverRecipes } from "./recipes.js";
import { launchRecipe, stopRecipe, checkDeployments, forceStopVllm, isVllmContainerRunning, isStopping, getTrackedDeployments, reattachLogs } from "./runtime/vllm.js";
import { deployModel as ollamaDeployModel, stopModel as ollamaStopModel, checkOllamaHealth, getOllamaModels } from "./runtime/ollama.js";

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

    // Reconcile tracked deployments after restart
    const tracked = getTrackedDeployments();
    if (tracked.length > 0) {
      console.log(`Reconciling ${tracked.length} tracked deployment(s)`);
      const containerUp = isVllmContainerRunning();
      for (const t of tracked) {
        sendMsg("agent:deployment:status", {
          deploymentId: t.deploymentId,
          status: containerUp ? "running" : "failed",
          port: t.port,
          error: containerUp ? undefined : "Container not running after agent restart",
        });
        // Reattach to docker logs for live streaming
        if (containerUp) {
          reattachLogs(t.deploymentId, (line) => {
            sendMsg("agent:deployment:log", { deploymentId: t.deploymentId, log: line });
          });
        }
      }
    } else if (isVllmContainerRunning()) {
      // Container running but no tracked deployment — likely a Ray worker
      // started by another node's head agent. Do NOT stop it automatically
      // as it may be part of an active cluster deployment.
      console.log("Found vLLM container with no local tracking — may be a cluster worker, leaving it running");
    }

    // Discover and report available vLLM recipes
    const recipes = discoverRecipes();
    if (recipes.length > 0) {
      ws!.send(JSON.stringify({
        type: "agent:recipes",
        payload: { recipes },
      }));
    }

    // Report available Ollama models
    const ollamaModels = getOllamaModels();
    if (ollamaModels.length > 0) {
      ws!.send(JSON.stringify({
        type: "agent:ollama-models",
        payload: { models: ollamaModels },
      }));
    }

    // Start metrics loop — includes vLLM stats when available
    // Only start if not already running (survives reconnects)
    if (metricsTimer) clearInterval(metricsTimer);
    metricsTimer = setInterval(async () => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      const m = await collectMetrics();

      // Enrich with vLLM deployment metrics
      let activeRequests: number | null = null;
      let tps: number | null = null;
      try {
        const statuses = await checkDeployments();
        const active = statuses.filter((s) => s.containerRunning);
        if (active.length > 0) {
          activeRequests = active.reduce((sum, s) => sum + (s.requestsRunning ?? 0) + (s.requestsWaiting ?? 0), 0);
          tps = active.reduce((sum, s) => sum + (s.tps ?? 0), 0) || null;
        }
      } catch { /* ignore */ }

      ws.send(JSON.stringify({
        type: "agent:metrics",
        payload: {
          gpuUtil: m.gpuUtil,
          vramUsed: m.vramUsed,
          tps,
          activeRequests,
          temp: m.temperature,
          netInterfaces: m.netInterfaces,
          rdmaInterfaces: m.rdmaInterfaces,
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

        // Check Ollama deployments for eviction
        const { getActiveDeployments: getOllamaDeployments } = await import("./runtime/ollama.js");
        for (const [depId, modelName] of getOllamaDeployments()) {
          const health = await checkOllamaHealth(depId);
          if (health && !health.loaded) {
            sendMsg("agent:deployment:status", {
              deploymentId: depId,
              status: "evicted",
              error: `Model ${modelName} was unloaded from GPU memory by Ollama`,
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
    // Keep metrics and health timers running — they check ws.readyState
    // before sending and will resume reporting once reconnected.
    // Do NOT clear timers here — that stops deployment monitoring.
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, RECONNECT_MAX);
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err.message);
    ws?.close();
  });
}

function detectPhase(line: string): string | null {
  const l = line.toLowerCase();
  if (l.includes("building") || l.includes("=== building")) return "building";
  if (l.includes("copying") && l.includes("image to")) return "building";
  if (l.includes("downloading model") || l.includes("=== downloading")) return "downloading";
  if (l.includes("fetching") && l.includes("files")) return "downloading";
  if (l.includes("starting head node") || l.includes("applying mod")) return "launching";
  if (l.includes("starting ray") || l.includes("ray worker")) return "launching";
  if (l.includes("starting worker node")) return "launching";
  if (l.includes("loading safetensors") || l.includes("loading model")) return "loading";
  if (l.includes("application startup complete")) return "running";
  return null;
}

function sendMsg(type: string, payload: Record<string, unknown>) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type, payload }));
  }
}

function handleCommand(msg: { type: string; payload: Record<string, unknown> }) {
  switch (msg.type) {
    case "cmd:deploy": {
      const { deploymentId, recipeFile, config, clusterNodes, runtime, modelName, modelType } = msg.payload as {
        deploymentId: string;
        recipeFile?: string;
        config?: Record<string, unknown>;
        clusterNodes?: string[];
        runtime?: string;
        modelName?: string;
        modelType?: "chat" | "embedding";
      };

      // Ollama deployment
      if (runtime === "ollama" && modelName) {
        sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
        ollamaDeployModel(
          deploymentId,
          modelName,
          (line) => sendMsg("agent:deployment:log", { deploymentId, log: line }),
          (status, error) => {
            sendMsg("agent:deployment:status", {
              deploymentId,
              status,
              port: status === "running" ? 11434 : undefined,
              error,
            });
          },
          modelType
        ).catch((err) => {
          sendMsg("agent:deployment:status", {
            deploymentId,
            status: "failed",
            error: String(err),
          });
        });
        break;
      }

      // vLLM deployment
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
        let lastPhase = "starting";
        const port = launchRecipe(
          deploymentId,
          recipeFile,
          {
            port: (config?.port as number) ?? 8000,
            gpuMem: config?.gpuMem as number,
            maxModelLen: config?.maxModelLen as number,
            tensorParallel: config?.tensorParallel as number,
            pipelineParallel: config?.pipelineParallel as number,
            clusterNodes,
          },
          (line) => {
            sendMsg("agent:deployment:log", { deploymentId, log: line });

            // Detect deployment phase from log output
            const phase = detectPhase(line);
            if (phase && phase !== lastPhase) {
              lastPhase = phase;
              sendMsg("agent:deployment:status", {
                deploymentId,
                status: phase,
                port: phase === "running" ? (config?.port as number) ?? 8000 : undefined,
              });
            }
          },
          (code) => {
            // If we're being undeployed, don't report — the undeploy handler owns status
            if (isStopping(deploymentId)) {
              console.log(`[deploy] run-recipe.sh exited ${code} during undeploy, ignoring`);
              return;
            }
            // run-recipe.sh exits after launching the docker container.
            // Code 0 = container launched successfully (still running).
            // Code != 0 = setup/launch failed.
            if (code === 0) {
              // Container should be running — verify and keep status as running
              if (isVllmContainerRunning()) {
                console.log(`[deploy] run-recipe.sh exited 0, container still running`);
              } else {
                sendMsg("agent:deployment:status", {
                  deploymentId,
                  status: "failed",
                  error: "Container not running after launch script exited",
                });
              }
            } else {
              // Check if container started despite script error (e.g. download warning)
              if (isVllmContainerRunning()) {
                console.log(`[deploy] run-recipe.sh exited ${code}, but container is running`);
                sendMsg("agent:deployment:status", { deploymentId, status: "running", port });
              } else {
                sendMsg("agent:deployment:status", {
                  deploymentId,
                  status: "failed",
                  error: `Launch failed with exit code ${code}`,
                });
              }
            }
          }
        );
        // Status updates are driven by log phase detection, not sent here
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
      const { deploymentId, deleteAfter, clusterNodes, runtime } = msg.payload as {
        deploymentId: string; deleteAfter?: boolean; clusterNodes?: string[]; runtime?: string;
      };
      sendMsg("agent:deployment:status", { deploymentId, status: "stopping" });

      // Stop asynchronously so we can report progress
      (async () => {
        try {
          if (runtime === "ollama") {
            await ollamaStopModel(deploymentId);
            sendMsg("agent:deployment:status", {
              deploymentId,
              status: "stopped",
              deleteAfter: deleteAfter || false,
            });
            return;
          }

          stopRecipe(deploymentId, clusterNodes);
          forceStopVllm(clusterNodes);

          // Wait for container to actually stop
          let retries = 10;
          while (retries > 0 && isVllmContainerRunning()) {
            await new Promise((r) => setTimeout(r, 2000));
            retries--;
          }

          if (isVllmContainerRunning()) {
            sendMsg("agent:deployment:status", {
              deploymentId,
              status: "failed",
              error: "Container did not stop within timeout",
            });
          } else {
            sendMsg("agent:deployment:status", {
              deploymentId,
              status: "stopped",
              deleteAfter: deleteAfter || false,
            });
          }
        } catch (err) {
          sendMsg("agent:deployment:status", {
            deploymentId,
            status: "failed",
            error: `Stop failed: ${err}`,
          });
        }
      })();
      break;
    }

    default:
      console.log(`Unknown command: ${msg.type}`);
  }
}

connect();
