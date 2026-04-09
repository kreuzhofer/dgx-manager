import WebSocket from "ws";
import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { collectMetrics } from "./metrics.js";
import { discoverRecipes } from "./recipes.js";
import { launchRecipe, stopRecipe, checkDeployments, forceStopVllm, isVllmContainerRunning, isStopping, getTrackedDeployments, reattachLogs, generateLocalModelRecipe } from "./runtime/vllm.js";
import { deployModel as ollamaDeployModel, stopModel as ollamaStopModel, checkOllamaHealth, getOllamaModels } from "./runtime/ollama.js";
import { discoverTrainingRecipes } from "./training-recipes.js";
import { startFinetuneJob, stopFinetuneJob, mergeLoraAdapter } from "./runtime/finetune.js";

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
const ollamaLastState = new Map<string, string>(); // deploymentId → last reported state
const ollamaLastVram = new Map<string, number>(); // deploymentId → last reported vramActual
const vllmLastVram = new Map<string, number>(); // deploymentId → last reported vramActual

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

    // Discover and report available training recipes
    const trainingRecipes = discoverTrainingRecipes();
    if (trainingRecipes.length > 0) {
      ws!.send(JSON.stringify({
        type: "agent:training-recipes",
        payload: { recipes: trainingRecipes },
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
          } else if (status.containerRunning) {
            // Report vramActual for running vLLM containers
            const m = await collectMetrics();
            if (m.vramUsed > 0) {
              const prevVram = vllmLastVram.get(status.deploymentId);
              const changed = !prevVram || Math.abs(m.vramUsed - prevVram) > prevVram * 0.01;
              if (changed) {
                vllmLastVram.set(status.deploymentId, m.vramUsed);
                sendMsg("agent:deployment:status", {
                  deploymentId: status.deploymentId,
                  status: "running",
                  port: status.port,
                  vramActual: m.vramUsed,
                });
              }
            }
            if (status.error) {
              sendMsg("agent:deployment:log", {
                deploymentId: status.deploymentId,
                log: `[HEALTH] ${status.error}\n`,
              });
            }
          } else if (status.error) {
            sendMsg("agent:deployment:log", {
              deploymentId: status.deploymentId,
              log: `[HEALTH] ${status.error}\n`,
            });
          }
        }

        // Report all Ollama loaded models — server matches to deployments
        try {
          const { isOllamaRunning: ollamaUp } = await import("./runtime/ollama.js");
          if (await ollamaUp()) {
            const ps = await (await fetch("http://localhost:11434/api/ps")).json() as {
              models?: { name: string; size: number }[];
            };
            const loadedModels = (ps.models || []).map((m) => ({
              name: m.name,
              vramMB: Math.round(m.size / 1024 / 1024),
            }));
            sendMsg("agent:ollama-status", { models: loadedModels });
          }
        } catch { /* ollama not running */ }

        // Check tracked Ollama deployments for eviction
        const { getActiveDeployments: getOllamaDeployments } = await import("./runtime/ollama.js");
        for (const [depId, modelName] of getOllamaDeployments()) {
          const health = await checkOllamaHealth(depId);
          if (!health) continue;
          const prev = ollamaLastState.get(depId);
          if (!health.loaded && prev !== "evicted") {
            ollamaLastState.set(depId, "evicted");
            sendMsg("agent:deployment:status", {
              deploymentId: depId,
              status: "evicted",
              vramActual: 0,
              error: `Model ${modelName} was unloaded from GPU memory`,
            });
          } else if (health.loaded && (prev === "evicted" || !prev)) {
            ollamaLastState.set(depId, "running");
            sendMsg("agent:deployment:status", {
              deploymentId: depId,
              status: "running",
              port: 11434,
              vramActual: health.vramUsed,
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
            if (status !== "running") {
              sendMsg("agent:deployment:status", { deploymentId, status, error });
            }
            // "running" is sent below with vramActual from the return value
          },
          modelType
        ).then((result) => {
          sendMsg("agent:deployment:status", {
            deploymentId,
            status: "running",
            port: result.port,
            vramActual: result.vramActual,
          });
        }).catch((err) => {
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
      const { deploymentId, deleteAfter, clusterNodes, runtime, modelName: undeployModelName } = msg.payload as {
        deploymentId: string; deleteAfter?: boolean; clusterNodes?: string[]; runtime?: string; modelName?: string;
      };
      sendMsg("agent:deployment:status", { deploymentId, status: "stopping" });

      // Stop asynchronously so we can report progress
      (async () => {
        try {
          if (runtime === "ollama") {
            await ollamaStopModel(deploymentId, undeployModelName);
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

    case "cmd:finetune:start": {
      const { jobId, recipeFile, dataset, outputDir, config } = msg.payload as {
        jobId: string;
        recipeFile: string;
        dataset: string;
        outputDir: string;
        config?: Record<string, unknown>;
      };

      console.log(`[finetune] Starting job ${jobId} with recipe ${recipeFile}`);

      startFinetuneJob(jobId, recipeFile, dataset, outputDir, config || {}, {
        onLog: (line) => {
          sendMsg("agent:finetune:progress", {
            jobId,
            log: line,
          });
        },
        onProgress: (phase, phaseProgress, extra) => {
          sendMsg("agent:finetune:progress", {
            jobId,
            phase,
            phaseProgress,
            step: extra?.step,
            totalSteps: extra?.totalSteps,
            loss: extra?.loss,
            etaSeconds: extra?.etaSeconds,
          });
        },
        onComplete: (status, outputPath, error) => {
          console.log(`[finetune] Job ${jobId} ${status}${error ? `: ${error}` : ""}`);
          sendMsg("agent:finetune:complete", {
            jobId,
            status,
            outputPath: outputPath ?? null,
            error: error ?? undefined,
          });
        },
      });
      break;
    }

    case "cmd:finetune:stop":
    case "cmd:finetune:cancel": {
      const { jobId } = msg.payload as { jobId: string };
      console.log(`[finetune] Stopping job ${jobId}`);
      const stopped = stopFinetuneJob(jobId);
      if (stopped) {
        sendMsg("agent:finetune:complete", {
          jobId,
          status: "stopped",
        });
      }
      break;
    }

    case "cmd:finetune:merge": {
      const { jobId, baseModel, adapterPath, mergedOutputDir } = msg.payload as {
        jobId: string; baseModel: string; adapterPath: string; mergedOutputDir: string;
      };

      console.log(`[finetune] Merging job ${jobId}: ${baseModel} + ${adapterPath}`);
      mergeLoraAdapter(jobId, baseModel, adapterPath, mergedOutputDir, {
        onLog: (line) => {
          sendMsg("agent:finetune:merge-progress", { jobId, log: line });
        },
        onProgress: (phase, phaseProgress) => {
          sendMsg("agent:finetune:merge-progress", { jobId, phase, phaseProgress });
        },
        onComplete: (status, outputPath, error) => {
          console.log(`[finetune] Merge ${jobId} ${status}${error ? `: ${error}` : ""}`);
          sendMsg("agent:finetune:merge-complete", {
            jobId, status, mergedPath: outputPath ?? null, error: error ?? undefined,
          });
        },
      });
      break;
    }

    case "cmd:finetune:deploy": {
      const { jobId, deploymentId, modelPath, config } = msg.payload as {
        jobId: string; deploymentId: string; modelPath: string; config?: Record<string, unknown>;
      };

      console.log(`[finetune] Deploying merged model from ${modelPath}`);
      try {
        const recipeFile = generateLocalModelRecipe({
          jobId,
          modelPath,
          port: (config?.port as number) ?? 8000,
          gpuMemoryUtilization: (config?.gpuMem as number) ?? 0.85,
          maxModelLen: (config?.maxModelLen as number) ?? 4096,
        });

        sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
        let lastPhase = "starting";
        launchRecipe(
          deploymentId,
          recipeFile,
          { port: (config?.port as number) ?? 8000 },
          (line) => {
            sendMsg("agent:deployment:log", { deploymentId, log: line });
            const phase = detectPhase(line);
            if (phase && phase !== lastPhase) {
              lastPhase = phase;
              sendMsg("agent:deployment:status", { deploymentId, status: phase });
            }
          },
          (code) => {
            if (code === 0 && isVllmContainerRunning()) {
              console.log(`[finetune] Deploy run-recipe.sh exited 0, container running`);
            } else if (!isVllmContainerRunning()) {
              sendMsg("agent:deployment:status", {
                deploymentId, status: "failed",
                error: code === 0 ? "Container not running after launch" : `Launch failed with exit code ${code}`,
              });
            }
          }
        );
      } catch (err) {
        sendMsg("agent:deployment:status", {
          deploymentId, status: "failed", error: String(err),
        });
      }
      break;
    }

    default:
      console.log(`Unknown command: ${msg.type}`);
  }
}

connect();
