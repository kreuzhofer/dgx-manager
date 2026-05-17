import WebSocket from "ws";
import { readFileSync, writeFileSync, existsSync } from "fs";
import { execSync, spawn } from "child_process";
import { hostname as osHostname } from "os";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { collectMetrics } from "./metrics.js";
import { discoverRecipes } from "./recipes.js";
import { launchRecipe, stopRecipe, checkDeployments, forceStopVllm, isVllmContainerRunning, isStopping, getTrackedDeployments, reattachLogs, generateLocalModelRecipe, isLaunchInProgress } from "./runtime/vllm.js";
import { removeDeployment } from "./runtime/deployment-store.js";
import { deployModel as ollamaDeployModel, stopModel as ollamaStopModel, checkOllamaHealth } from "./runtime/ollama.js";
import { discoverTrainingRecipes } from "./training-recipes.js";
import { startFinetuneJob, stopFinetuneJob, mergeLoraAdapter, reattachFinetuneJobs } from "./runtime/finetune.js";
import { quantizeMergedToFp8 } from "./runtime/finetune-quantize.js";
import { selfAudit } from "./self-audit.js";

const __dirname = dirname(fileURLToPath(import.meta.url));
const AGENT_DIR = join(__dirname, "..");
const AGENT_VERSION: string = JSON.parse(
  readFileSync(join(AGENT_DIR, "package.json"), "utf-8")
).version;
const AGENT_ARCH: string =
  process.arch === "x64" ? "amd64" : process.arch === "arm64" ? "arm64" : process.arch;

const MANAGER_URL = process.env.MANAGER_URL || "ws://localhost:4000/ws/agent";
const JOIN_TOKEN = process.env.JOIN_TOKEN || "";
const NODE_ID_FILE = join(AGENT_DIR, "node-id");
// Subnets in priority order (left wins) considered as the "fast" inter-node
// fabric. Override at deploy time with FAST_NET_SUBNETS=10.0.0.0/8,...
const FAST_NET_SUBNETS = (process.env.FAST_NET_SUBNETS || "192.168.100.0/24")
  .split(",").map((s) => s.trim()).filter(Boolean);

// Absolute path to the fine-tune-recipes repo on the shared NFS mount. Used
// to resolve a deploy-time `recipeFile` (the relative recipe directory the
// training job ran out of) into an absolute path so the deploy step can
// pick up that recipe's `inference.yaml`/`inference.j2` overrides.
const FINETUNE_RECIPES_REPO = process.env.FINETUNE_RECIPES_REPO
  || `${process.env.SHARED_STORAGE || "/mnt/tank"}/src/github/dgx-manager-fine-tune-recipes`;

/**
 * Detect this node's IP on the fast inter-node fabric (e.g. 192.168.100.x),
 * if any. Returns null if no interface matches the configured subnets.
 * Used by the server to route bulk node↔node transfers (image sync, model
 * copy) over the fast network instead of the management network.
 */
function detectFastIp(): string | null {
  try {
    const out = execSync("ip -4 -o addr show", { timeout: 3000, encoding: "utf-8" });
    // Parse lines like: `5: enp1s0    inet 192.168.100.42/24 brd ...`
    const candidates: string[] = [];
    for (const line of out.split("\n")) {
      const m = line.match(/inet\s+(\d+\.\d+\.\d+\.\d+)\/(\d+)/);
      if (!m) continue;
      candidates.push(m[1]);
    }
    for (const subnet of FAST_NET_SUBNETS) {
      const [base, maskStr] = subnet.split("/");
      const mask = parseInt(maskStr, 10);
      const baseInt = ipToInt(base);
      const maskInt = mask === 0 ? 0 : (~0 << (32 - mask)) >>> 0;
      const subnetVal = (baseInt & maskInt) >>> 0;
      for (const ip of candidates) {
        if (((ipToInt(ip) & maskInt) >>> 0) === subnetVal) return ip;
      }
    }
  } catch { /* fall through */ }
  return null;
}

function ipToInt(ip: string): number {
  return ip.split(".").reduce((acc, oct) => ((acc << 8) + parseInt(oct, 10)) >>> 0, 0);
}
const METRICS_INTERVAL = 5_000;
const HEALTH_CHECK_INTERVAL = 15_000;
const RECONNECT_BASE = 1_000;
const RECONNECT_MAX = 30_000;

/**
 * Resolve the node ID from (in priority order):
 * 1. Persisted node-id file (from previous token registration)
 * 2. NODE_ID env var
 * 3. Empty string (will use JOIN_TOKEN flow)
 */
function resolveNodeId(): string {
  // Check persisted file first
  if (existsSync(NODE_ID_FILE)) {
    const id = readFileSync(NODE_ID_FILE, "utf-8").trim();
    if (id) return id;
  }
  // Fall back to env var
  const envId = process.env.NODE_ID || "";
  if (envId && envId !== "unknown") return envId;
  return "";
}

let nodeId = resolveNodeId();
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

    // Register — use token flow if no nodeId persisted
    const metrics = await collectMetrics();
    const fastIpAddress = detectFastIp();
    if (fastIpAddress) {
      console.log(`Detected fast-fabric IP: ${fastIpAddress}`);
    }
    if (nodeId) {
      ws!.send(JSON.stringify({
        type: "agent:register",
        payload: {
          nodeId,
          hostname: osHostname() || "unknown",
          gpuModel: metrics.gpuModel,
          vramTotal: metrics.vramTotal,
          agentVersion: AGENT_VERSION,
          arch: AGENT_ARCH,
          fastIpAddress,
        },
      }));
    } else if (JOIN_TOKEN) {
      console.log("Registering with join token...");
      ws!.send(JSON.stringify({
        type: "agent:register-token",
        payload: {
          token: JOIN_TOKEN,
          hostname: osHostname() || "unknown",
          gpuModel: metrics.gpuModel,
          vramTotal: metrics.vramTotal,
          agentVersion: AGENT_VERSION,
          arch: AGENT_ARCH,
          fastIpAddress,
        },
      }));
      // Wait for register:accepted before continuing setup
      return;
    } else {
      console.error("No NODE_ID or JOIN_TOKEN configured. Cannot register.");
      ws!.close();
      return;
    }

    postRegistrationSetup();
  });

  /** Setup tasks that run after successful registration (either nodeId or token flow). */
  function postRegistrationSetup() {
    // Reconcile tracked deployments after restart.
    //
    // Important: this runs on every WS reconnect, not just after a full agent
    // process restart. A WS-only reconnect (e.g. server container redeploy)
    // leaves the in-process `running` map intact — the launch subprocess and
    // its log forwarder are still alive. In that case we must NOT report
    // "failed" just because no docker container exists yet; the subprocess
    // could simply still be downloading or building. Reporting "failed" in
    // that window flips the deployment to failed in the DB while bytes are
    // still streaming to disk.
    //
    // Decision tree for each tracked deployment on reconnect:
    //   1. Launch subprocess still alive in this process → in-progress, do
    //      nothing. The existing log/phase stream will drive status.
    //   2. Container running → "running" (post-restart reattach).
    //   3. Container not running AND subprocess dead → "failed".
    const tracked = getTrackedDeployments();
    if (tracked.length > 0) {
      console.log(`Reconciling ${tracked.length} tracked deployment(s)`);
      const containerUp = isVllmContainerRunning();
      for (const t of tracked) {
        if (isLaunchInProgress(t.deploymentId)) {
          console.log(`[reconcile] ${t.deploymentId}: launch subprocess still alive, leaving status to existing log stream`);
          continue;
        }
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

    // Reattach to any running finetune containers (survives agent restart)
    reattachFinetuneJobs(sendMsg);

    // Self-audit: report local prereq status so the dashboard can render the
    // same checklist we'd get from an SSH audit. Runs once per connection.
    try {
      const audit = selfAudit();
      sendMsg("agent:self-audit", { systemInfo: audit.systemInfo, checks: audit.checks });
    } catch (err) {
      console.error("Self-audit failed:", err);
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
          diskDevices: m.diskDevices,
          memory: m.memory,
          pressure: m.pressure,
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
            // Untrack so the next health tick doesn't keep reporting this
            // deployment as failed/running based on a shared vllm_node
            // container. Without this, stale records from prior failed
            // launches caused live containers to be misattributed across
            // multiple deployment IDs.
            removeDeployment(status.deploymentId);
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
        const { getActiveDeployments: getOllamaDeployments, decideOllamaStateTransition } = await import("./runtime/ollama.js");
        for (const [depId, modelName] of getOllamaDeployments()) {
          const health = await checkOllamaHealth(depId);
          if (!health) continue;
          const transition = decideOllamaStateTransition(health.loaded, ollamaLastState.get(depId));
          if (transition === "evicted") {
            ollamaLastState.set(depId, "evicted");
            sendMsg("agent:deployment:status", {
              deploymentId: depId,
              status: "evicted",
              vramActual: 0,
              error: `Model ${modelName} was unloaded from GPU memory`,
            });
          } else if (transition === "running") {
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
  }

  ws.on("message", (data) => {
    try {
      const msg = JSON.parse(data.toString());
      console.log(`Received: ${msg.type}`);

      // Handle registration acceptance (token flow)
      if (msg.type === "register:accepted") {
        nodeId = msg.payload.nodeId;
        console.log(`Registered as node: ${nodeId}`);
        // Persist node ID for future reconnects
        try {
          writeFileSync(NODE_ID_FILE, nodeId, "utf-8");
          console.log(`Node ID persisted to ${NODE_ID_FILE}`);
        } catch (err) {
          console.error(`Failed to persist node ID: ${err}`);
        }
        postRegistrationSetup();
        return;
      }

      // Handle registration rejection
      if (msg.type === "register:rejected") {
        console.error(`Registration rejected: ${msg.payload?.error || "unknown reason"}`);
        ws?.close();
        return;
      }

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

/**
 * Parse huggingface_hub's tqdm progress for the multi-file fetch:
 *   "Fetching 56 files:   2%|▏         | 1/56 [00:00<00:24,  2.21it/s]"
 * Returns null if the line isn't a recognizable progress line.
 *
 * Per-file lines like "model-00001-of-00010.safetensors:  12%|..." also exist
 * but are too granular to surface at the row level — we only parse the
 * aggregate "Fetching N files" line.
 */
const FETCHING_RE = /Fetching\s+(\d+)\s+files:\s+(\d+(?:\.\d+)?)%\|[^|]*\|\s*(\d+)\/(\d+)\s*\[([^<\]]+)<([^,\]]+)/;
function parseFetchingProgress(line: string): {
  percent: number;
  current: number;
  total: number;
  elapsed: string;
  eta: string;
} | null {
  const m = line.match(FETCHING_RE);
  if (!m) return null;
  return {
    percent: parseFloat(m[2]),
    current: parseInt(m[3], 10),
    total: parseInt(m[4], 10),
    elapsed: m[5].trim(),
    eta: m[6].trim(),
  };
}

/**
 * Collapse `\r`-only carriage returns to a sane single-line representation.
 * tqdm rewrites the same line in a terminal using `\r`; when streamed verbatim
 * to a browser pre-block they accumulate as one ever-growing line. Within a
 * single chunk we keep only the FINAL segment between `\n` boundaries so the
 * dashboard sees one updating line, not 50 concatenated ones.
 */
function collapseCarriageReturns(chunk: string): string {
  if (!chunk.includes("\r")) return chunk;
  // Split on \n to preserve real line boundaries, collapse \r within each line
  return chunk.split("\n").map((segment) => {
    if (!segment.includes("\r")) return segment;
    const parts = segment.split("\r").filter((s) => s.length > 0);
    return parts[parts.length - 1] || "";
  }).join("\n");
}

// Per-deployment throttle for progress emissions. tqdm fires many times a
// second; we cap to ~1/sec to keep the WS quiet without losing fidelity.
const lastProgressEmit = new Map<string, number>();
function emitDeploymentProgress(deploymentId: string, progress: ReturnType<typeof parseFetchingProgress>) {
  if (!progress) return;
  const now = Date.now();
  const last = lastProgressEmit.get(deploymentId) || 0;
  // Always emit the first tick, the 100% tick, and at most one per second between
  if (now - last < 1000 && progress.percent < 100) return;
  lastProgressEmit.set(deploymentId, now);
  sendMsg("agent:deployment:progress", {
    deploymentId,
    phase: "downloading",
    phaseProgress: progress.percent,
    current: progress.current,
    total: progress.total,
    elapsed: progress.elapsed,
    eta: progress.eta,
  });
}

function handleCommand(msg: { type: string; payload: Record<string, unknown> }) {
  switch (msg.type) {
    case "cmd:deploy": {
      const { deploymentId, recipeFile, config, clusterNodes, clusterNodeFastIps, runtime, modelName, modelType, servedModelName } = msg.payload as {
        deploymentId: string;
        recipeFile?: string;
        config?: Record<string, unknown>;
        clusterNodes?: string[];
        // Per-cluster-node fast-fabric IP, ordered head-first matching
        // clusterNodes. Element is null when that node didn't report a fast IP.
        clusterNodeFastIps?: (string | null)[];
        runtime?: string;
        modelName?: string;
        modelType?: "chat" | "embedding";
        /** Per-deploy override for vLLM's --served-model-name. */
        servedModelName?: string;
      };

      // Ollama deployment
      if (runtime === "ollama" && modelName) {
        // Reset health-check state machine so the first tick after a
        // restart doesn't see stale "running" from the previous cycle
        // and false-flag the in-progress load as eviction.
        ollamaLastState.delete(deploymentId);
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
          modelType,
          (p) =>
            sendMsg("agent:ollama:pull-progress", {
              deploymentId,
              status: p.status,
              percent: p.percent,
              current: p.current,
              total: p.total,
            }),
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
        // Auto-skip the model-download phase when the recipe's `model:` field
        // is a local path (e.g. `/workspace/outputs/.../merged`). Without
        // this, run-recipe.sh's hf-download step interprets the path as a
        // HuggingFace repo id ("Repo id must be in the form 'repo_name' or
        // 'namespace/repo_name'") and the deploy fails before vLLM starts.
        // cmd:finetune:deploy already does this; cmd:deploy now matches.
        let recipeModelIsLocal = false;
        try {
          const recipeContent = readFileSync(`${process.env.VLLM_REPO_PATH || `${process.env.SHARED_STORAGE || "/mnt/tank"}/src/github/spark-vllm-docker`}/${recipeFile}`, "utf-8");
          const m = recipeContent.match(/^model:\s*(.+)$/m);
          if (m && m[1].trim().startsWith("/")) recipeModelIsLocal = true;
        } catch { /* if we can't read the recipe, fall back to default behavior */ }
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
            clusterNodeFastIps,
            skipSetup: recipeModelIsLocal,
            servedModelName,
          },
          (line) => {
            // Collapse tqdm carriage-return updates so the log viewer doesn't
            // accumulate one massive line per `Fetching ... files:` tick.
            const cleaned = collapseCarriageReturns(line);
            sendMsg("agent:deployment:log", { deploymentId, log: cleaned });

            // Parse aggregate download progress (huggingface_hub multi-file
            // tqdm) and emit a throttled progress event for the UI bar.
            const progress = parseFetchingProgress(line);
            if (progress) emitDeploymentProgress(deploymentId, progress);

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
                removeDeployment(deploymentId);
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
                removeDeployment(deploymentId);
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
        removeDeployment(deploymentId);
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
      const { jobId, recipeFile, dataset, outputDir, config, clusterNodeIps, resumeFromCheckpoint } = msg.payload as {
        jobId: string;
        recipeFile: string;
        dataset: string;
        outputDir: string;
        config?: Record<string, unknown>;
        clusterNodeIps?: string[];
        resumeFromCheckpoint?: boolean;
      };

      console.log(`[finetune] Starting job ${jobId} with recipe ${recipeFile}${clusterNodeIps ? ` (${clusterNodeIps.length} nodes)` : ""}${resumeFromCheckpoint ? " [RESUME]" : ""}`);

      startFinetuneJob(jobId, recipeFile, dataset, outputDir, config || {}, clusterNodeIps, {
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
            lr: extra?.lr,
            evalLoss: extra?.evalLoss,
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
      }, resumeFromCheckpoint);
      break;
    }

    case "cmd:finetune:stop":
    case "cmd:finetune:cancel": {
      const { jobId } = msg.payload as { jobId: string };
      console.log(`[finetune] Stopping job ${jobId}`);
      stopFinetuneJob(jobId);
      // Always confirm: even if there were no containers to remove (already
      // cleaned, stale record, etc.), the job is "stopped" from the server's
      // perspective. Without this, jobs with no live containers get stuck in
      // "stopping" state in the DB forever.
      sendMsg("agent:finetune:complete", { jobId, status: "stopped" });
      break;
    }

    case "cmd:finetune:merge": {
      const { jobId, baseModel, adapterPath, mergedOutputDir, mergeScript } = msg.payload as {
        jobId: string; baseModel: string; adapterPath: string; mergedOutputDir: string; mergeScript?: string;
      };

      console.log(`[finetune] Merging job ${jobId}: ${baseModel} + ${adapterPath} (script=${mergeScript || "scripts/merge.py"})`);
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
      }, mergeScript);
      break;
    }

    case "cmd:finetune:quantize": {
      const { jobId, mergedPath, quantizedOutputDir, quantizeScript } = msg.payload as {
        jobId: string; mergedPath: string; quantizedOutputDir: string; quantizeScript: string;
      };

      console.log(`[finetune] Quantizing job ${jobId}: ${mergedPath} -> ${quantizedOutputDir} (script=${quantizeScript})`);
      quantizeMergedToFp8(jobId, mergedPath, quantizedOutputDir, {
        onLog: (line) => sendMsg("agent:finetune:quantize-progress", { jobId, log: line }),
        onProgress: (phase, phaseProgress) => sendMsg("agent:finetune:quantize-progress", { jobId, phase, phaseProgress }),
        onComplete: (status, outputPath, error) => {
          console.log(`[finetune] Quantize ${jobId} ${status}${error ? `: ${error}` : ""}`);
          sendMsg("agent:finetune:quantize-complete", {
            jobId, status, quantizedPath: outputPath ?? null, error: error ?? undefined,
          });
        },
      }, quantizeScript);
      break;
    }

    case "cmd:finetune:deploy": {
      const {
        jobId, deploymentId, modelPath, deployContainer, config,
        clusterNodes, clusterNodeFastIps, modelName, recipeFile, artifactVariant,
      } = msg.payload as {
        jobId: string;
        deploymentId: string;
        modelPath: string;
        deployContainer?: string;
        config?: Record<string, unknown>;
        clusterNodes?: string[];
        clusterNodeFastIps?: (string | null)[];
        modelName?: string;
        recipeFile?: string;
        artifactVariant?: "bf16" | "fp8";
      };

      const isCluster = Array.isArray(clusterNodes) && clusterNodes.length > 1;
      const port = (config?.port as number) ?? 8000;
      const gpuMem = (config?.gpuMem as number) ?? 0.85;
      const maxModelLen = (config?.maxModelLen as number) ?? 4096;
      const tensorParallel = config?.tensorParallel as number | undefined;
      const pipelineParallel = config?.pipelineParallel as number | undefined;

      console.log(`[finetune] Deploying merged model from ${modelPath} (container: ${deployContainer || "vllm-node"}, cluster: ${isCluster ? clusterNodes!.length + " nodes" : "solo"}${recipeFile ? `, recipe: ${recipeFile}` : ""})`);

      try {
        // Resolve the training recipe's directory on the NFS share so
        // generateLocalModelRecipe can pick up an `inference.yaml` /
        // `inference.j2` override shipped alongside the recipe. Falls back
        // to undefined → fully auto-generated recipe when the deploy
        // payload doesn't carry a recipeFile.
        const recipeDir = recipeFile
          ? join(FINETUNE_RECIPES_REPO, recipeFile)
          : undefined;

        const generatedRecipeFile = generateLocalModelRecipe({
          jobId,
          modelPath,
          container: deployContainer || "vllm-node",
          port,
          gpuMemoryUtilization: gpuMem,
          maxModelLen,
          // For cluster mode, both the recipe YAML's solo_only marker AND
          // the actual launch topology need to be set. The YAML drops
          // solo_only; the command gains `--distributed-executor-backend
          // ray` + a SPREAD placement env. tensorParallel/pipelineParallel
          // are embedded as `defaults` AND substituted into the command
          // template — without this, run-recipe.py silently dropped the
          // `--tp N` CLI override (no `{tensor_parallel}` placeholder to
          // substitute into) and vLLM defaulted to TP=1 even though Ray
          // was correctly spanning the cluster.
          isCluster,
          tensorParallel: tensorParallel ?? 1,
          pipelineParallel: pipelineParallel ?? 1,
          servedModelName: modelName,
          recipeDir,
          artifactVariant: artifactVariant ?? "bf16",
        });

        sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
        let lastPhase = "starting";
        launchRecipe(
          deploymentId,
          generatedRecipeFile,
          {
            port,
            gpuMem,
            maxModelLen,
            tensorParallel,
            pipelineParallel,
            clusterNodes: isCluster ? clusterNodes : undefined,
            clusterNodeFastIps: isCluster ? clusterNodeFastIps : undefined,
            skipSetup: true,
          },
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
              removeDeployment(deploymentId);
            }
          }
        );
      } catch (err) {
        sendMsg("agent:deployment:status", {
          deploymentId, status: "failed", error: String(err),
        });
        removeDeployment(deploymentId);
      }
      break;
    }

    case "cmd:rescan-recipes": {
      // Re-scan local recipe directories on demand. Without this, recipes
      // added to the NFS share after agent startup stay invisible until
      // the agent reconnects.
      try {
        const recipes = discoverRecipes();
        sendMsg("agent:recipes", { recipes });
        const trainingRecipes = discoverTrainingRecipes();
        sendMsg("agent:training-recipes", { recipes: trainingRecipes });
        console.log(`[rescan] vllm=${recipes.length} training=${trainingRecipes.length}`);
      } catch (err) {
        console.error(`[rescan] failed: ${err}`);
      }
      break;
    }

    case "cmd:update": {
      const { bundleUrl, version } = msg.payload as { bundleUrl: string; version: string };
      console.log(`[update] Updating agent to v${version} from ${bundleUrl}`);
      sendMsg("agent:update-status", { status: "downloading", version });

      try {
        // Download bundle
        execSync(`curl -sL -o /tmp/agent-bundle.tar.gz "${bundleUrl}"`, { timeout: 120_000 });

        // /opt/dgx-agent* paths require root. Agent runs as a non-root systemd
        // user, so all writes under /opt must go through sudo (configured
        // NOPASSWD in the install script).
        execSync("sudo rm -rf /opt/dgx-agent-new && sudo mkdir -p /opt/dgx-agent-new", { timeout: 10_000 });
        execSync("sudo tar -xzf /tmp/agent-bundle.tar.gz -C /opt/dgx-agent-new/", { timeout: 30_000 });

        // Preserve node-id file
        if (existsSync(NODE_ID_FILE)) {
          execSync(`sudo cp "${NODE_ID_FILE}" /opt/dgx-agent-new/node-id`, { timeout: 5_000 });
        }

        // Atomic swap
        execSync("sudo rm -rf /opt/dgx-agent-old", { timeout: 5_000 });
        execSync("sudo mv /opt/dgx-agent /opt/dgx-agent-old && sudo mv /opt/dgx-agent-new /opt/dgx-agent", { timeout: 10_000 });
        execSync("rm -f /tmp/agent-bundle.tar.gz", { timeout: 5_000 });

        sendMsg("agent:update-status", { status: "restarting", version });
        console.log(`[update] Agent updated to v${version}, restarting...`);

        // Restart the systemd service (agent will reconnect with new version)
        setTimeout(() => {
          try {
            execSync("sudo systemctl restart dgx-agent", { timeout: 10_000 });
          } catch (err) {
            console.error(`[update] Failed to restart service: ${err}`);
          }
        }, 500);
      } catch (err) {
        console.error(`[update] Update failed: ${err}`);
        sendMsg("agent:update-status", { status: "failed", error: String(err) });
        // Clean up staging
        try { execSync("sudo rm -rf /opt/dgx-agent-new && rm -f /tmp/agent-bundle.tar.gz"); } catch { /* */ }
      }
      break;
    }

    case "cmd:deprovision": {
      console.log("[deprovision] Received deprovision command — uninstalling agent");
      sendMsg("agent:update-status", { status: "deprovisioning", version: AGENT_VERSION });

      // Write a detached cleanup script. It needs to run AFTER this process
      // dies (it's going to `systemctl stop` us), so we spawn it with
      // `detached: true` + `.unref()` so it re-parents to init and survives.
      const script = `#!/bin/bash
set +e
# Wait briefly so the server can persist the delete before we vanish.
sleep 2
sudo systemctl stop dgx-agent
sudo systemctl disable dgx-agent
sudo rm -f /etc/systemd/system/dgx-agent.service
sudo rm -f /etc/sudoers.d/dgx-agent
sudo systemctl daemon-reload
sudo rm -rf /opt/dgx-agent /opt/dgx-agent-old /opt/dgx-agent-new
rm -f /tmp/dgx-deprovision.sh
`;
      try {
        writeFileSync("/tmp/dgx-deprovision.sh", script, { mode: 0o755 });
        // Spawn via `systemd-run` in its own transient unit so the cleanup
        // escapes dgx-agent.service's cgroup. Otherwise `systemctl stop
        // dgx-agent` would kill our bash child (default KillMode=control-group)
        // before it could `disable` the unit and `rm` the files.
        const child = spawn(
          "sudo",
          [
            "systemd-run",
            "--unit=dgx-deprovision",
            "--slice=system.slice",
            "--collect",
            "bash",
            "/tmp/dgx-deprovision.sh",
          ],
          { detached: true, stdio: "ignore" }
        );
        child.unref();
        // Close WS so the server sees us go cleanly.
        ws?.close();
      } catch (err) {
        console.error(`[deprovision] Failed to launch cleanup: ${err}`);
      }
      break;
    }

    default:
      console.log(`Unknown command: ${msg.type}`);
  }
}

connect();
