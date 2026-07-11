import WebSocket from "ws";
import { readFileSync, writeFileSync, existsSync, mkdirSync } from "fs";
import { execSync, spawn } from "child_process";
import { hostname as osHostname, homedir } from "os";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import { collectMetrics } from "./metrics.js";
import { shouldReportStatus } from "./runtime/deploy-report.js";
import { detectPhase, phaseRank } from "./runtime/deploy-phase.js";
import { discoverRecipes, updateRegistries } from "./recipes.js";
import { writeRegistriesFile, type RegistryWire } from "./registries.js";
import { untrackDeployment } from "./runtime/vllm.js";
import { classifyDeadContainer, reconcileDeployStatus } from "./runtime/deploy-status.js";
import { launchSparkrun, stopSparkrun, isWorkloadRunning, writeInlineRecipe, removeInlineRecipe, resolveHfHome } from "./runtime/sparkrun.js";
import { buildInventory, deleteCachedRepo, type RepoKind } from "./runtime/hf-cache.js";
import { checkSparkrunDeployments, sparkrunRunningStatus, parseLoadingShards } from "./runtime/sparkrun-metrics.js";
import { launchDgxrun, stopDgxrun, inspectDgxrunContainerResult } from "./runtime/dgxrun/dgxrun.js";
import { reconcileDgxrunAction } from "./runtime/dgxrun/dgxrun-reconcile.js";
import { deployCancels, launchExitAction } from "./runtime/deploy-cancel.js";
import { checkDgxrunDeployments } from "./runtime/dgxrun/dgxrun-metrics.js";
import type { DgxrunRecipe } from "./runtime/dgxrun/dgxrun-args.js";
import { loadDeployments, saveDeployment } from "./runtime/deployment-store.js";
import { deployModel as ollamaDeployModel, stopModel as ollamaStopModel, checkOllamaHealth } from "./runtime/ollama.js";
import { discoverTrainingRecipes } from "./training-recipes.js";
import { findInferenceTemplate, applyFinetuneSubstitutions, renderSparkrunFinetuneRecipe } from "./runtime/inference-template.js";
import { startFinetuneJob, stopFinetuneJob, mergeLoraAdapter, reattachFinetuneJobs } from "./runtime/finetune.js";
import { quantizeMergedToFp8 } from "./runtime/finetune-quantize.js";
import { selfAudit } from "./self-audit.js";
import { applyOllamaFirewall } from "./firewall.js";
import { powerCommand, powerUnitName, powerLaunchCommand, type PowerAction } from "./runtime/power.js";
import { CapRegistry } from "./caps/registry.js";
import { makeExecCap } from "./caps/exec-cap.js";
import { makeJobCaps } from "./caps/job-cap.js";
import { collectDiag } from "./sysinfo/diag.js";
import { readSysInfo } from "./sysinfo/proc-read.js";
import { launchUpdater } from "./update-launch.js";

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
const SPARKRUN_ADHOC_DIR = join(homedir(), ".dgx-agent", "adhoc");
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
const deployLastStatus = new Map<string, string>(); // deploymentId → last reported deploy status

const caps = new CapRegistry();
caps.register({ name: "diag.collect", handle: async () => collectDiag() });
caps.register(makeExecCap(undefined, (a) => sendMsg("agent:audit", { cap: "exec", ...a })));
// Long-running benchmark jobs, owned by systemd so they outlive agent rolls.
for (const c of makeJobCaps()) caps.register(c);

function connect() {
  console.log(`Connecting to ${MANAGER_URL}...`);
  ws = new WebSocket(MANAGER_URL, { perMessageDeflate: false });

  ws.on("open", async () => {
    console.log("Connected to manager");
    reconnectDelay = RECONNECT_BASE;

    try {
      execSync("mkdir -p /run/dgx-agent && touch /run/dgx-agent/connected");
    } catch { /* marker best-effort */ }
    // A connected agent proves no update is legitimately in flight — every
    // update path (success, rollback, or a killed/crashed updater that never
    // even wrote a result) ends in a restart. Clear the lock unconditionally
    // so a killed updater can never wedge future cmd:update calls forever.
    try { execSync("rm -f /run/dgx-agent/updating"); } catch { /* best-effort stale-lock clear */ }
    // Report the outcome of a just-completed self-update (written by the detached
    // updater), so a rollback/failure is visible instead of silent. Success needs
    // no report — the new version shows up in metrics.
    try {
      const rp = "/run/dgx-agent/update-result.json";
      if (existsSync(rp)) {
        const r = JSON.parse(readFileSync(rp, "utf-8")) as { version: string; outcome: string; error?: string };
        if (r.outcome !== "success") {
          sendMsg("agent:update-status", { status: "failed", version: r.version, error: `${r.outcome}: ${r.error ?? ""}` });
        }
        execSync(`rm -f ${rp}`);
      }
    } catch { /* result report best-effort */ }

    // Register — use token flow if no nodeId persisted
    const metrics = await collectMetrics();
    const fastIpAddress = detectFastIp();
    if (fastIpAddress) {
      console.log(`Detected fast-fabric IP: ${fastIpAddress}`);
    }
    // Explicit management-IP override. Needed when the agent is co-located with
    // the server on a docker bridge (e.g. the manager host is also a node): the
    // server would otherwise record the bridge gateway as this node's IP from
    // the WS source. Set NODE_ADVERTISE_IP=<real NIC IP> on such a node.
    const advertiseIp = process.env.NODE_ADVERTISE_IP || undefined;
    if (advertiseIp) {
      console.log(`Advertising management IP: ${advertiseIp}`);
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
          advertiseIp,
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
          advertiseIp,
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
    // Reconcile sparkrun deployments from the persistent store.
    // On a WS-only reconnect the sparkrun workload may still be running as a
    // separate cluster job — we use check-job liveness (isWorkloadRunning) to
    // determine status rather than tracking a local subprocess.
    const sparkrunDeployments = loadDeployments().filter((d) => d.kind === "sparkrun");
    if (sparkrunDeployments.length > 0) {
      console.log(`Reconciling ${sparkrunDeployments.length} sparkrun deployment(s)`);
      for (const d of sparkrunDeployments) {
        const target = d.clusterId ?? d.recipeFile;
        const hosts = d.clusterNodes ?? [];
        const listed = isWorkloadRunning(target, hosts);
        const status = reconcileDeployStatus({ launcherAlive: false, listed });
        console.log(`[reconcile-sparkrun] ${d.deploymentId}: target=${target} listed=${listed} → ${status}`);
        sendMsg("agent:deployment:status", {
          deploymentId: d.deploymentId,
          status,
          port: d.port,
          error: status === "failed" ? "Workload not running after agent restart" : undefined,
        });
      }
    }

    // Reconcile dgxrun deployments — each agent re-checks ITS OWN rank
    // container via `docker inspect` (no cross-node liveness; the manager
    // aggregates head health). A dead rank reports failed → manager tears the
    // whole mp cluster down.
    const dgxrunDeployments = loadDeployments().filter((d) => d.kind === "dgxrun");
    if (dgxrunDeployments.length > 0) {
      console.log(`Reconciling ${dgxrunDeployments.length} dgxrun deployment(s)`);
      for (const d of dgxrunDeployments) {
        const rank = d.rank ?? 0;
        // NB: inspectDgxrunContainerResult, not isDgxrunRunning — the latter
        // collapses an inconclusive `docker inspect` into "not running", and one
        // `failed` rank tears down the whole cluster. An agent roll is exactly
        // when the daemon is busiest. See runtime/dgxrun/dgxrun-reconcile.ts.
        const action = reconcileDgxrunAction(inspectDgxrunContainerResult(d.deploymentId), {
          rank, port: d.port,
        });
        console.log(`[reconcile-dgxrun] ${d.deploymentId}: rank=${rank} → ${action.kind}`);
        if (action.kind === "skip") {
          console.warn(`[reconcile-dgxrun] ${d.deploymentId}: ${action.reason}; leaving to the health loop`);
          continue;
        }
        if (action.kind === "phase") {
          // Head is running but may still be loading/compiling. On a WS reconnect
          // reportPhase drops this "starting" if we've already advanced past it,
          // so the dashboard doesn't snap back to "starting" mid-load.
          reportPhase(d.deploymentId, "starting");
          continue;
        }
        sendMsg("agent:deployment:status", {
          deploymentId: d.deploymentId,
          status: action.status,
          port: action.port,
          error: action.error,
        });
      }
    }

    // Reattach to any running finetune containers (survives agent restart)
    reattachFinetuneJobs(sendMsg);

    // Self-audit: report local prereq status so the dashboard can render the
    // same checklist we'd get from an SSH audit. Runs once per connection;
    // also re-sent when the fire-and-forget firewall apply finishes (below).
    sendSelfAudit();

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

      // Enrich with vLLM/sparkrun deployment metrics
      let activeRequests: number | null = null;
      let tps: number | null = null;
      try {
        const statuses = await checkSparkrunDeployments();
        const active = statuses.filter((s) => s.containerRunning);
        if (active.length > 0) {
          activeRequests = active.reduce((sum, s) => sum + (s.requestsRunning ?? 0) + (s.requestsWaiting ?? 0), 0);
          tps = active.reduce((sum, s) => sum + (s.tps ?? 0), 0) || null;
        }
        // Fold in dgxrun deployments too — otherwise a dgxrun (mp) model reports
        // no throughput. Only the head rank scrapes /metrics (workers return
        // null tps), so its tps/active-requests belong in this node's metrics.
        const dgx = (await checkDgxrunDeployments()).filter((s) => s.containerRunning);
        if (dgx.length > 0) {
          activeRequests = (activeRequests ?? 0) +
            dgx.reduce((sum, s) => sum + (s.requestsRunning ?? 0) + (s.requestsWaiting ?? 0), 0);
          const dgxTps = dgx.reduce((sum, s) => sum + (s.tps ?? 0), 0);
          if (dgxTps > 0) tps = (tps ?? 0) + dgxTps;
        }
      } catch { /* ignore */ }

      ws.send(JSON.stringify({
        type: "agent:metrics",
        payload: {
          gpuUtil: m.gpuUtil,
          vramUsed: m.vramUsed,
          // Carried on every tick so a stale/zero register-time vramTotal
          // self-heals (the metrics path has the GB10 system-RAM fallback).
          vramTotal: m.vramTotal,
          tps,
          activeRequests,
          temp: m.temperature,
          netInterfaces: m.netInterfaces,
          rdmaInterfaces: m.rdmaInterfaces,
          diskDevices: m.diskDevices,
          memory: m.memory,
          pressure: m.pressure,
          sysinfo: readSysInfo(),
        },
      }));
    }, METRICS_INTERVAL);

    // Start deployment health check loop
    if (healthTimer) clearInterval(healthTimer);
    healthTimer = setInterval(async () => {
      if (ws?.readyState !== WebSocket.OPEN) return;
      try {
        const statuses = await checkSparkrunDeployments();
        for (const status of statuses) {
          // Report if container died or has errors.
          // Intentional stops (cmd:undeploy set stopping===true in the store)
          // are already excluded by checkSparkrunDeployments, so reaching here
          // means the workload died on its own — treat as a crash.
          if (!status.containerRunning && !status.alive) {
            // When we have captured container output (crash-loop or clean exit
            // with logs), emit it to the deployment logstream so the real error
            // (e.g. vllm serve argument errors) is visible via GET /api/deployments/:id/logs.
            if (status.capturedLog || status.crashLoop) {
              const head = status.crashLoop
                ? `[agent] container is crash-looping (restart #${status.restartCount}) — capturing container output:`
                : `[agent] container exited — capturing container output:`;
              sendMsg("agent:deployment:log", { deploymentId: status.deploymentId, log: `${head}\n${status.capturedLog ?? ""}\n` });
            }
            sendMsg("agent:deployment:status", {
              deploymentId: status.deploymentId,
              status: "failed",
              error: status.error ?? classifyDeadContainer(false, status.error).error,
            });
            // Stop the actual container to cancel the unless-stopped restart loop,
            // then untrack so the next health tick doesn't re-report this deployment.
            const d = loadDeployments().find((x) => x.deploymentId === status.deploymentId);
            if (d) {
              try { stopSparkrun(d.deploymentId, d.clusterId ?? d.recipeFile, d.clusterNodes ?? [], d.tp); } catch { /* best effort */ }
            } else {
              untrackDeployment(status.deploymentId);
            }
            deployLastStatus.delete(status.deploymentId);
          } else if (status.containerRunning) {
            // Report status for vLLM containers: "running" only once the API is
            // ready (apiReady===true), "starting" while shards are still loading.
            const deployStatus = sparkrunRunningStatus(status);
            const m = await collectMetrics();
            const s = deployStatus;
            if (shouldReportStatus({ lastStatus: deployLastStatus.get(status.deploymentId), status: s, lastVram: vllmLastVram.get(status.deploymentId), vramUsed: m.vramUsed })) {
              deployLastStatus.set(status.deploymentId, s);
              if (m.vramUsed > 0) vllmLastVram.set(status.deploymentId, m.vramUsed);
              sendMsg("agent:deployment:status", {
                deploymentId: status.deploymentId,
                status: s,
                // Only advertise the port once the API server is actually bound.
                port: s === "running" ? status.port : undefined,
                ...(m.vramUsed > 0 ? { vramActual: m.vramUsed } : {}),
              });
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

        // dgxrun health: each agent reports ONLY its own rank container. Head
        // (rank 0) drives running/starting; workers stay quiet unless they die
        // (a dead rank hangs the whole mp cluster → manager coordinates teardown).
        try {
          const dgxStatuses = await checkDgxrunDeployments();
          for (const status of dgxStatuses) {
            const isHead = (status.rank ?? 0) === 0;
            if (!status.containerRunning && !status.alive) {
              if (status.capturedLog || status.crashLoop) {
                const head = status.crashLoop
                  ? `[agent] dgxrun rank ${status.rank ?? 0} crash-looping (restart #${status.restartCount}) — container output:`
                  : `[agent] dgxrun rank ${status.rank ?? 0} exited — container output:`;
                sendMsg("agent:deployment:log", { deploymentId: status.deploymentId, log: `${head}\n${status.capturedLog ?? ""}\n` });
              }
              // Report failure from ANY rank so the manager tears down all ranks.
              sendMsg("agent:deployment:status", {
                deploymentId: status.deploymentId,
                status: "failed",
                error: status.error ?? `dgxrun rank ${status.rank ?? 0} died`,
              });
              // Stop the local container (cancel restart loop) + untrack.
              try { stopDgxrun(status.deploymentId); } catch { /* best effort */ }
              deployLastStatus.delete(status.deploymentId);
            } else if (status.containerRunning && isHead) {
              const deployStatus = sparkrunRunningStatus(status);
              const m = await collectMetrics();
              const s = deployStatus;
              if (shouldReportStatus({ lastStatus: deployLastStatus.get(status.deploymentId), status: s, lastVram: vllmLastVram.get(status.deploymentId), vramUsed: m.vramUsed })) {
                deployLastStatus.set(status.deploymentId, s);
                if (m.vramUsed > 0) vllmLastVram.set(status.deploymentId, m.vramUsed);
                sendMsg("agent:deployment:status", {
                  deploymentId: status.deploymentId,
                  status: s,
                  port: s === "running" ? status.port : undefined,
                  ...(m.vramUsed > 0 ? { vramActual: m.vramUsed } : {}),
                });
              }
            }
            // Running workers stay silent — the head's status is the sole gate.
          }
        } catch { /* ignore */ }

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

      void handleCommand(msg).catch((err) => console.error("handleCommand error:", err));
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

// Monotonic per-deployment phase reporting: the dashboard status only ever moves
// FORWARD through the lifecycle (see runtime/deploy-phase). A lower-ranked match
// arriving late — the post-load "Prefetching checkpoint files" line (reads as
// download) or a reconnect re-emitting "starting" — is dropped rather than
// regressing the status, which is what read as "downloading while compiling".
const deployPhaseRank = new Map<string, number>();
function reportPhase(deploymentId: string, phase: string, extra: Record<string, unknown> = {}): void {
  const r = phaseRank(phase);
  if (r >= 0) {
    if (r <= (deployPhaseRank.get(deploymentId) ?? -1)) return; // not forward — ignore
    deployPhaseRank.set(deploymentId, r);
  }
  sendMsg("agent:deployment:status", { deploymentId, status: phase, ...extra });
}
function clearPhaseTracking(deploymentId: string): void {
  deployPhaseRank.delete(deploymentId);
}

function sendMsg(type: string, payload: Record<string, unknown>) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type, payload }));
  }
}

/**
 * Report local prereq status (incl. the Ollama firewall state) so the
 * dashboard can render the same checklist as an SSH audit. Safe to call any
 * time: sendMsg no-ops when the socket isn't open. Sent once per WS connect,
 * and again when the fire-and-forget firewall apply settles — so a stable
 * agent that never reconnects still clears the transient "in progress".
 */
function sendSelfAudit() {
  try {
    const audit = selfAudit();
    sendMsg("agent:self-audit", { systemInfo: audit.systemInfo, checks: audit.checks });
  } catch (err) {
    console.error("Self-audit failed:", err);
  }
}

/** Scan HF_HOME and push the inventory. `error` carries a preceding command
 *  failure (e.g. a failed delete) so it surfaces in the dashboard instead of
 *  vanishing. A scan failure itself (unmounted HF_HOME) is also reported as
 *  an inventory with `error` — never silently dropped. */
function sendHfCacheInventory(error?: string) {
  const hfHome = resolveHfHome();
  try {
    const inventory = buildInventory(hfHome);
    sendMsg("agent:hf-cache", { ...inventory, ...(error ? { error } : {}) });
  } catch (err) {
    // cacheId "" → the server falls back to a per-node group for error rows
    sendMsg("agent:hf-cache", {
      cacheId: "",
      hfHome,
      scannedAt: new Date().toISOString(),
      totalBytes: 0,
      diskFreeBytes: 0,
      repos: [],
      error: error ? `${error}; scan also failed: ${err}` : `scan failed: ${err}`,
    });
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

type ProgressData = { percent: number; current: number; total: number; elapsed?: string; eta?: string };

/**
 * Emit a throttled `agent:deployment:progress` message for either the
 * "downloading" (HuggingFace fetch) or "loading" (vLLM shard load) phase.
 * Emits at most once per second, but always lets the first and 100% ticks through.
 */
function emitDeploymentProgress(
  deploymentId: string,
  phase: "downloading" | "loading",
  progress: ProgressData | null,
) {
  if (!progress) return;
  const now = Date.now();
  const last = lastProgressEmit.get(deploymentId) || 0;
  // Always emit the first tick, the 100% tick, and at most one per second between
  if (now - last < 1000 && progress.percent < 100) return;
  lastProgressEmit.set(deploymentId, now);
  sendMsg("agent:deployment:progress", {
    deploymentId,
    phase,
    phaseProgress: progress.percent,
    current: progress.current,
    total: progress.total,
    ...(progress.elapsed !== undefined && { elapsed: progress.elapsed }),
    ...(progress.eta !== undefined && { eta: progress.eta }),
  });
}

async function handleCommand(msg: { type: string; payload: Record<string, unknown> }) {
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

      // dgxrun deployment — our own mp multi-node runner. The manager fans a
      // cmd:deploy to EACH cluster node with its rank; this node launches only
      // its own rank's container locally (no SSH, no cross-node orchestration).
      const dgxrunPayload = msg.payload as {
        kind?: string;
        recipe?: DgxrunRecipe;
        rank?: number;
        nnodes?: number;
        masterAddr?: string;
        masterPort?: number;
        headless?: boolean;
        params?: Record<string, string | number | undefined>;
      };
      if (dgxrunPayload.kind === "dgxrun") {
        const { recipe, rank, nnodes, masterAddr, masterPort, params } = dgxrunPayload;
        if (!recipe || rank == null || !nnodes || !masterAddr || masterPort == null) {
          sendMsg("agent:deployment:status", {
            deploymentId, status: "failed",
            error: "dgxrun deploy missing recipe/rank/nnodes/masterAddr/masterPort",
          });
          break;
        }
        // A fresh launch supersedes any cancel left over from a previous stop of
        // this id (restart reuses the id). Must precede launchDgxrun.
        deployCancels.beginDeploy(deploymentId);
        // Only the head drives lifecycle status; workers are silent unless they fail.
        if (rank === 0) reportPhase(deploymentId, "starting");
        // Port override arrives via params (server folds config into params for
        // dgxrun); fall back to the recipe's default port.
        const deployPort = Number(params?.port ?? recipe.defaults?.port ?? 8000);
        try {
          launchDgxrun(
            deploymentId,
            { recipe, rank, nnodes, masterAddr, masterPort, port: deployPort, params },
            (line) => {
              const cleaned = collapseCarriageReturns(line);
              sendMsg("agent:deployment:log", { deploymentId, log: cleaned });
              const loadProgress = parseLoadingShards(line);
              if (loadProgress && rank === 0) emitDeploymentProgress(deploymentId, "loading", loadProgress);
              const phase = detectPhase(line);
              // Forward-only: reportPhase drops a late lower-ranked match (e.g. the
              // post-load "Prefetching checkpoint files" line) instead of regressing.
              if (phase && rank === 0) {
                reportPhase(deploymentId, phase, phase === "running" ? { port: deployPort } : {});
              }
            },
            (code) => {
              // `docker run -d` has exited, so the container provably exists (or
              // provably failed). This is the only safe point to observe a stop
              // that raced the launch: cmd:undeploy's own `docker rm -f` may have
              // run while the container was still being created, removing nothing
              // and leaving it orphaned (2026-07-09). See runtime/deploy-cancel.ts.
              const action = launchExitAction({
                code, rank, cancel: deployCancels.pendingCancel(deploymentId),
              });
              if (action.kind === "running") return; // health loop confirms serving
              if (action.kind === "failed") {
                // The manager treats a dead rank as a coordinated teardown trigger.
                sendMsg("agent:deployment:status", {
                  deploymentId, status: "failed", error: action.error,
                });
                return;
              }
              sendMsg("agent:deployment:log", {
                deploymentId,
                log: `\n=== Stop raced the launch — tearing down rank ${rank} ===\n`,
              });
              try { stopDgxrun(deploymentId); }
              catch (e) { console.warn(`[deploy] cancel teardown error (continuing): ${e}`); }
              deployCancels.forget(deploymentId);
              sendMsg("agent:deployment:status", {
                deploymentId, status: "stopped", deleteAfter: action.deleteAfter,
              });
            },
          );
        } catch (err) {
          sendMsg("agent:deployment:status", { deploymentId, status: "failed", error: String(err) });
        }
        break;
      }

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

      // sparkrun deployment — triggered by inlineRecipeYaml or an explicit recipeRef field
      const { inlineRecipeYaml, recipeRef: payloadRecipeRef, displayName: payloadDisplayName } = msg.payload as {
        inlineRecipeYaml?: string;
        recipeRef?: string;
        displayName?: string;
      };
      if (inlineRecipeYaml != null || payloadRecipeRef != null) {
        try {
          sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
          let lastPhase = "starting";

          // Resolve the recipe reference: inline YAML wins, then explicit recipeRef
          const recipeRef = inlineRecipeYaml != null
            ? writeInlineRecipe(deploymentId, inlineRecipeYaml, SPARKRUN_ADHOC_DIR)
            : payloadRecipeRef!;

          // Build hosts: clusterNodes head-first when provided, else local solo
          const hosts: string[] = (Array.isArray(clusterNodes) && clusterNodes.length > 0)
            ? clusterNodes as string[]
            : ["localhost"];

          const opts = {
            hosts,
            tp: config?.tensorParallel as number | undefined,
            pp: config?.pipelineParallel as number | undefined,
            port: (config?.port as number) ?? 8000,
            gpuMem: config?.gpuMem as number | undefined,
            maxModelLen: config?.maxModelLen as number | undefined,
            servedModelName: servedModelName ?? undefined,
            // For inline deployments recipeRef is a temp file path — use the
            // server-sent ref, display name, or deployment id as the human-
            // readable name instead. recipeRef (the temp path) is still passed
            // as the first positional arg to launchSparkrun below.
            recipeName: payloadRecipeRef ?? payloadDisplayName ?? deploymentId,
          };

          launchSparkrun(
            deploymentId,
            recipeRef,
            opts,
            (line) => {
              const cleaned = collapseCarriageReturns(line);
              sendMsg("agent:deployment:log", { deploymentId, log: cleaned });

              const fetchProgress = parseFetchingProgress(line);
              if (fetchProgress) emitDeploymentProgress(deploymentId, "downloading", fetchProgress);

              const loadProgress = parseLoadingShards(line);
              if (loadProgress) emitDeploymentProgress(deploymentId, "loading", loadProgress);

              const phase = detectPhase(line);
              if (phase && phase !== lastPhase) {
                lastPhase = phase;
                sendMsg("agent:deployment:status", {
                  deploymentId,
                  status: phase,
                  port: phase === "running" ? opts.port : undefined,
                });
              }
            },
            (code) => {
              // sparkrun run exits 0 once the workload is launched (--no-follow).
              // Non-zero means the launch itself failed.
              if (code === 0) {
                console.log(`[sparkrun:deploy] launcher exited 0 for ${deploymentId}`);
                // Workload should now be discoverable via check-job; health-check
                // loop will confirm and report "running".
              } else {
                sendMsg("agent:deployment:status", {
                  deploymentId,
                  status: "failed",
                  error: `Sparkrun launch failed with exit code ${code}`,
                });
              }
            },
          );
          // Status updates are driven by log phase detection and health-check loop
        } catch (err) {
          sendMsg("agent:deployment:status", {
            deploymentId,
            status: "failed",
            error: String(err),
          });
        }
        break;
      }

      // Unknown deploy kind (no inlineRecipeYaml, no recipeRef, no ollama runtime)
      sendMsg("agent:deployment:status", {
        deploymentId,
        status: "failed",
        error: "No recipeRef or inlineRecipeYaml specified",
      });
      break;
    }

    case "cmd:undeploy": {
      const { deploymentId, deleteAfter, clusterNodes, runtime, modelName: undeployModelName } = msg.payload as {
        deploymentId: string; deleteAfter?: boolean; clusterNodes?: string[]; runtime?: string; modelName?: string;
      };
      sendMsg("agent:deployment:status", { deploymentId, status: "stopping" });
      clearPhaseTracking(deploymentId); // reset forward-only phase tracking for this id
      // Record the cancel SYNCHRONOUSLY, before yielding: a launch whose
      // `docker run -d` is still in flight re-checks this at exit and tears the
      // container down. Nothing here can await before the flag is set.
      deployCancels.requestCancel(deploymentId, deleteAfter || false);

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

          // sparkrun/dgxrun deployment — identified by presence in the deployment store
          const stored = loadDeployments().find((d) => d.deploymentId === deploymentId);
          if (stored != null && stored.kind === "dgxrun") {
            sendMsg("agent:deployment:log", {
              deploymentId,
              log: "\n=== Stop requested — tearing down local dgxrun rank container ===\n",
            });
            // Mark stopping so a racing health tick doesn't classify this as a crash.
            saveDeployment({ ...stored, stopping: true });
            try { stopDgxrun(deploymentId); }
            catch (stopErr) { console.warn(`[undeploy] dgxrun stop error (continuing): ${stopErr}`); }
            sendMsg("agent:deployment:status", {
              deploymentId, status: "stopped", deleteAfter: deleteAfter || false,
            });
            return;
          }
          if (stored != null) {
            sendMsg("agent:deployment:log", {
              deploymentId,
              log: "\n=== Stop requested by user — stopping sparkrun workload ===\n",
            });
            // Mark stopping BEFORE calling stopSparkrun so that any health tick
            // racing between here and the workload vanishing sees stopping===true
            // and does NOT classify the deployment as failed.
            saveDeployment({ ...stored, stopping: true });
            const target = stored.clusterId ?? stored.recipeFile;
            const hosts = stored.clusterNodes ?? [];
            try {
              stopSparkrun(deploymentId, target, hosts, stored.tp);
            } catch (stopErr) {
              console.warn(`[undeploy] sparkrun stop error (continuing): ${stopErr}`);
            }
            // Clean up any inline recipe YAML file we may have written
            removeInlineRecipe(deploymentId, SPARKRUN_ADHOC_DIR);
            sendMsg("agent:deployment:status", {
              deploymentId,
              status: "stopped",
              deleteAfter: deleteAfter || false,
            });
            return;
          }

          // No tracked deployment found. Do NOT just report stopped: an orphaned
          // dgxrun container from a raced launch is untracked by definition, and
          // reporting stopped without removing it is why a second DELETE never
          // cleaned one up. `docker rm -f dgxrun_<id>` is idempotent and a no-op
          // for every other runtime.
          try { stopDgxrun(deploymentId); }
          catch (e) { console.warn(`[undeploy] orphan sweep error (continuing): ${e}`); }
          sendMsg("agent:deployment:status", {
            deploymentId,
            status: "stopped",
            deleteAfter: deleteAfter || false,
          });
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
        clusterNodes, modelName, recipeFile, artifactVariant,
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

      const port = (config?.port as number) ?? 8000;
      const gpuMem = (config?.gpuMem as number) ?? 0.85;
      const maxModelLen = (config?.maxModelLen as number) ?? 4096;
      const tensorParallel = config?.tensorParallel as number | undefined;
      const pipelineParallel = config?.pipelineParallel as number | undefined;
      const servedModelName = modelName || jobId;

      console.log(`[finetune] Deploying merged model via sparkrun from ${modelPath} (container: ${deployContainer || "vllm-node"}${recipeFile ? `, recipe: ${recipeFile}` : ""})`);

      try {
        // Resolve the recipe YAML to deploy.
        // PREFER an inference template if the training recipe ships one —
        // those templates are sparkrun-compatible and carry hand-tuned
        // launch flags for the specific model family.
        // Fall back to synthesising a generic sparkrun recipe from params.
        let recipeYaml: string;
        const recipeDir = recipeFile
          ? join(FINETUNE_RECIPES_REPO, recipeFile)
          : undefined;
        const variantId = artifactVariant ?? "bf16";
        const templatePath = recipeDir
          ? findInferenceTemplate(recipeDir, variantId)
          : null;

        if (templatePath) {
          console.log(`[finetune] Using inference template: ${templatePath}`);
          const raw = readFileSync(templatePath, "utf-8");
          recipeYaml = applyFinetuneSubstitutions(raw, {
            modelPath,
            servedModelName,
          });
        } else {
          console.log(`[finetune] No inference template found — synthesising sparkrun recipe`);
          recipeYaml = renderSparkrunFinetuneRecipe({
            mergedModelPath: modelPath,
            servedModelName,
            container: deployContainer || "vllm-node",
            maxModelLen,
            gpuMem,
          });
        }

        // Write the resolved YAML to shared storage so sparkrun can locate it.
        const recipesDir = `${process.env.SHARED_STORAGE || "/mnt/tank"}/recipes`;
        mkdirSync(recipesDir, { recursive: true });
        const recipeFilePath = join(recipesDir, `finetune-${jobId.slice(0, 12)}.yaml`);
        writeFileSync(recipeFilePath, recipeYaml, "utf-8");
        console.log(`[finetune] Wrote sparkrun recipe to ${recipeFilePath}`);

        // Build hosts: clusterNodes head-first when provided, else local solo.
        const hosts: string[] = (Array.isArray(clusterNodes) && clusterNodes.length > 0)
          ? clusterNodes
          : ["localhost"];

        const opts = {
          hosts,
          tp: tensorParallel,
          pp: pipelineParallel,
          port,
          gpuMem,
          maxModelLen,
          servedModelName,
          recipeName: servedModelName,
        };

        sendMsg("agent:deployment:status", { deploymentId, status: "starting" });
        let lastPhase = "starting";

        launchSparkrun(
          deploymentId,
          recipeFilePath,
          opts,
          (line) => {
            const cleaned = collapseCarriageReturns(line);
            sendMsg("agent:deployment:log", { deploymentId, log: cleaned });

            const fetchProgress = parseFetchingProgress(line);
            if (fetchProgress) emitDeploymentProgress(deploymentId, "downloading", fetchProgress);

            const loadProgress = parseLoadingShards(line);
            if (loadProgress) emitDeploymentProgress(deploymentId, "loading", loadProgress);

            const phase = detectPhase(line);
            if (phase && phase !== lastPhase) {
              lastPhase = phase;
              sendMsg("agent:deployment:status", {
                deploymentId,
                status: phase,
                port: phase === "running" ? port : undefined,
              });
            }
          },
          (code) => {
            // sparkrun run exits 0 once the workload is launched (--no-follow).
            // Non-zero means the launch itself failed.
            if (code === 0) {
              console.log(`[finetune] sparkrun launcher exited 0 for ${deploymentId}`);
              // Workload should now be discoverable via check-job; health-check
              // loop will confirm and report "running".
            } else {
              sendMsg("agent:deployment:status", {
                deploymentId,
                status: "failed",
                error: `Sparkrun finetune deploy failed with exit code ${code}`,
              });
            }
          },
        );
      } catch (err) {
        sendMsg("agent:deployment:status", {
          deploymentId, status: "failed", error: String(err),
        });
      }
      break;
    }

    case "cmd:set-registries": {
      const registries = (msg.payload?.registries ?? []) as RegistryWire[];
      try {
        writeRegistriesFile(registries);
        const recipes = discoverRecipes();
        sendMsg("agent:recipes", { recipes });
        console.log(`Applied ${registries.length} registries; re-discovered ${recipes.length} recipes`);
      } catch (err) {
        console.error("cmd:set-registries failed:", err);
      }
      break;
    }

    case "cmd:rescan-recipes": {
      // Re-scan local recipe directories on demand. Without this, recipes
      // added to the NFS share after agent startup stay invisible until
      // the agent reconnects. Pull the registries from git first so edits to
      // existing recipes are reflected (sparkrun list/run use cached clones
      // that are never auto-pulled).
      try {
        updateRegistries();
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

    case "cmd:hf-cache:scan": {
      console.log("[hf-cache] scan requested");
      sendHfCacheInventory();
      break;
    }

    case "cmd:hf-cache:delete": {
      const { repoId, kind } = msg.payload as { repoId: string; kind?: RepoKind };
      try {
        deleteCachedRepo(resolveHfHome(), kind ?? "model", repoId);
        console.log(`[hf-cache] deleted ${kind ?? "model"} ${repoId}`);
        sendHfCacheInventory();
      } catch (err) {
        console.error(`[hf-cache] delete failed: ${err}`);
        sendHfCacheInventory(`delete ${repoId} failed: ${err}`);
      }
      break;
    }

    case "cmd:update": {
      const { bundleUrl, version } = msg.payload as { bundleUrl: string; version: string };
      const updaterPath = join(__dirname, "updater.js"); // dist/updater.js — same dir as index.js (verified: tsc outDir=dist, ExecStart runs /opt/dgx-agent/dist/index.js)
      const tmpPath = `/tmp/dgx-updater-${Date.now()}.js`;
      const RUN_DIR = "/run/dgx-agent";
      const outcome = launchUpdater({
        bundleUrl, version, updaterPath, nodeIdFile: NODE_ID_FILE, tmpPath,
        lockExists: () => existsSync(`${RUN_DIR}/updating`),
        makeLock: () => { try { execSync(`mkdir -p ${RUN_DIR} && touch ${RUN_DIR}/updating`); } catch { /* */ } },
        copyFile: (src, dest) => execSync(`cp "${src}" "${dest}"`),
        spawnDetached: (cmd, cargs) => { const c = spawn(cmd, cargs, { detached: true, stdio: "ignore" }); c.unref(); },
      });
      console.log(`[update] ${outcome === "launched" ? `launched detached updater for v${version}` : "update already in flight — ignored"}`);
      sendMsg("agent:update-status", { status: outcome === "launched" ? "downloading" : "in-flight", version });
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

    case "cmd:power": {
      const { action, force } = msg.payload as { action?: string; force?: boolean };
      if (action !== "reboot" && action !== "shutdown" && action !== "sleep") {
        console.error(`[power] invalid action: ${action}`);
        sendMsg("agent:power:error", { action: String(action), error: "invalid action" });
        break;
      }
      const isForce = force === true;
      console.log(`[power] ${action}${isForce ? " (force)" : ""}`);

      // Best-effort: read our own MAC (so the server can persist it for WOL) and,
      // for a shutdown/suspend, arm Wake-on-LAN on the egress interface — the NIC
      // usually ships with WOL off and the setting doesn't survive a reboot.
      let mac: string | null = null;
      try {
        const route = execSync("ip route get 1.1.1.1", { timeout: 5_000 }).toString();
        const iface = route.match(/\bdev\s+(\S+)/)?.[1];
        if (iface) {
          mac = readFileSync(`/sys/class/net/${iface}/address`, "utf-8").trim().toLowerCase() || null;
          if (action !== "reboot") {
            execSync(`sudo ethtool -s ${iface} wol g`, { timeout: 5_000 });
          }
        }
      } catch (err) {
        console.error(`[power] WOL arm / MAC read failed (non-fatal): ${err}`);
      }

      // Launch the power command in a transient systemd unit that (a) runs in
      // system.slice so it escapes dgx-agent.service's cgroup and survives our
      // teardown during shutdown, and (b) sleeps briefly so the accept ack can
      // flush before the node goes down. We run systemd-run SYNCHRONOUSLY and
      // ack ONLY after it returns 0 — a fork-starved node (or any launch
      // failure) throws here and is reported as an error instead of being
      // masked by an ack sent before the reboot ever launched. A unique unit
      // name avoids a "Unit dgx-power.service already exists" collision on retry.
      try {
        const cmd = powerCommand(action as PowerAction, { force: isForce });
        writeFileSync("/tmp/dgx-power.sh", `#!/bin/bash\nset +e\nsleep 1\n${cmd}\n`, { mode: 0o755 });
        execSync(powerLaunchCommand(powerUnitName(Date.now()), "/tmp/dgx-power.sh"), {
          timeout: 10_000,
          stdio: "ignore",
        });
        // Ack only after the power command actually launched.
        sendMsg("agent:power:accepted", { action, force: isForce, mac });
      } catch (err) {
        console.error(`[power] failed to launch power command: ${err}`);
        sendMsg("agent:power:error", { action, error: String(err) });
      }
      break;
    }

    case "agent:cap:request": {
      const { id, name, input } = msg.payload as { id: string; name: string; input: unknown };
      const ctx = { emitChunk: (stream: "stdout" | "stderr", data: string) => sendMsg("agent:cap:chunk", { id, stream, data }) };
      const result = await caps.dispatch(name, input, ctx);
      sendMsg("agent:cap:result", { id, ...result });
      break;
    }

    default:
      console.log(`Unknown command: ${msg.type}`);
  }
}

// Restrict Ollama's unauthenticated :11434 to the manager + loopback.
// Fire-and-forget at boot: never blocks the WS connect, and a firewall
// failure must not take down metrics/deploy duties — applyOllamaFirewall
// logs loudly to the journal instead of throwing.
void applyOllamaFirewall(MANAGER_URL).finally(() => {
  // Firewall state is final now (applied/failed). Re-send the self-audit so a
  // connected dashboard clears the transient "Startup apply still in progress"
  // without waiting for a reconnect. If the apply finished BEFORE the WS
  // connected, the connect handler's own audit already saw the final state;
  // if the socket isn't open, sendSelfAudit no-ops harmlessly.
  sendSelfAudit();
});

connect();
