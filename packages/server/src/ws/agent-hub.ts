import { WebSocketServer, WebSocket } from "ws";
import { appendFileSync, mkdirSync } from "fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { metricsBuffer } from "../metrics-buffer.js";
import type { HfCacheNodeInventory } from "../hf-cache/grouping.js";
import { resolveNodeIp, isValidIpv4 } from "./node-ip.js";
import { scheduleDebouncedReseed } from "../ssh/known-hosts-trigger.js";
import { pushRegistriesToAgent } from "../registries/push.js";
import { normalizeMac } from "../nodes/power.js";
import { coordinatedDgxrunTeardown } from "../deployments/dgxrun-teardown.js";
import { CapClient } from "../caps/cap-client.js";

export interface OllamaModelInfo {
  name: string;
  size: string;
  description: string;
}

export type RecipeArch = "amd64" | "arm64" | "any";

export interface VllmRecipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  solo_only?: boolean;
  /** Target CPU arch of the recipe; used for per-node filtering + admission. */
  arch: RecipeArch;
  defaults: Record<string, unknown>;
}

export interface TrainingRecipe {
  file: string;
  name: string;
  description?: string;
  base_model: string;
  framework: string;
  method: string;
  dataset_format?: string;
  container: { image: string; name: string; build_context?: string };
  scripts: {
    entrypoint: string;
    train: string;
    launch: string;
    ds_config?: string;
    merge?: string;
    /** Post-merge FP8 quantization wrapper (shared scripts/quantize_fp8.py).
     *  Mirrors the agent-side TrainingRecipe.scripts.quantize_fp8 field. */
    quantize_fp8?: string;
  };
  defaults: Record<string, unknown>;
  hardware: { min_nodes: number; gpus_per_node: number; vram_estimate_mb: number };
  deploy?: { container: string; gpu_memory_utilization?: number; max_model_len?: number };
  /** Mirror of agent's TrainingRecipe.inferenceVariants — one entry per
   *  inference*.yaml file in the recipe dir. Empty/undefined when the
   *  recipe doesn't ship inference templates. */
  inferenceVariants?: {
    id: string;
    filename: string;
    name: string;
    description?: string;
  }[];
}

interface AgentConnection {
  ws: WebSocket;
  nodeId: string;
}

/**
 * Shape of the `sysinfo` blob Agent v2 attaches to `agent:metrics` payloads
 * (packages/agent/src/sysinfo/proc-read.ts::readSysInfo). Only the fields the
 * hub reads to populate MetricSnapshot summary columns are declared here —
 * the full blob is passed through to dashboards untouched via the existing
 * `...msg.payload` spread in the `node:metrics` SSE broadcast. Older agents
 * omit this field entirely, so every access below must be optional-chained.
 */
interface AgentSysInfo {
  pressure?: { memory?: { some?: { avg10?: number } } };
  load?: { totalProcs?: number };
  fds?: { allocated?: number };
  sshd?: Record<string, number>;
  thermalsC?: number[];
}

export interface OllamaPullProgressMsg {
  deploymentId: string;
  status: string;
  percent: number | null;
  current: number | null;
  total: number | null;
}

/**
 * Translate an `agent:ollama:pull-progress` payload into the canonical
 * `deployment:progress` SSE shape the dashboard already renders. Kept as a
 * named export so it can be unit-tested without spinning up a WebSocket.
 */
export function handleOllamaPullProgress(payload: OllamaPullProgressMsg): void {
  sseBroadcast({
    type: "deployment:progress",
    payload: {
      deploymentId: payload.deploymentId,
      phase: payload.status === "downloading" ? "downloading" : payload.status,
      phaseProgress: payload.percent ?? 0,
      current: payload.current,
      total: payload.total,
    },
  });
}

export class AgentHub {
  private wss: WebSocketServer;
  private agents = new Map<string, AgentConnection>();
  private recipes: VllmRecipe[] = [];
  private trainingRecipes: TrainingRecipe[] = [];
  private ollamaModels: OllamaModelInfo[] = [];
  /** Latest HF-cache inventory per node, pushed by agents on cmd:hf-cache:scan
   *  or after a delete. In-memory only — the filesystem is the source of truth. */
  private hfCacheInventories = new Map<string, HfCacheNodeInventory>();
  private onMetrics?: (nodeId: string, metrics: Record<string, unknown>) => void;
  private onRecipes?: (recipes: VllmRecipe[]) => void;
  private onTrainingRecipes?: (recipes: TrainingRecipe[]) => void;
  /** Request/response + streaming bridge for Agent v2 capability invocations
   *  (diag.collect, exec). Routed to the target node's WS via sendToAgent;
   *  results/chunks come back through the agent:cap:result/chunk cases below. */
  readonly capClient: CapClient;

  constructor() {
    this.wss = new WebSocketServer({ noServer: true, perMessageDeflate: false });
    this.wss.on("connection", (ws) => this.handleConnection(ws));
    this.capClient = new CapClient((nodeId, msg) => this.sendToAgent(nodeId, msg as Record<string, unknown>));
  }

  handleUpgrade(request: import("http").IncomingMessage, socket: import("stream").Duplex, head: Buffer) {
    this.wss.handleUpgrade(request, socket, head, (ws) => {
      this.wss.emit("connection", ws, request);
    });
  }

  setMetricsHandler(handler: (nodeId: string, metrics: Record<string, unknown>) => void) {
    this.onMetrics = handler;
  }

  setRecipesHandler(handler: (recipes: VllmRecipe[]) => void) {
    this.onRecipes = handler;
  }

  setTrainingRecipesHandler(handler: (recipes: TrainingRecipe[]) => void) {
    this.onTrainingRecipes = handler;
  }

  getRecipes(): VllmRecipe[] {
    return this.recipes;
  }

  getTrainingRecipes(): TrainingRecipe[] {
    return this.trainingRecipes;
  }

  getOllamaModels(): OllamaModelInfo[] {
    return this.ollamaModels;
  }

  getHfCacheInventories(): HfCacheNodeInventory[] {
    return [...this.hfCacheInventories.values()];
  }

  private handleConnection(ws: WebSocket) {
    let nodeId: string | null = null;
    // Agent-supplied management IP (NODE_ADVERTISE_IP), remembered for this
    // connection so the per-metric-tick self-heal below doesn't clobber it back
    // to the WS source (e.g. the docker bridge gateway for a co-located node).
    let advertiseIp: string | null = null;

    ws.on("message", async (data) => {
      try {
        const msg = JSON.parse(data.toString());

        switch (msg.type) {
          case "agent:register": {
            nodeId = msg.payload.nodeId;
            this.agents.set(nodeId!, { ws, nodeId: nodeId! });
            const agentVersion = msg.payload.agentVersion || null;
            const reportedArch = msg.payload.arch;
            const archValue =
              reportedArch === "amd64" || reportedArch === "arm64" ? reportedArch : undefined;
            // fastIpAddress: explicitly null clears (interface went away),
            // string overwrites, undefined leaves prior value untouched.
            const fastIp = msg.payload.fastIpAddress;
            const fastIpUpdate =
              fastIp === undefined ? {} : { fastIpAddress: typeof fastIp === "string" ? fastIp : null };
            // ipAddress: refresh from the WebSocket source on every reconnect.
            // The node's management IP can change (DHCP lease, netplan edit,
            // NIC swap), and stale values break the SSH-based audit/provision
            // flows since they target whatever IP we have on file.
            const remoteIp = (ws as unknown as { _socket?: { remoteAddress?: string } })._socket?.remoteAddress?.replace("::ffff:", "");
            advertiseIp = isValidIpv4(msg.payload.advertiseIp) ? msg.payload.advertiseIp : null;
            const resolvedIp = resolveNodeIp(advertiseIp, remoteIp);
            const ipUpdate = resolvedIp ? { ipAddress: resolvedIp } : {};
            await prisma.node.update({
              where: { id: nodeId! },
              data: {
                status: "online",
                // a reconnecting agent means the box is back up — clear any off/rebooting/waking intent
                powerState: "on",
                gpuModel: msg.payload.gpuModel,
                vramTotal: msg.payload.vramTotal,
                agentVersion,
                ...(archValue ? { arch: archValue } : {}),
                ...fastIpUpdate,
                ...ipUpdate,
                lastSeen: new Date(),
              },
            });
            console.log(`Agent registered: ${nodeId} (v${agentVersion || "unknown"})`);
            sseBroadcast({ type: "node:status", payload: { nodeId, status: "online", agentVersion } });
            // A (re)connected agent may be a re-imaged node whose host key
            // changed — refresh the cluster known_hosts mesh (debounced + throttled).
            scheduleDebouncedReseed();
            // Reconcile this node's sparkrun registries to the manager's source-of-truth set.
            await pushRegistriesToAgent(this, nodeId!).catch((err) =>
              console.error(`Failed to push registries to ${nodeId}:`, err),
            );
            break;
          }

          case "agent:register-token": {
            const { token, hostname, gpuModel, vramTotal, agentVersion: tokenAgentVersion, arch: reportedArch, fastIpAddress, advertiseIp: advertiseIpRaw } = msg.payload;
            const archValue =
              reportedArch === "amd64" || reportedArch === "arm64" ? reportedArch : null;

            // Validate token
            const joinToken = await prisma.joinToken.findUnique({ where: { token } });
            if (!joinToken) {
              ws.send(JSON.stringify({ type: "register:rejected", payload: { error: "Invalid token" } }));
              ws.close();
              return;
            }
            if (joinToken.revokedAt) {
              ws.send(JSON.stringify({ type: "register:rejected", payload: { error: "Token has been revoked" } }));
              ws.close();
              return;
            }
            if (joinToken.usedAt) {
              ws.send(JSON.stringify({ type: "register:rejected", payload: { error: "Token has already been used" } }));
              ws.close();
              return;
            }
            if (joinToken.expiresAt && joinToken.expiresAt < new Date()) {
              ws.send(JSON.stringify({ type: "register:rejected", payload: { error: "Token has expired" } }));
              ws.close();
              return;
            }

            // Resolve unique name (append suffix on collision)
            let nodeName = hostname || "node";
            let suffix = 0;
            while (true) {
              const candidateName = suffix === 0 ? nodeName : `${nodeName}-${suffix}`;
              const existing = await prisma.node.findUnique({ where: { name: candidateName } });
              if (!existing) {
                nodeName = candidateName;
                break;
              }
              suffix++;
            }

            // Extract IP from WebSocket connection; prefer an agent-supplied
            // advertise IP (NODE_ADVERTISE_IP) when valid.
            const remoteIp = (ws as unknown as { _socket?: { remoteAddress?: string } })._socket?.remoteAddress?.replace("::ffff:", "") || null;
            advertiseIp = isValidIpv4(advertiseIpRaw) ? advertiseIpRaw : null;

            // Create node record
            const node = await prisma.node.create({
              data: {
                name: nodeName,
                ipAddress: resolveNodeIp(advertiseIp, remoteIp),
                fastIpAddress: typeof fastIpAddress === "string" ? fastIpAddress : null,
                status: "online",
                // a reconnecting agent means the box is back up — clear any off/rebooting/waking intent
                powerState: "on",
                provisionStatus: "agent-deployed",
                bootstrapMethod: "token",
                gpuModel: gpuModel || null,
                vramTotal: vramTotal || null,
                agentVersion: tokenAgentVersion || null,
                arch: archValue,
                dockerAvailable: true, // install script ensures Docker is present
                lastSeen: new Date(),
              },
            });

            // Mark token as used
            await prisma.joinToken.update({
              where: { id: joinToken.id },
              data: { usedAt: new Date(), usedByNodeId: node.id },
            });

            nodeId = node.id;
            this.agents.set(nodeId, { ws, nodeId });

            console.log(`Agent registered via token: ${nodeId} (${nodeName}, v${tokenAgentVersion || "unknown"})`);

            // Send acceptance with nodeId
            ws.send(JSON.stringify({ type: "register:accepted", payload: { nodeId: node.id } }));

            // Tell dashboards a new node record was created (so they can append
            // it to their list without a reload), then broadcast the usual
            // online status update.
            sseBroadcast({ type: "node:created", payload: node });
            sseBroadcast({ type: "node:status", payload: { nodeId, status: "online", agentVersion: tokenAgentVersion } });
            // A (re)connected agent may be a re-imaged node whose host key
            // changed — refresh the cluster known_hosts mesh (debounced + throttled).
            scheduleDebouncedReseed();
            // Reconcile this node's sparkrun registries to the manager's source-of-truth set.
            await pushRegistriesToAgent(this, nodeId).catch((err) =>
              console.error(`Failed to push registries to ${nodeId}:`, err),
            );
            break;
          }

          case "agent:self-audit": {
            if (!nodeId) break;
            const { systemInfo, checks } = msg.payload as {
              systemInfo: string;
              checks: { name: string; status: "green" | "yellow" | "red"; detail: string }[];
            };
            const report = { reachable: true, sudoAvailable: true, systemInfo, checks };
            const dockerGreen = checks.find((c) => c.name === "Docker")?.status === "green";
            const ollamaGreen = checks.find((c) => c.name === "Ollama")?.status === "green";
            await prisma.node.update({
              where: { id: nodeId },
              data: {
                provisionStatus: "agent-deployed",
                provisionLog: JSON.stringify(report),
                dockerAvailable: dockerGreen,
                ollamaInstalled: ollamaGreen,
              },
            }).catch(() => {});
            sseBroadcast({
              type: "node:provision",
              payload: {
                nodeId,
                step: "Self-audit complete",
                status: "done",
                provisionStatus: "agent-deployed",
                report,
              },
            });
            break;
          }

          case "agent:update-status": {
            console.log(`Agent ${nodeId} update: ${msg.payload.status} (v${msg.payload.version || "?"})`);
            sseBroadcast({ type: "node:update-status", payload: { nodeId, ...msg.payload } });
            break;
          }

          case "agent:power:accepted": {
            if (!nodeId) break;
            // The agent accepted a cmd:power and is about to go down. Persist the
            // MAC it reported so a later /wake works even if the node was never
            // SSH-audited. powerState was already set optimistically by /power.
            const mac = normalizeMac(String(msg.payload.mac ?? ""));
            console.log(`Agent ${nodeId} power ${msg.payload.action} accepted${mac ? ` (mac ${mac})` : ""}`);
            if (mac) {
              await prisma.node.update({ where: { id: nodeId }, data: { macAddress: mac } }).catch(() => {});
            }
            break;
          }

          case "agent:power:error": {
            if (!nodeId) break;
            // The agent could not run the power command (e.g. invalid action). The
            // node is still up, so undo the optimistic powerState so the card
            // doesn't stay stuck as off/rebooting.
            console.error(`Agent ${nodeId} power error: ${msg.payload.error}`);
            await prisma.node.update({ where: { id: nodeId }, data: { powerState: "on" } }).catch(() => {});
            sseBroadcast({ type: "node:status", payload: { nodeId, powerState: "on" } });
            break;
          }

          case "agent:recipes": {
            const incoming = (msg.payload.recipes as VllmRecipe[]).map((r) => ({
              // Legacy agents (pre arch-aware) omit `arch`. Default to "arm64"
              // — the existing fleet is all DGX Spark (arm64) — as an explicit,
              // observable fallback so these recipes still filter/admit sanely.
              ...r,
              arch: r.arch === "amd64" || r.arch === "arm64" || r.arch === "any" ? r.arch : "arm64",
            }));
            this.recipes = incoming;
            console.log(`Received ${incoming.length} vLLM recipes from agent ${nodeId}`);
            this.onRecipes?.(incoming);
            break;
          }

          case "agent:training-recipes": {
            const incoming = msg.payload.recipes as TrainingRecipe[];
            this.trainingRecipes = incoming;
            console.log(`Received ${incoming.length} training recipe(s) from agent ${nodeId}`);
            this.onTrainingRecipes?.(incoming);
            break;
          }

          case "agent:ollama-models": {
            this.ollamaModels = msg.payload.models as OllamaModelInfo[];
            console.log(`Received ${this.ollamaModels.length} Ollama models from agent ${nodeId}`);
            break;
          }

          case "agent:hf-cache": {
            if (!nodeId) break;
            const inventory: HfCacheNodeInventory = {
              ...(msg.payload as Omit<HfCacheNodeInventory, "nodeId">),
              nodeId,
            };
            this.hfCacheInventories.set(nodeId, inventory);
            console.log(
              `[hf-cache] inventory from ${nodeId}: ${inventory.repos?.length ?? 0} repos` +
                (inventory.error ? ` (error: ${inventory.error})` : ""),
            );
            sseBroadcast({ type: "hf-cache:inventory", payload: inventory });
            break;
          }

          case "agent:ollama-status": {
            // Match loaded Ollama models to active deployments and update vramActual
            if (!nodeId) break;
            const loadedModels = msg.payload.models as { name: string; vramMB: number }[];
            const activeOllama = await prisma.deployment.findMany({
              where: {
                nodeId,
                status: { in: ["running", "evicted"] },
                model: { runtime: "ollama" },
              },
              include: { model: true },
            });
            for (const dep of activeOllama) {
              const loaded = loadedModels.find((m) => m.name.startsWith(dep.model.name));
              if (loaded) {
                if (dep.vramActual !== loaded.vramMB) {
                  await prisma.deployment.update({
                    where: { id: dep.id },
                    data: { vramActual: loaded.vramMB, status: "running" },
                  });
                }
              } else if (dep.status === "running") {
                await prisma.deployment.update({
                  where: { id: dep.id },
                  data: { vramActual: 0, status: "evicted" },
                });
                sseBroadcast({ type: "deployment:status", payload: { deploymentId: dep.id, status: "evicted", vramActual: 0 } });
              }
            }
            break;
          }

          case "agent:metrics": {
            if (!nodeId) break;
            const mem = msg.payload.memory ?? null;
            const psi = msg.payload.pressure ?? null;
            // Agent v2 sysinfo diagnostics (packages/agent/src/sysinfo/proc-read.ts).
            // Older agents don't send this field at all — every read below is
            // optional-chained so a missing/partial blob just leaves the
            // corresponding column null rather than throwing.
            const sysinfo = msg.payload.sysinfo as AgentSysInfo | undefined;
            // sysinfo has no dedicated process-count field (that only exists on
            // the on-demand diag.collect capability's readdir-based count) — use
            // /proc/loadavg's totalProcs as the best available live-tick proxy.
            // Exclude LISTEN: sshd's listening socket is always present whenever
            // sshd is running, so summing all states would make sshdConns never
            // read 0 and mask the always-on baseline. The signal this field
            // exists to detect is active + pre-auth connection pileup
            // (ESTABLISHED + SYN_RECV), not the passive listener.
            const sshdConns = sysinfo?.sshd
              ? Object.entries(sysinfo.sshd).reduce((n, [state, c]) => n + (state === "LISTEN" ? 0 : c), 0)
              : null;
            const tempC = sysinfo?.thermalsC?.length ? Math.max(...sysinfo.thermalsC) : null;
            await prisma.metricSnapshot.create({
              data: {
                nodeId,
                gpuUtil: msg.payload.gpuUtil,
                vramUsed: msg.payload.vramUsed,
                tps: msg.payload.tps ?? null,
                activeRequests: msg.payload.activeRequests ?? null,
                temperature: msg.payload.temp ?? null,
                memTotalMb: mem?.memTotalMb ?? null,
                memAvailableMb: mem?.memAvailableMb ?? null,
                memCachedMb: mem?.memCachedMb ?? null,
                swapTotalMb: mem?.swapTotalMb ?? null,
                swapUsedMb: mem?.swapUsedMb ?? null,
                pressureMemoryAvg10: psi?.memorySomeAvg10 ?? null,
                pressureIoAvg10: psi?.ioSomeAvg10 ?? null,
                pressureCpuAvg10: psi?.cpuSomeAvg10 ?? null,
                psiMemSome10: sysinfo?.pressure?.memory?.some?.avg10 ?? null,
                pidCount: sysinfo?.load?.totalProcs ?? null,
                fdAllocated: sysinfo?.fds?.allocated ?? null,
                sshdConns,
                tempC,
              },
            });
            // Self-heal stale ipAddress: agent:register only fires on WS
            // connect, so a netplan change that doesn't drop the WS leaves
            // the DB stale until the agent restarts. Refresh from the WS
            // source on every metric tick — same pattern as line 110.
            const remoteIp = (ws as unknown as { _socket?: { remoteAddress?: string } })._socket?.remoteAddress?.replace("::ffff:", "");
            const resolvedTickIp = resolveNodeIp(advertiseIp, remoteIp);
            await prisma.node.update({
              where: { id: nodeId },
              data: {
                lastSeen: new Date(),
                ...(resolvedTickIp ? { ipAddress: resolvedTickIp } : {}),
                // Self-heal vramTotal from the live metric tick (the metrics path
                // has the GB10 system-RAM fallback). vramTotal is otherwise only
                // set at register, so a node that registered with a transient 0
                // (e.g. a freshly re-onboarded node) would never recover and its
                // used/free memory can't render. Only overwrite with a real value.
                ...(typeof msg.payload.vramTotal === "number" && msg.payload.vramTotal > 0
                  ? { vramTotal: msg.payload.vramTotal }
                  : {}),
              },
            });
            const now = Date.now();
            metricsBuffer.push(nodeId, {
              timestamp: now,
              gpuUtil: msg.payload.gpuUtil,
              vramUsed: msg.payload.vramUsed,
              temperature: msg.payload.temp ?? null,
              tps: msg.payload.tps ?? null,
              activeRequests: msg.payload.activeRequests ?? null,
              netInterfaces: msg.payload.netInterfaces ?? undefined,
              rdmaInterfaces: msg.payload.rdmaInterfaces ?? undefined,
              diskDevices: msg.payload.diskDevices ?? undefined,
              memory: mem ?? undefined,
              pressure: psi ?? undefined,
            });
            this.onMetrics?.(nodeId, msg.payload);
            sseBroadcast({ type: "node:metrics", payload: { nodeId, timestamp: now, ...msg.payload } });
            break;
          }

          case "agent:deployment:status": {
            const { deploymentId, status, port, error, deleteAfter, vramActual } = msg.payload;
            try {
              const isStopped = ["stopped", "failed", "evicted"].includes(status as string);
              await prisma.deployment.update({
                where: { id: deploymentId },
                data: {
                  status,
                  port: port ?? undefined,
                  vramActual: isStopped ? 0 : (vramActual ? Number(vramActual) : undefined),
                },
              });
            } catch {
              // Deployment may already be deleted
              break;
            }
            if (error) console.error(`Deployment ${deploymentId} error: ${error}`);
            const isStopped = ["stopped", "failed", "evicted"].includes(status as string);
            sseBroadcast({ type: "deployment:status", payload: { deploymentId, status, port, error, vramActual: isStopped ? 0 : (vramActual ? Number(vramActual) : undefined) } });

            // Update cluster node statuses when deployment changes
            if (["stopped", "failed", "running"].includes(status)) {
              await prisma.clusterNode.updateMany({
                where: { deploymentId },
                data: { status },
              }).catch(() => {});
            }

            // dgxrun coordinated teardown: the mp executor has no recovery, so
            // ONE dead rank hangs the whole cluster. When any rank reports
            // failed, tear down every rank. No-op for non-dgxrun deployments.
            if (status === "failed") {
              await coordinatedDgxrunTeardown(this, deploymentId).catch((err) =>
                console.error(`[dgxrun] teardown failed for ${deploymentId}:`, err),
              );
            }

            // Auto-delete record after confirmed stop
            if (status === "stopped" && deleteAfter) {
              try {
                await prisma.clusterNode.deleteMany({ where: { deploymentId } });
                await prisma.loadBalancerEndpoint.deleteMany({ where: { deploymentId } });
                await prisma.deployment.delete({ where: { id: deploymentId } });
                sseBroadcast({ type: "deployment:deleted", payload: { deploymentId } });
                console.log(`Deployment ${deploymentId} deleted after stop`);
              } catch { /* already deleted */ }
            }
            break;
          }

          case "agent:deployment:log": {
            const { deploymentId, log } = msg.payload;
            // Persist deployment logs to file
            try {
              const logDir = `${SHARED_STORAGE}/logs/deployments`;
              mkdirSync(logDir, { recursive: true, mode: 0o777 });
              appendFileSync(`${logDir}/${deploymentId}.log`, log as string, { mode: 0o666 });
            } catch { /* best effort — NFS may not be mounted in dev */ }
            sseBroadcast({ type: "deployment:log", payload: { deploymentId, log } });
            break;
          }

          case "agent:deployment:progress": {
            // In-flight phase progress (e.g. HF download %). No DB persistence
            // — this is ephemeral state, just rebroadcast for the UI.
            sseBroadcast({ type: "deployment:progress", payload: msg.payload });
            break;
          }

          case "agent:ollama:pull-progress": {
            handleOllamaPullProgress(msg.payload as OllamaPullProgressMsg);
            break;
          }

          case "agent:finetune:progress": {
            let { jobId, phase, phaseProgress, step, totalSteps, loss, lr, evalLoss, etaSeconds, log } = msg.payload;
            // Resolve truncated job IDs from reattached containers (12 chars → full cuid)
            if (typeof jobId === "string" && jobId.length < 20) {
              const match = await prisma.fineTuneJob.findFirst({
                where: { id: { startsWith: jobId as string } },
                select: { id: true },
              });
              if (match) jobId = match.id;
            }
            // Persist training progress to DB
            if (phase === "training" && typeof phaseProgress === "number" && phaseProgress > 0) {
              await prisma.fineTuneJob.update({
                where: { id: jobId },
                data: { progress: phaseProgress },
              }).catch(() => {});
            }
            // Persist training metrics for loss curve visualization. Agent
            // emits multiple events per step ([TRAIN] line + dict log + tqdm
            // bar can all match), so dedupe via upsert on (jobId, step).
            if (phase === "training" && typeof step === "number" && typeof loss === "number") {
              await prisma.trainingMetric.upsert({
                where: { jobId_step: { jobId: jobId as string, step: step as number } },
                create: {
                  jobId: jobId as string,
                  step: step as number,
                  loss: loss as number,
                  lr: typeof lr === "number" ? lr : null,
                },
                update: {
                  loss: loss as number,
                  lr: typeof lr === "number" ? lr : undefined,
                },
              }).catch(() => {});
            }
            // Persist eval loss
            if (typeof evalLoss === "number" && typeof jobId === "string") {
              // Attach eval loss to the latest metric for this job
              const latest = await prisma.trainingMetric.findFirst({
                where: { jobId: jobId as string },
                orderBy: { step: "desc" },
              });
              if (latest) {
                await prisma.trainingMetric.update({
                  where: { id: latest.id },
                  data: { evalLoss: evalLoss as number },
                }).catch(() => {});
              }
            }
            sseBroadcast({ type: "finetune:log", payload: { jobId, phase, phaseProgress, step, totalSteps, loss, lr, evalLoss, etaSeconds, log } });
            break;
          }

          case "agent:finetune:complete": {
            const job = msg.payload;
            await prisma.fineTuneJob.update({
              where: { id: job.jobId },
              data: {
                status: job.status,
                outputPath: job.outputPath ?? null,
                logs: job.error ?? undefined,
                completedAt: new Date(),
              },
            });
            sseBroadcast({ type: "finetune:status", payload: { jobId: job.jobId, status: job.status, outputPath: job.outputPath, error: job.error } });
            break;
          }

          case "agent:finetune:merge-progress": {
            const { jobId, phase, phaseProgress, log } = msg.payload;
            sseBroadcast({ type: "finetune:merge-progress", payload: { jobId, phase, phaseProgress, log } });
            break;
          }

          case "agent:finetune:merge-complete": {
            const { jobId, status, mergedPath, error } = msg.payload;
            await prisma.fineTuneJob.update({
              where: { id: jobId },
              data: {
                mergeStatus: status,
                mergedPath: mergedPath ?? null,
              },
            });
            sseBroadcast({ type: "finetune:merge-status", payload: { jobId, status, mergedPath, error } });
            break;
          }

          case "agent:finetune:quantize-progress":
          case "agent:finetune:quantize-complete":
            await this.dispatchFinetuneQuantizeMessage(msg);
            break;

          case "agent:cap:result": {
            this.capClient.onResult(msg.payload as { id: string; ok: boolean; data?: unknown; error?: string });
            break;
          }

          case "agent:cap:chunk": {
            this.capClient.onChunk(msg.payload as { id: string; stream: string; data: string });
            break;
          }

          case "agent:audit": {
            if (!nodeId) break;
            const { cap, cmd, args, reason, code } = msg.payload as {
              cap: string;
              cmd?: string;
              args?: string[];
              reason?: string;
              code?: number;
            };
            await prisma.auditEvent.create({
              data: {
                nodeId,
                cap,
                // cmd is the only free-text field on AuditEvent, so fold args
                // into it — otherwise the audit row records "bash" without
                // its ["-c", "rm -rf /"] arguments, losing the actual invocation.
                cmd: cmd !== undefined ? [cmd, ...(args ?? [])].join(" ") : undefined,
                reason,
                code,
              },
            }).catch((e) => console.error("agent:audit persist failed:", e));
            break;
          }
        }
      } catch (err) {
        console.error("Agent message error:", err);
      }
    });

    ws.on("close", async () => {
      if (nodeId) {
        this.agents.delete(nodeId);
        await prisma.node.update({
          where: { id: nodeId },
          data: { status: "offline" },
        }).catch(() => {});
        console.log(`Agent disconnected: ${nodeId}`);
        sseBroadcast({ type: "node:status", payload: { nodeId, status: "offline" } });
      }
    });
  }

  /**
   * Test-only entry point: lets unit tests invoke the agent-message
   * dispatcher without spinning up a real WebSocket. Routes to the
   * same private dispatchers used by handleConnection's switch.
   */
  async handleAgentMessage(msg: { type: string; payload: Record<string, unknown> }): Promise<void> {
    switch (msg.type) {
      case "agent:finetune:quantize-progress":
      case "agent:finetune:quantize-complete":
        await this.dispatchFinetuneQuantizeMessage(msg);
        break;

      default:
        // Other message types are handled via the WebSocket connection path
        break;
    }
  }

  /**
   * Shared handler for quantize-progress and quantize-complete messages.
   * Called from both handleConnection's switch and the test-only
   * handleAgentMessage entry point so the logic lives in exactly one place.
   */
  private async dispatchFinetuneQuantizeMessage(msg: { type: string; payload: Record<string, unknown> }): Promise<void> {
    switch (msg.type) {
      case "agent:finetune:quantize-progress": {
        const { jobId, phase, phaseProgress, log } = msg.payload;
        sseBroadcast({ type: "finetune:quantize-progress", payload: { jobId, phase, phaseProgress, log } });
        break;
      }

      case "agent:finetune:quantize-complete": {
        const { jobId, status, quantizedPath, error } = msg.payload as {
          jobId: string;
          status: "completed" | "failed";
          quantizedPath: string | null;
          error?: string;
        };
        await prisma.fineTuneJob.update({
          where: { id: jobId },
          data: {
            quantizationStatus: status === "completed" ? "quantized" : "failed",
            quantizedPath: status === "completed" ? quantizedPath : null,
            quantizationLog: error ?? null,
            quantizedAt: status === "completed" ? new Date() : null,
          },
        });
        sseBroadcast({ type: "finetune:quantize-status", payload: { jobId, status, quantizedPath, error } });
        break;
      }
    }
  }

  sendToAgent(nodeId: string, message: Record<string, unknown>) {
    const agent = this.agents.get(nodeId);
    if (agent && agent.ws.readyState === WebSocket.OPEN) {
      agent.ws.send(JSON.stringify(message));
    }
  }

  isAgentOnline(nodeId: string): boolean {
    const agent = this.agents.get(nodeId);
    return !!agent && agent.ws.readyState === WebSocket.OPEN;
  }

  getConnectedNodeIds(): string[] {
    return Array.from(this.agents.keys());
  }
}
