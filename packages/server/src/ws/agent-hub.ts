import { WebSocketServer, WebSocket } from "ws";
import { appendFileSync, mkdirSync } from "fs";
import { prisma } from "../prisma.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { metricsBuffer } from "../metrics-buffer.js";

export interface OllamaModelInfo {
  name: string;
  size: string;
  description: string;
}

export interface VllmRecipe {
  file: string;
  name: string;
  description?: string;
  model?: string;
  container: string;
  cluster_only?: boolean;
  solo_only?: boolean;
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
  scripts: { entrypoint: string; train: string; launch: string; ds_config?: string };
  defaults: Record<string, unknown>;
  hardware: { min_nodes: number; gpus_per_node: number; vram_estimate_mb: number };
  deploy?: { container: string; gpu_memory_utilization?: number; max_model_len?: number };
}

interface AgentConnection {
  ws: WebSocket;
  nodeId: string;
}

export class AgentHub {
  private wss: WebSocketServer;
  private agents = new Map<string, AgentConnection>();
  private recipes: VllmRecipe[] = [];
  private trainingRecipes: TrainingRecipe[] = [];
  private ollamaModels: OllamaModelInfo[] = [];
  private onMetrics?: (nodeId: string, metrics: Record<string, unknown>) => void;
  private onRecipes?: (recipes: VllmRecipe[]) => void;
  private onTrainingRecipes?: (recipes: TrainingRecipe[]) => void;

  constructor() {
    this.wss = new WebSocketServer({ noServer: true, perMessageDeflate: false });
    this.wss.on("connection", (ws) => this.handleConnection(ws));
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

  private handleConnection(ws: WebSocket) {
    let nodeId: string | null = null;

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
            await prisma.node.update({
              where: { id: nodeId! },
              data: {
                status: "online",
                gpuModel: msg.payload.gpuModel,
                vramTotal: msg.payload.vramTotal,
                agentVersion,
                ...(archValue ? { arch: archValue } : {}),
                ...fastIpUpdate,
                lastSeen: new Date(),
              },
            });
            console.log(`Agent registered: ${nodeId} (v${agentVersion || "unknown"})`);
            sseBroadcast({ type: "node:status", payload: { nodeId, status: "online", agentVersion } });
            break;
          }

          case "agent:register-token": {
            const { token, hostname, gpuModel, vramTotal, agentVersion: tokenAgentVersion, arch: reportedArch, fastIpAddress } = msg.payload;
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

            // Extract IP from WebSocket connection
            const remoteIp = (ws as unknown as { _socket?: { remoteAddress?: string } })._socket?.remoteAddress?.replace("::ffff:", "") || null;

            // Create node record
            const node = await prisma.node.create({
              data: {
                name: nodeName,
                ipAddress: remoteIp,
                fastIpAddress: typeof fastIpAddress === "string" ? fastIpAddress : null,
                status: "online",
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

          case "agent:recipes": {
            const incoming = msg.payload.recipes as VllmRecipe[];
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
            await prisma.metricSnapshot.create({
              data: {
                nodeId,
                gpuUtil: msg.payload.gpuUtil,
                vramUsed: msg.payload.vramUsed,
                tps: msg.payload.tps ?? null,
                activeRequests: msg.payload.activeRequests ?? null,
                temperature: msg.payload.temp ?? null,
              },
            });
            await prisma.node.update({
              where: { id: nodeId },
              data: { lastSeen: new Date() },
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
