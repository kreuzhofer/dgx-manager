import { WebSocketServer, WebSocket } from "ws";
import { prisma } from "../prisma.js";
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

interface AgentConnection {
  ws: WebSocket;
  nodeId: string;
}

export class AgentHub {
  private wss: WebSocketServer;
  private agents = new Map<string, AgentConnection>();
  private recipes: VllmRecipe[] = [];
  private ollamaModels: OllamaModelInfo[] = [];
  private onMetrics?: (nodeId: string, metrics: Record<string, unknown>) => void;
  private onRecipes?: (recipes: VllmRecipe[]) => void;

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

  getRecipes(): VllmRecipe[] {
    return this.recipes;
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
            await prisma.node.update({
              where: { id: nodeId! },
              data: {
                status: "online",
                gpuModel: msg.payload.gpuModel,
                vramTotal: msg.payload.vramTotal,
                agentVersion,
                lastSeen: new Date(),
              },
            });
            console.log(`Agent registered: ${nodeId} (v${agentVersion || "unknown"})`);
            sseBroadcast({ type: "node:status", payload: { nodeId, status: "online", agentVersion } });
            break;
          }

          case "agent:recipes": {
            const incoming = msg.payload.recipes as VllmRecipe[];
            this.recipes = incoming;
            console.log(`Received ${incoming.length} vLLM recipes from agent ${nodeId}`);
            this.onRecipes?.(incoming);
            break;
          }

          case "agent:ollama-models": {
            this.ollamaModels = msg.payload.models as OllamaModelInfo[];
            console.log(`Received ${this.ollamaModels.length} Ollama models from agent ${nodeId}`);
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
              await prisma.deployment.update({
                where: { id: deploymentId },
                data: {
                  status,
                  port: port ?? undefined,
                  vramActual: vramActual ? Number(vramActual) : undefined,
                },
              });
            } catch {
              // Deployment may already be deleted
              break;
            }
            if (error) console.error(`Deployment ${deploymentId} error: ${error}`);
            sseBroadcast({ type: "deployment:status", payload: { deploymentId, status, port, error } });

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
            sseBroadcast({ type: "deployment:log", payload: { deploymentId, log } });
            break;
          }

          case "agent:finetune:progress": {
            const { jobId, progress, logs } = msg.payload;
            await prisma.fineTuneJob.update({
              where: { id: jobId },
              data: { progress, logs },
            });
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
