import { Server as HttpServer } from "http";
import { WebSocketServer, WebSocket } from "ws";
import { prisma } from "../prisma.js";

interface AgentConnection {
  ws: WebSocket;
  nodeId: string;
}

export class AgentHub {
  private wss: WebSocketServer;
  private agents = new Map<string, AgentConnection>();
  private onMetrics?: (nodeId: string, metrics: Record<string, unknown>) => void;

  constructor(server: HttpServer) {
    this.wss = new WebSocketServer({ server, path: "/ws/agent" });
    this.wss.on("connection", (ws) => this.handleConnection(ws));
  }

  setMetricsHandler(handler: (nodeId: string, metrics: Record<string, unknown>) => void) {
    this.onMetrics = handler;
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
            await prisma.node.update({
              where: { id: nodeId! },
              data: {
                status: "online",
                gpuModel: msg.payload.gpuModel,
                vramTotal: msg.payload.vramTotal,
                lastSeen: new Date(),
              },
            });
            console.log(`Agent registered: ${nodeId}`);
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
            this.onMetrics?.(nodeId, msg.payload);
            break;
          }

          case "agent:deployment:status": {
            const { deploymentId, status, port, error } = msg.payload;
            await prisma.deployment.update({
              where: { id: deploymentId },
              data: { status, port: port ?? undefined },
            });
            if (error) console.error(`Deployment ${deploymentId} error: ${error}`);
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
