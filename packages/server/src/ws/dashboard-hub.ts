import { Server as HttpServer } from "http";
import { WebSocketServer, WebSocket } from "ws";

export class DashboardHub {
  private wss: WebSocketServer;
  private clients = new Set<WebSocket>();

  constructor(server: HttpServer) {
    this.wss = new WebSocketServer({ server, path: "/ws/dashboard" });
    this.wss.on("connection", (ws) => {
      this.clients.add(ws);
      ws.on("close", () => this.clients.delete(ws));
    });
  }

  broadcast(type: string, payload: unknown) {
    const msg = JSON.stringify({ type, payload });
    for (const client of this.clients) {
      if (client.readyState === WebSocket.OPEN) {
        client.send(msg);
      }
    }
  }
}
