import type { Request, Response } from "express";

export type SseEvent = {
  type: string;
  payload: unknown;
};

const clients = new Set<Response>();

export function sseHandler(_req: Request, res: Response) {
  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
  });
  res.write("\n");

  clients.add(res);
  _req.on("close", () => clients.delete(res));
}

export function broadcast(event: SseEvent) {
  const data = `data: ${JSON.stringify(event)}\n\n`;
  for (const client of clients) {
    client.write(data);
  }
}
