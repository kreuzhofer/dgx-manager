interface Pending {
  resolve: (r: { ok: boolean; data?: unknown; error?: string }) => void;
  timer: ReturnType<typeof setTimeout>;
  onChunk?: (c: { stream: string; data: string }) => void;
}

export class CapClient {
  private pending = new Map<string, Pending>();
  private seq = 0;

  constructor(
    private send: (nodeId: string, msg: unknown) => void,
    private opts: { timeoutMs?: number } = {},
  ) {}

  invoke(
    nodeId: string,
    name: string,
    input: unknown,
    onChunk?: (c: { stream: string; data: string }) => void,
  ): Promise<{ ok: boolean; data?: unknown; error?: string }> {
    const id = `cap-${++this.seq}`;
    return new Promise((resolve) => {
      const timer = setTimeout(
        () => {
          this.pending.delete(id);
          resolve({ ok: false, error: "cap timeout" });
        },
        this.opts.timeoutMs ?? 30_000,
      );
      this.pending.set(id, { resolve, timer, onChunk });
      this.send(nodeId, {
        type: "agent:cap:request",
        payload: { id, name, input },
      });
    });
  }

  onChunk(payload: { id: string; stream: string; data: string }): void {
    this.pending.get(payload.id)?.onChunk?.({
      stream: payload.stream,
      data: payload.data,
    });
  }

  onResult(payload: {
    id: string;
    ok: boolean;
    data?: unknown;
    error?: string;
  }): void {
    const p = this.pending.get(payload.id);
    if (!p) return;
    clearTimeout(p.timer);
    this.pending.delete(payload.id);
    p.resolve({ ok: payload.ok, data: payload.data, error: payload.error });
  }
}
