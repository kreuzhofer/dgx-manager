export interface CapContext {
  emitChunk(stream: "stdout" | "stderr", data: string): void;
}

export interface Capability {
  name: string;
  handle(input: unknown, ctx: CapContext): Promise<unknown>;
}

export type CapResult = { ok: true; data: unknown } | { ok: false; error: string };

export class CapRegistry {
  private caps = new Map<string, Capability>();

  register(c: Capability): void {
    this.caps.set(c.name, c);
  }

  async dispatch(name: string, input: unknown, ctx: CapContext): Promise<CapResult> {
    const cap = this.caps.get(name);
    if (!cap) return { ok: false, error: `unknown capability: ${name}` };
    try {
      return { ok: true, data: await cap.handle(input, ctx) };
    } catch (e) {
      return { ok: false, error: (e as Error).message };
    }
  }
}
