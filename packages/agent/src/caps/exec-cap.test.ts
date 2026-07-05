import { describe, it, expect, vi } from "vitest";
import { EventEmitter } from "events";
import { makeExecCap } from "./exec-cap.js";

function fakeSpawn(out: string, code: number) {
  return () => {
    const child: any = new EventEmitter();
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.kill = () => {};
    setTimeout(() => {
      child.stdout.emit("data", Buffer.from(out));
      child.emit("close", code);
    }, 0);
    return child;
  };
}

describe("exec cap", () => {
  it("rejects when reason is missing", async () => {
    const cap = makeExecCap(fakeSpawn("hi", 0));
    await expect(
      cap.handle({ cmd: "echo", args: ["hi"] }, { emitChunk: () => {} }),
    ).rejects.toThrow(/reason/i);
  });

  it("streams output, returns code, and audits", async () => {
    const audits: any[] = [];
    const cap = makeExecCap(fakeSpawn("hello", 0), (a) => audits.push(a));
    const chunks: string[] = [];
    const res: any = await cap.handle(
      { cmd: "echo", args: ["hello"], reason: "debugging .36 sshd" },
      { emitChunk: (_s, d) => chunks.push(d) },
    );
    expect(res.code).toBe(0);
    expect(chunks.join("")).toContain("hello");
    expect(audits[0].reason).toBe("debugging .36 sshd");
  });
});
