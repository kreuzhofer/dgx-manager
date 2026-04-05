import { Client } from "ssh2";
import { readFileSync } from "fs";
import { homedir } from "os";
import { join } from "path";

export interface SshResult {
  code: number;
  stdout: string;
  stderr: string;
}

export async function sshExec(
  host: string,
  command: string,
  options?: { timeout?: number }
): Promise<SshResult> {
  const timeout = options?.timeout ?? 30_000;

  return new Promise((resolve, reject) => {
    const conn = new Client();
    const timer = setTimeout(() => {
      conn.end();
      reject(new Error(`SSH command timed out after ${timeout}ms`));
    }, timeout);

    conn
      .on("ready", () => {
        conn.exec(command, (err, stream) => {
          if (err) {
            clearTimeout(timer);
            conn.end();
            return reject(err);
          }

          let stdout = "";
          let stderr = "";

          stream.on("data", (data: Buffer) => {
            stdout += data.toString();
          });
          stream.stderr.on("data", (data: Buffer) => {
            stderr += data.toString();
          });
          stream.on("close", (code: number) => {
            clearTimeout(timer);
            conn.end();
            resolve({ code: code ?? 0, stdout: stdout.trim(), stderr: stderr.trim() });
          });
        });
      })
      .on("error", (err) => {
        clearTimeout(timer);
        reject(err);
      })
      .connect({
        host,
        port: 22,
        username: process.env.SSH_USER || process.env.USER || "ubuntu",
        privateKey: readFileSync(join(homedir(), ".ssh", "id_rsa")),
      });
  });
}
