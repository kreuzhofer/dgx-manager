import { readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";
import http from "http";

const __dirname = dirname(fileURLToPath(import.meta.url));
const OLLAMA_API = process.env.OLLAMA_HOST || "http://localhost:11434";
const OLLAMA_PORT = 11434;

export interface OllamaModel {
  name: string;
  size: string;
  description: string;
}

export interface OllamaStatus {
  deploymentId: string;
  modelName: string;
  loaded: boolean;
  vramUsed: number | null;
}

// Track active Ollama deployments
const activeDeployments = new Map<string, string>(); // deploymentId → modelName

/** Load curated model list. */
export function getOllamaModels(): OllamaModel[] {
  try {
    return JSON.parse(readFileSync(join(__dirname, "../ollama-models.json"), "utf-8"));
  } catch {
    return [];
  }
}

/** Simple fetch helper for Ollama API. */
async function ollamaFetch(path: string, options?: {
  method?: string;
  body?: Record<string, unknown>;
  timeout?: number;
}): Promise<unknown> {
  const url = `${OLLAMA_API}${path}`;
  const res = await fetch(url, {
    method: options?.method || "GET",
    headers: options?.body ? { "Content-Type": "application/json" } : undefined,
    body: options?.body ? JSON.stringify(options.body) : undefined,
    signal: options?.timeout ? AbortSignal.timeout(options.timeout) : undefined,
  });
  if (!res.ok) throw new Error(`Ollama API ${res.status}: ${await res.text()}`);
  return res.json();
}

/** Pull a model with streaming progress. */
async function pullModel(
  modelName: string,
  onLog?: (line: string) => void
): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = new URL(`${OLLAMA_API}/api/pull`);

    const req = http.request(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    }, (res) => {
      let buffer = "";
      res.on("data", (chunk: Buffer) => {
        buffer += chunk.toString();
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const msg = JSON.parse(line);
            if (msg.status) {
              const pct = msg.completed && msg.total
                ? ` ${Math.round(msg.completed / msg.total * 100)}%`
                : "";
              onLog?.(`${msg.status}${pct}\n`);
            }
            if (msg.error) {
              reject(new Error(msg.error));
              return;
            }
          } catch { /* partial JSON */ }
        }
      });
      res.on("end", () => resolve());
      res.on("error", reject);
    });

    req.on("error", reject);
    req.write(JSON.stringify({ name: modelName, stream: true }));
    req.end();
  });
}

/** Deploy an Ollama model: pull if needed, load into GPU memory. */
export async function deployModel(
  deploymentId: string,
  modelName: string,
  onLog?: (line: string) => void,
  onStatus?: (status: string, error?: string) => void
): Promise<number> {
  activeDeployments.set(deploymentId, modelName);

  try {
    // Pull model (streams progress, skips if cached)
    onStatus?.("downloading");
    onLog?.(`Pulling ${modelName}...\n`);
    await pullModel(modelName, onLog);
    onLog?.(`Pull complete.\n`);

    // Load model into GPU memory by sending an empty chat
    onStatus?.("loading");
    onLog?.(`Loading ${modelName} into GPU memory...\n`);
    await ollamaFetch("/api/chat", {
      method: "POST",
      body: { model: modelName, messages: [], stream: false },
      timeout: 300_000, // 5 min for large model loading
    });

    // Verify loaded
    const ps = await ollamaFetch("/api/ps") as { models?: { name: string }[] };
    const loaded = ps.models?.some((m) => m.name.startsWith(modelName));
    if (!loaded) {
      throw new Error("Model not in GPU memory after loading");
    }

    onLog?.(`${modelName} is running on port ${OLLAMA_PORT}\n`);
    onStatus?.("running");
    return OLLAMA_PORT;
  } catch (err) {
    activeDeployments.delete(deploymentId);
    onStatus?.("failed", String(err));
    throw err;
  }
}

/** Unload a model from GPU memory. */
export async function stopModel(deploymentId: string): Promise<void> {
  const modelName = activeDeployments.get(deploymentId);
  if (!modelName) return;

  try {
    // Set keep_alive to 0 to immediately unload
    await ollamaFetch("/api/generate", {
      method: "POST",
      body: { model: modelName, prompt: "", keep_alive: 0 },
      timeout: 30_000,
    });
  } catch { /* model may already be unloaded */ }

  activeDeployments.delete(deploymentId);
}

/** Check health of an Ollama deployment. */
export async function checkOllamaHealth(deploymentId: string): Promise<OllamaStatus | null> {
  const modelName = activeDeployments.get(deploymentId);
  if (!modelName) return null;

  try {
    const ps = await ollamaFetch("/api/ps", { timeout: 5000 }) as {
      models?: { name: string; size: number }[];
    };
    const model = ps.models?.find((m) => m.name.startsWith(modelName));
    return {
      deploymentId,
      modelName,
      loaded: !!model,
      vramUsed: model ? Math.round(model.size / 1024 / 1024) : null, // bytes → MB
    };
  } catch {
    return { deploymentId, modelName, loaded: false, vramUsed: null };
  }
}

/** Check if Ollama service is reachable. */
export async function isOllamaRunning(): Promise<boolean> {
  try {
    await ollamaFetch("/api/tags", { timeout: 3000 });
    return true;
  } catch {
    return false;
  }
}

/** Get active deployment IDs. */
export function getActiveDeployments(): Map<string, string> {
  return activeDeployments;
}
