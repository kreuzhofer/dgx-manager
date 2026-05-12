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
  type: "chat" | "embedding";
  description: string;
}

export interface OllamaStatus {
  deploymentId: string;
  modelName: string;
  loaded: boolean;
  vramUsed: number | null;
}

// Track active Ollama deployments and their abort controllers
const activeDeployments = new Map<string, string>(); // deploymentId → modelName
const activeAbortControllers = new Map<string, AbortController>(); // deploymentId → AbortController

/** Load curated model list. */
export function getOllamaModels(): OllamaModel[] {
  try {
    return JSON.parse(readFileSync(join(__dirname, "../../ollama-models.json"), "utf-8"));
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

/** Pull a model with streaming progress. Abortable via signal. */
async function pullModel(
  modelName: string,
  onLog?: (line: string) => void,
  signal?: AbortSignal
): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) { reject(new Error("Aborted")); return; }

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

    req.on("error", (err) => {
      if (signal?.aborted) resolve(); // Treat abort as success (clean stop)
      else reject(err);
    });
    signal?.addEventListener("abort", () => req.destroy());
    req.write(JSON.stringify({ name: modelName, stream: true }));
    req.end();
  });
}

/** Deploy an Ollama model: pull if needed, load into GPU memory. */
export async function deployModel(
  deploymentId: string,
  modelName: string,
  onLog?: (line: string) => void,
  onStatus?: (status: string, error?: string) => void,
  modelType?: "chat" | "embedding"
): Promise<{ port: number; vramActual: number }> {
  activeDeployments.set(deploymentId, modelName);
  const abortController = new AbortController();
  activeAbortControllers.set(deploymentId, abortController);

  try {
    // Verify Ollama service is reachable, try to start it if not
    if (!await isOllamaRunning()) {
      onLog?.("Ollama service not running, attempting to start...\n");
      try {
        const { execSync } = await import("child_process");
        execSync("sudo systemctl start ollama", { timeout: 10_000 });
        // Wait for it to come up
        for (let i = 0; i < 5; i++) {
          await new Promise((r) => setTimeout(r, 2000));
          if (await isOllamaRunning()) break;
        }
      } catch { /* ignore */ }
      if (!await isOllamaRunning()) {
        throw new Error("Ollama service is not running and could not be started (port 11434 unreachable)");
      }
      onLog?.("Ollama service started.\n");
    }

    // Pull model (streams progress, skips if cached)
    onStatus?.("downloading");
    onLog?.(`Pulling ${modelName}...\n`);
    await pullModel(modelName, onLog, abortController.signal);
    if (abortController.signal.aborted) return { port: OLLAMA_PORT, vramActual: 0 };
    onLog?.(`Pull complete.\n`);

    // Load model into GPU memory
    onStatus?.("loading");
    onLog?.(`Loading ${modelName} into GPU memory...\n`);
    if (modelType === "embedding") {
      await ollamaFetch("/api/embed", {
        method: "POST",
        body: { model: modelName, input: "test", keep_alive: -1 },
        timeout: 300_000,
      });
    } else {
      await ollamaFetch("/api/chat", {
        method: "POST",
        body: { model: modelName, messages: [], stream: false, keep_alive: -1 },
        timeout: 300_000,
      });
    }

    // Verify loaded and get actual VRAM usage
    const ps = await ollamaFetch("/api/ps") as { models?: { name: string; size: number }[] };
    const loadedModel = ps.models?.find((m) => m.name.startsWith(modelName));
    if (!loadedModel) {
      throw new Error("Model not in GPU memory after loading");
    }

    const vramActualMB = Math.round(loadedModel.size / 1024 / 1024);
    onLog?.(`${modelName} is running on port ${OLLAMA_PORT} (${Math.round(vramActualMB / 1024)}GB VRAM)\n`);
    onStatus?.("running");
    return { port: OLLAMA_PORT, vramActual: vramActualMB };
  } catch (err) {
    activeDeployments.delete(deploymentId);
    onStatus?.("failed", String(err));
    throw err;
  }
}

/** Stop an Ollama deployment: abort any in-progress pull, unload model from GPU. */
export async function stopModel(deploymentId: string, modelNameOverride?: string): Promise<void> {
  // Abort in-progress pull if any
  const controller = activeAbortControllers.get(deploymentId);
  if (controller) {
    controller.abort();
    activeAbortControllers.delete(deploymentId);
  }

  const modelName = activeDeployments.get(deploymentId) || modelNameOverride;
  if (!modelName) return;

  try {
    // Set keep_alive to 0 to immediately unload from GPU
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

/**
 * Decide what (if any) status transition to report from one health-check tick.
 *
 * `loaded` is whether the deployment's model is currently in Ollama's
 * /api/ps response. `prev` is the last status we reported for this
 * deployment (undefined if we haven't reported anything yet in this
 * deploy cycle — agents clear it at the start of cmd:deploy).
 *
 * Returns "evicted" only on the running -> not-loaded transition,
 * "running" on the (undefined|evicted) -> loaded transition, and null
 * otherwise (steady state or still-starting).
 */
export function decideOllamaStateTransition(
  loaded: boolean,
  prev: string | undefined,
): "evicted" | "running" | null {
  if (!loaded && prev === "running") return "evicted";
  if (loaded && (prev === undefined || prev === "evicted")) return "running";
  return null;
}
