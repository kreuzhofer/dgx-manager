import { dirname } from "path";
import { fileURLToPath } from "url";
import http from "http";
import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

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

export interface OllamaPullProgress {
  /** Raw status string from Ollama (e.g. "pulling manifest", "downloading"). */
  status: string;
  /** 0-100 if Ollama reported byte counts, else null. */
  percent: number | null;
  /** Bytes pulled so far if reported. */
  current: number | null;
  /** Total bytes for the current layer if reported. */
  total: number | null;
}

// Track active Ollama deployments and their abort controllers
const activeDeployments = new Map<string, string>(); // deploymentId → modelName
const activeAbortControllers = new Map<string, AbortController>(); // deploymentId → AbortController

/**
 * Legacy: the manager's Settings page is now the source of truth for the
 * Ollama catalog. Kept exported so callers don't break; always returns [].
 */
export function getOllamaModels(): OllamaModel[] {
  return [];
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
  signal?: AbortSignal,
  onProgress?: (p: OllamaPullProgress) => void,
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
              const percent = msg.completed && msg.total
                ? Math.round(msg.completed / msg.total * 100)
                : null;
              const pct = percent !== null ? ` ${percent}%` : "";
              onLog?.(`${msg.status}${pct}\n`);
              onProgress?.({
                status: String(msg.status),
                percent,
                current: typeof msg.completed === "number" ? msg.completed : null,
                total: typeof msg.total === "number" ? msg.total : null,
              });
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

/**
 * Injected dependencies for ensureOllamaRunning — lets tests drive the
 * decision/wait logic without systemd or a real Ollama HTTP API.
 */
export interface EnsureOllamaDeps {
  /** Is the Ollama HTTP API reachable right now? */
  isRunning: () => Promise<boolean>;
  /** Start the systemd unit (throws on failure). */
  startService: () => Promise<void>;
  /** Injectable sleep so tests never actually wait. */
  sleep: (ms: number) => Promise<void>;
  onLog?: (line: string) => void;
}

/**
 * Make sure the Ollama service is up before a deploy. Fleet policy disables
 * Ollama autostart on all nodes (unauthenticated :11434 API), so a stopped
 * service is the expected state — start it via systemd and poll the API
 * until it answers (bounded: maxAttempts x intervalMs, default ~20s).
 * Throws with a clear message if the start command fails or the API never
 * becomes reachable — no silent fallback.
 *
 * Concurrent deploys may both call this; the double-start is benign
 * (`systemctl start` is idempotent) and each caller polls independently.
 */
export async function ensureOllamaRunning(
  deps: EnsureOllamaDeps,
  opts: { maxAttempts?: number; intervalMs?: number } = {},
): Promise<void> {
  const maxAttempts = opts.maxAttempts ?? 10;
  const intervalMs = opts.intervalMs ?? 2000;

  if (await deps.isRunning()) return;

  deps.onLog?.("Ollama service not running, starting it (systemctl start ollama)...\n");
  try {
    await deps.startService();
  } catch (err) {
    // execFile errors already name the failing command, so don't repeat it here.
    throw new Error(
      `Failed to start Ollama service: ${err instanceof Error ? err.message : String(err)}`,
    );
  }

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    await deps.sleep(intervalMs);
    if (await deps.isRunning()) {
      deps.onLog?.("Ollama service started.\n");
      return;
    }
  }
  throw new Error(
    `Ollama service was started but the API did not become reachable after ${maxAttempts} attempts over ~${Math.round((maxAttempts * intervalMs) / 1000)}s (port ${OLLAMA_PORT})`,
  );
}

/** Start the Ollama systemd unit. `sudo -n` so a missing sudoers rule fails
 *  fast instead of hanging on a password prompt; argv-array exec (no shell). */
async function startOllamaService(): Promise<void> {
  await execFileAsync("sudo", ["-n", "systemctl", "start", "ollama"], { timeout: 15_000 });
}

/** Deploy an Ollama model: pull if needed, load into GPU memory. */
export async function deployModel(
  deploymentId: string,
  modelName: string,
  onLog?: (line: string) => void,
  onStatus?: (status: string, error?: string) => void,
  modelType?: "chat" | "embedding",
  onProgress?: (p: OllamaPullProgress) => void,
): Promise<{ port: number; vramActual: number }> {
  activeDeployments.set(deploymentId, modelName);
  const abortController = new AbortController();
  activeAbortControllers.set(deploymentId, abortController);

  try {
    // Fleet policy: Ollama's systemd unit does not auto-start (unauthenticated
    // :11434 API), so an Ollama deploy must start the service on demand.
    // Failure throws and is reported to the manager via the catch below.
    await ensureOllamaRunning({
      isRunning: isOllamaRunning,
      startService: startOllamaService,
      sleep: (ms) => new Promise((r) => setTimeout(r, ms)),
      onLog,
    });

    // Pull model (streams progress, skips if cached)
    onStatus?.("downloading");
    onLog?.(`Pulling ${modelName}...\n`);
    await pullModel(modelName, onLog, abortController.signal, onProgress);
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
  // NOTE: auto-stopping the Ollama service after the last deployment is
  // removed is intentionally out of scope for now; a future stop hook
  // (e.g. `systemctl stop ollama` when activeDeployments is empty) would go here.
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
