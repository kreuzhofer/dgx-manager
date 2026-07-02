import { sshExec, SshResult } from "./executor.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { macCaptureCmd, normalizeMac } from "../nodes/power.js";
import { prisma } from "../prisma.js";

// ---------------------------------------------------------------------------
// sparkrun command-string builders (pure, exported for callers + tests)
// ---------------------------------------------------------------------------

export const SPARKRUN_PKG = "sparkrun==0.2.38";
const SPARKRUN = `uvx --from ${SPARKRUN_PKG} sparkrun`;

/** Audit: is sparkrun runnable via uvx on this host? */
export function sparkrunAuditCmd(): string {
  return `${SPARKRUN} --version >/dev/null 2>&1 && echo installed`;
}

/** Non-interactive per-host setup steps (run over SSH on the node). */
export function sparkrunSetupCmds(host: string): string[] {
  return [
    `${SPARKRUN} setup install -H ${host}`,
    `${SPARKRUN} setup docker-group -H ${host}`,
    `${SPARKRUN} setup earlyoom -H ${host}`,
  ];
}

/**
 * Cluster-wide SSH mesh setup (all nodes must trust each other).
 * This is a cluster-level op — do NOT wire into per-node provisioning.
 * Exported for callers that orchestrate multi-node setup.
 */
export function sparkrunMeshCmd(hosts: string[]): string {
  return `${SPARKRUN} setup ssh -H ${hosts.join(",")}`;
}

/**
 * OPT-IN, EXPENSIVE (~15 min): pre-build the sparkrun image on a host by
 * launching a real recipe run.  Do NOT call this inside auditNode/provisionNode
 * automatically — it blocks a node for up to 15 minutes.
 */
export function sparkrunPrewarmCmd(recipe: string, host: string): string {
  return `${SPARKRUN} run ${recipe} -H ${host} --no-follow`;
}

export interface PrereqCheck {
  name: string;
  status: "green" | "yellow" | "red";
  detail: string;
}

// ---------------------------------------------------------------------------
// Ollama command-string builders + audit parsing (pure, exported for tests)
// ---------------------------------------------------------------------------

/**
 * Audit: is the Ollama binary installed, independent of service run-state?
 * Fleet policy keeps the ollama service disabled (unauthenticated :11434),
 * so a stopped-but-installed Ollama must still count as installed.
 *
 * Output (only when the binary exists): a version line ("0.20.3") followed
 * by the systemd active-state line ("active" / "inactive" / ...). Empty
 * output means not installed.
 */
export function ollamaAuditCmd(): string {
  return `if command -v ollama >/dev/null 2>&1; then ollama --version 2>&1 | grep -oP 'version is \\K[0-9.]+'; systemctl is-active ollama 2>/dev/null || true; fi`;
}

/** Parse ollamaAuditCmd() output into the Ollama prerequisite check. */
export function evalOllamaAudit(stdout: string): PrereqCheck {
  const lines = stdout
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);
  if (lines.length === 0) {
    return { name: "Ollama", status: "yellow", detail: "Not installed — can auto-install" };
  }
  const version = lines.find((l) => /^[0-9][0-9.]*$/.test(l));
  const serviceActive = lines.includes("active");
  return {
    name: "Ollama",
    status: "green",
    detail: `${version ? `v${version}` : "installed (version unknown)"} (service ${serviceActive ? "running" : "stopped"})`,
  };
}

/**
 * Install Ollama with the NFS models dir + OLLAMA_HOST drop-in, WITHOUT
 * boot auto-start. Ollama's :11434 API is unauthenticated; fleet policy is
 * autostart disabled everywhere, so the final step is `systemctl disable`.
 * The service is deliberately NOT stopped — it stays in whatever run-state
 * the installer left it in for this boot (an Ollama deploy may use it).
 */
export function ollamaInstallCmd(sshUser: string): string {
  // sshUser is interpolated inside a single-quoted string piped to a
  // root-privileged tee — fail fast on anything that could break quoting.
  if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(sshUser)) {
    throw new Error(`ollamaInstallCmd: invalid sshUser ${JSON.stringify(sshUser)}`);
  }
  return [
    "curl -fsSL https://ollama.ai/install.sh | sh",
    // Ensure systemd service exists with OLLAMA_MODELS on NFS
    "sudo mkdir -p /etc/systemd/system/ollama.service.d",
    `echo -e '[Service]\\nUser=${sshUser}\\nEnvironment=HOME=/home/${sshUser}\\nEnvironment=OLLAMA_MODELS=${SHARED_STORAGE}/models/ollama\\nEnvironment=OLLAMA_HOST=0.0.0.0\\nEnvironment=OLLAMA_MAX_LOADED_MODELS=0' | sudo tee /etc/systemd/system/ollama.service.d/override.conf`,
    "sudo systemctl daemon-reload",
    "sudo systemctl restart ollama",
    // No autostart on boot (installer enables it by default — undo that).
    "sudo systemctl disable ollama",
  ].join(" && ");
}

export interface ProvisionReport {
  reachable: boolean;
  sudoAvailable: boolean;
  systemInfo: string;
  checks: PrereqCheck[];
}

async function check(host: string, cmd: string): Promise<SshResult> {
  try {
    return await sshExec(host, cmd);
  } catch {
    return { code: -1, stdout: "", stderr: "unreachable" };
  }
}

export async function auditNode(host: string, nodeId?: string): Promise<ProvisionReport> {
  const emit = (step: string, status: string, detail?: string) => {
    if (nodeId) {
      sseBroadcast({ type: "node:provision", payload: { nodeId, step, status, detail } });
    }
  };

  // Step 1: Connectivity + sudo
  emit("Connecting", "running", `SSH to ${host}...`);
  const echoResult = await check(host, "echo ok");
  if (echoResult.code !== 0) {
    emit("Connecting", "failed", "Unreachable");
    return { reachable: false, sudoAvailable: false, systemInfo: "", checks: [] };
  }
  emit("Connecting", "done", "Reachable");

  emit("Sudo check", "running");
  const sudoResult = await check(host, "sudo -n true");
  const sudoAvailable = sudoResult.code === 0;
  emit("Sudo check", "done", sudoAvailable ? "Available" : "Not available");

  // Step 2: System info
  const sysInfo = await check(host, "uname -a && lsb_release -ds 2>/dev/null || true");

  // Step 3: Prerequisite checks
  const checks: PrereqCheck[] = [];

  const checkItems: { name: string; cmd: string; eval: (r: SshResult) => PrereqCheck }[] = [
    {
      name: "NVIDIA Drivers",
      cmd: "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null",
      eval: (r) => ({
        name: "NVIDIA Drivers",
        status: r.code === 0 ? "green" : "red",
        detail: r.code === 0 ? r.stdout : "nvidia-smi not found — install drivers manually",
      }),
    },
    {
      name: "Docker",
      cmd: "docker --version 2>/dev/null",
      eval: (r) => ({
        name: "Docker",
        status: r.code === 0 ? "green" : sudoAvailable ? "yellow" : "red",
        detail: r.code === 0 ? r.stdout : sudoAvailable ? "Not installed — can auto-install" : "Not installed — need sudo",
      }),
    },
    {
      // Independent of "Docker" install — Docker can be present (factory
      // image) while the SSH user is missing from the docker group, in
      // which case `docker ps` fails with EACCES on the socket and ssh-in
      // diagnostics are blocked.
      name: "Docker group",
      cmd: "id -nG \"$USER\" | tr ' ' '\\n' | grep -qx docker && echo member",
      eval: (r) => ({
        name: "Docker group",
        status: r.stdout.includes("member") ? "green" : sudoAvailable ? "yellow" : "red",
        detail: r.stdout.includes("member") ? "User in docker group" : sudoAvailable ? "User not in docker group — can auto-install" : "User not in docker group — need sudo",
      }),
    },
    {
      // spark-vllm-docker's hf-download.sh hard-requires uvx to fetch
      // HuggingFace models on first-time deploys. Without it the agent
      // silently fails on a fresh node ("uvx: command not found") on
      // the very first model download. Astral's installer drops the
      // binaries in ~/.local/bin and is idempotent.
      name: "uv (uvx)",
      cmd: "test -x \"$HOME/.local/bin/uvx\" && echo installed",
      eval: (r) => ({
        name: "uv (uvx)",
        status: r.stdout.includes("installed") ? "green" : "yellow",
        detail: r.stdout.includes("installed") ? "Installed" : "Not installed — can auto-install (no sudo needed)",
      }),
    },
    {
      name: "nvidia-container-toolkit",
      cmd: "dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q ^ii && echo installed",
      eval: (r) => ({
        name: "nvidia-container-toolkit",
        status: r.stdout.includes("installed") ? "green" : sudoAvailable ? "yellow" : "red",
        detail: r.stdout.includes("installed") ? "Installed" : sudoAvailable ? "Not installed — can auto-install" : "Not installed — need sudo",
      }),
    },
    {
      name: "Node.js",
      cmd: "node --version 2>/dev/null",
      eval: (r) => ({
        name: "Node.js",
        status: r.code === 0 ? "green" : sudoAvailable ? "yellow" : "red",
        detail: r.code === 0 ? r.stdout : sudoAvailable ? "Not installed — can auto-install" : "Not installed — need sudo",
      }),
    },
    {
      // Detects the binary, not the HTTP API — fleet policy keeps the
      // ollama service disabled (unauthenticated :11434), so installed-
      // but-stopped must still report green.
      name: "Ollama",
      cmd: ollamaAuditCmd(),
      eval: (r) => evalOllamaAudit(r.stdout),
    },
    {
      name: `NFS ${SHARED_STORAGE}`,
      cmd: `mountpoint -q ${SHARED_STORAGE} && echo mounted`,
      eval: (r) => ({
        name: `NFS ${SHARED_STORAGE}`,
        status: r.stdout.includes("mounted") ? "green" : "red",
        detail: r.stdout.includes("mounted") ? "Mounted" : "Not mounted — configure NFS manually",
      }),
    },
    {
      // sparkrun orchestrates multi-node vLLM launches via uvx; no sudo
      // needed — runs entirely as the SSH user under ~/.local/bin.
      name: "sparkrun",
      cmd: sparkrunAuditCmd(),
      eval: (r) => ({
        name: "sparkrun",
        status: r.stdout.includes("installed") ? "green" : "yellow",
        detail: r.stdout.includes("installed")
          ? `Installed (${SPARKRUN_PKG})`
          : "Not installed — can auto-install (no sudo needed)",
      }),
    },
  ];

  for (const item of checkItems) {
    emit(item.name, "checking");
    const result = await check(host, item.cmd);
    const prereq = item.eval(result);
    checks.push(prereq);
    emit(item.name, prereq.status, prereq.detail);
  }

  if (nodeId) {
    try {
      const r = await sshExec(host, macCaptureCmd(host), { timeout: 10_000 });
      const mac = normalizeMac(r.stdout);
      if (mac) await prisma.node.update({ where: { id: nodeId }, data: { macAddress: mac } });
    } catch {
      // non-fatal — audit should not fail because MAC capture failed
    }
  }

  return {
    reachable: true,
    sudoAvailable,
    systemInfo: sysInfo.stdout,
    checks,
  };
}

export async function provisionNode(host: string, checks: PrereqCheck[], nodeId?: string): Promise<string> {
  const logs: string[] = [];

  const emit = (step: string, status: string, detail?: string) => {
    if (nodeId) {
      sseBroadcast({ type: "node:provision", payload: { nodeId, step, status, detail } });
    }
  };

  const toInstall = checks.filter((c) => c.status === "yellow");
  let completed = 0;

  for (const item of toInstall) {
    completed++;
    const progress = `${completed}/${toInstall.length}`;
    emit(item.name, "installing", `Installing ${item.name}... (${progress})`);
    logs.push(`Installing ${item.name}...`);

    let r;
    switch (item.name) {
      case "Docker":
        r = await sshExec(host, "curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER", { timeout: 120_000 });
        break;
      case "Docker group":
        r = await sshExec(host, "sudo usermod -aG docker $USER", { timeout: 30_000 });
        break;
      case "uv (uvx)":
        // Astral's installer; runs as the SSH user (no sudo), drops
        // binaries in ~/.local/bin. Idempotent on re-run.
        r = await sshExec(host, "curl -LsSf https://astral.sh/uv/install.sh | sh", { timeout: 60_000 });
        break;
      case "nvidia-container-toolkit":
        r = await sshExec(host, [
          "distribution=$(. /etc/os-release;echo $ID$VERSION_ID)",
          "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
          "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
          "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
          "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
        ].join(" && "), { timeout: 120_000 });
        break;
      case "Node.js":
        r = await sshExec(host, "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs", { timeout: 120_000 });
        break;
      case "Ollama":
        r = await sshExec(host, ollamaInstallCmd(process.env.SSH_USER || "ubuntu"), { timeout: 300_000 });
        break;
      case "sparkrun": {
        // Run setup steps in sequence; no sudo required.
        const setupCmds = sparkrunSetupCmds(host);
        let lastResult: SshResult = { code: 0, stdout: "", stderr: "" };
        for (const cmd of setupCmds) {
          lastResult = await sshExec(host, cmd, { timeout: 120_000 });
          if (lastResult.code !== 0) break;
        }
        r = lastResult;
        // Remind operators that image pre-warm is opt-in (~15 min).
        console.log(
          `[provisioner] sparkrun setup complete for ${host}. ` +
          `Image pre-warm is opt-in (first deploy will build ~15 min). ` +
          `To pre-warm manually: sparkrunPrewarmCmd("<recipe>", "${host}")`
        );
        emit(
          "sparkrun",
          "info",
          `Image pre-warm is opt-in (~15 min first-build). ` +
          `Use sparkrunPrewarmCmd() to trigger it explicitly.`
        );
        break;
      }
      default:
        continue;
    }

    if (r!.code === 0) {
      emit(item.name, "installed", `${item.name} installed successfully`);
      logs.push(`${item.name} installed`);
    } else {
      emit(item.name, "failed", `${item.name} install failed: ${r!.stderr.slice(0, 200)}`);
      logs.push(`${item.name} install failed: ${r!.stderr}`);
    }
  }

  emit("Re-auditing", "running", "Verifying installation...");

  return logs.join("\n");
}
