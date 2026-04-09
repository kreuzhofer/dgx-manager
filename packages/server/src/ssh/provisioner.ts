import { sshExec, SshResult } from "./executor.js";
import { SHARED_STORAGE } from "../env.js";
import { broadcast as sseBroadcast } from "../sse.js";

export interface PrereqCheck {
  name: string;
  status: "green" | "yellow" | "red";
  detail: string;
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
      name: "Ollama",
      cmd: "curl -sf --max-time 2 http://localhost:11434/api/tags > /dev/null 2>&1 && ollama --version 2>&1 | grep -oP 'version is \\K[0-9.]+' || echo ''",
      eval: (r) => ({
        name: "Ollama",
        status: r.code === 0 && r.stdout.trim() ? "green" : "yellow",
        detail: r.code === 0 && r.stdout.trim() ? `v${r.stdout.trim()} (running)` : "Not running — install or start service",
      }),
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
  ];

  for (const item of checkItems) {
    emit(item.name, "checking");
    const result = await check(host, item.cmd);
    const prereq = item.eval(result);
    checks.push(prereq);
    emit(item.name, prereq.status, prereq.detail);
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
        r = await sshExec(host, [
          "curl -fsSL https://ollama.ai/install.sh | sh",
          // Ensure systemd service exists with OLLAMA_MODELS on NFS
          "sudo mkdir -p /etc/systemd/system/ollama.service.d",
          `echo -e '[Service]\\nUser=${process.env.SSH_USER || "ubuntu"}\\nEnvironment=HOME=/home/${process.env.SSH_USER || "ubuntu"}\\nEnvironment=OLLAMA_MODELS=${SHARED_STORAGE}/models/ollama\\nEnvironment=OLLAMA_HOST=0.0.0.0\\nEnvironment=OLLAMA_MAX_LOADED_MODELS=0' | sudo tee /etc/systemd/system/ollama.service.d/override.conf`,
          "sudo systemctl daemon-reload",
          "sudo systemctl enable ollama",
          "sudo systemctl restart ollama",
        ].join(" && "), { timeout: 300_000 });
        break;
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
