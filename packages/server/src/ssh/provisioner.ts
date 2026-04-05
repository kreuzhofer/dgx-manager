import { sshExec, SshResult } from "./executor.js";

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

export async function auditNode(host: string): Promise<ProvisionReport> {
  // Step 1: Connectivity + sudo
  const echoResult = await check(host, "echo ok");
  if (echoResult.code !== 0) {
    return { reachable: false, sudoAvailable: false, systemInfo: "", checks: [] };
  }

  const sudoResult = await check(host, "sudo -n true");
  const sudoAvailable = sudoResult.code === 0;

  // Step 2: System info
  const sysInfo = await check(host, "uname -a && lsb_release -ds 2>/dev/null || true");

  // Step 3: Prerequisite checks
  const checks: PrereqCheck[] = [];

  const nvidiaSmi = await check(host, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null");
  checks.push({
    name: "NVIDIA Drivers",
    status: nvidiaSmi.code === 0 ? "green" : "red",
    detail: nvidiaSmi.code === 0 ? nvidiaSmi.stdout : "nvidia-smi not found — install drivers manually",
  });

  const docker = await check(host, "docker --version 2>/dev/null");
  checks.push({
    name: "Docker",
    status: docker.code === 0 ? "green" : sudoAvailable ? "yellow" : "red",
    detail: docker.code === 0 ? docker.stdout : sudoAvailable ? "Not installed — can auto-install" : "Not installed — need sudo",
  });

  const nvidiaDocker = await check(host, "dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q ^ii && echo installed");
  checks.push({
    name: "nvidia-container-toolkit",
    status: nvidiaDocker.stdout.includes("installed") ? "green" : sudoAvailable ? "yellow" : "red",
    detail: nvidiaDocker.stdout.includes("installed") ? "Installed" : sudoAvailable ? "Not installed — can auto-install" : "Not installed — need sudo",
  });

  const nodeJs = await check(host, "node --version 2>/dev/null");
  checks.push({
    name: "Node.js",
    status: nodeJs.code === 0 ? "green" : sudoAvailable ? "yellow" : "red",
    detail: nodeJs.code === 0 ? nodeJs.stdout : sudoAvailable ? "Not installed — can auto-install" : "Not installed — need sudo",
  });

  const ollama = await check(host, "ollama --version 2>/dev/null");
  checks.push({
    name: "Ollama",
    status: ollama.code === 0 ? "green" : "yellow",
    detail: ollama.code === 0 ? ollama.stdout : "Not installed — can auto-install",
  });

  const nfs = await check(host, "mountpoint -q /mnt/tank && echo mounted");
  checks.push({
    name: "NFS /mnt/tank",
    status: nfs.stdout.includes("mounted") ? "green" : "red",
    detail: nfs.stdout.includes("mounted") ? "Mounted" : "Not mounted — configure NFS manually",
  });

  return {
    reachable: true,
    sudoAvailable,
    systemInfo: sysInfo.stdout,
    checks,
  };
}

export async function provisionNode(host: string, checks: PrereqCheck[]): Promise<string> {
  const logs: string[] = [];

  for (const item of checks) {
    if (item.status !== "yellow") continue;

    logs.push(`Installing ${item.name}...`);

    switch (item.name) {
      case "Docker": {
        const r = await sshExec(host, "curl -fsSL https://get.docker.com | sudo sh && sudo usermod -aG docker $USER", { timeout: 120_000 });
        logs.push(r.code === 0 ? "Docker installed" : `Docker install failed: ${r.stderr}`);
        break;
      }
      case "nvidia-container-toolkit": {
        const r = await sshExec(host, [
          "distribution=$(. /etc/os-release;echo $ID$VERSION_ID)",
          "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
          "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
          "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
          "sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker",
        ].join(" && "), { timeout: 120_000 });
        logs.push(r.code === 0 ? "nvidia-container-toolkit installed" : `Install failed: ${r.stderr}`);
        break;
      }
      case "Node.js": {
        const r = await sshExec(host, "curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash - && sudo apt-get install -y nodejs", { timeout: 120_000 });
        logs.push(r.code === 0 ? "Node.js installed" : `Install failed: ${r.stderr}`);
        break;
      }
      case "Ollama": {
        const r = await sshExec(host, "curl -fsSL https://ollama.ai/install.sh | sh", { timeout: 120_000 });
        logs.push(r.code === 0 ? "Ollama installed" : `Install failed: ${r.stderr}`);
        break;
      }
    }
  }

  return logs.join("\n");
}
