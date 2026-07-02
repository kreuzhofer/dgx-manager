import { execSync } from "child_process";
import { hostname as osHostname } from "os";
import { firewallAuditCheck } from "./firewall.js";

export interface SelfAuditCheck {
  name: string;
  status: "green" | "yellow" | "red";
  detail: string;
}

export interface SelfAuditReport {
  systemInfo: string;
  checks: SelfAuditCheck[];
}

function run(cmd: string, timeout = 5000): string | null {
  try {
    return execSync(cmd, { encoding: "utf-8", timeout, stdio: ["ignore", "pipe", "ignore"] }).trim();
  } catch {
    return null;
  }
}

export function selfAudit(): SelfAuditReport {
  const checks: SelfAuditCheck[] = [];

  const nvidia = run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader");
  checks.push(
    nvidia
      ? { name: "NVIDIA Drivers", status: "green", detail: nvidia }
      : { name: "NVIDIA Drivers", status: "red", detail: "nvidia-smi unavailable" }
  );

  const docker = run("docker --version");
  checks.push(
    docker
      ? { name: "Docker", status: "green", detail: docker }
      : { name: "Docker", status: "red", detail: "docker command not found" }
  );

  const ctk = run("dpkg -l nvidia-container-toolkit 2>/dev/null | awk '/^ii/{print $3}' | head -1");
  checks.push(
    ctk
      ? { name: "nvidia-container-toolkit", status: "green", detail: `Installed (${ctk})` }
      : { name: "nvidia-container-toolkit", status: "red", detail: "Not installed" }
  );

  checks.push({ name: "Node.js", status: "green", detail: process.version });

  const ollama = run("ollama --version 2>&1 | head -1");
  checks.push(
    ollama && !ollama.toLowerCase().includes("not found")
      ? { name: "Ollama", status: "green", detail: ollama.replace(/^ollama version is\s*/i, "v") }
      : { name: "Ollama", status: "yellow", detail: "Not installed (optional)" }
  );

  // Ollama :11434 firewall state — applied fire-and-forget at agent boot;
  // this reads the CURRENT state (may be "pending" on the very first audit
  // of a fresh process; reconnect audits deliver the final state).
  checks.push(firewallAuditCheck());

  const tank = run("findmnt -n /mnt/tank -o SOURCE,FSTYPE 2>/dev/null");
  checks.push(
    tank
      ? { name: "NFS /mnt/tank", status: "green", detail: `Mounted (${tank})` }
      : { name: "NFS /mnt/tank", status: "yellow", detail: "Not mounted (optional)" }
  );

  const sysInfo =
    run("uname -srm") || osHostname();
  const osPretty = run(". /etc/os-release && echo $PRETTY_NAME") || "";
  const systemInfo = [sysInfo, osPretty].filter(Boolean).join("\n");

  return { systemInfo, checks };
}
