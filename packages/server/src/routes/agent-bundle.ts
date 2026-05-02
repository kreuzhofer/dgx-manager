import { Router } from "express";
import { existsSync, createReadStream, readFileSync } from "fs";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));

export const agentBundleRouter = Router();

// Per-arch bundles live under packages/server/agent-bundles/agent-bundle-<arch>.tar.gz
// Produced by scripts/build-agent-bundles.sh via docker buildx.
const BUNDLES_DIR = join(__dirname, "../../agent-bundles");
const AGENT_PKG_PATH = join(__dirname, "../../../agent/package.json");

const SUPPORTED_ARCHES = ["amd64", "arm64"] as const;
type Arch = (typeof SUPPORTED_ARCHES)[number];

function managerArch(): Arch {
  return process.arch === "x64" ? "amd64" : "arm64";
}

function bundlePath(arch: Arch): string {
  return join(BUNDLES_DIR, `agent-bundle-${arch}.tar.gz`);
}

function getAgentVersion(): string {
  try {
    return JSON.parse(readFileSync(AGENT_PKG_PATH, "utf-8")).version;
  } catch {
    return "unknown";
  }
}

// GET /api/agent/version — return the bundled agent version
agentBundleRouter.get("/version", (_req, res) => {
  res.json({ version: getAgentVersion() });
});

// GET /api/agent/bundle?arch=amd64|arm64 — serve the agent tarball for the requested arch.
// When arch is omitted, falls back to the manager's own arch (back-compat for pre-multi-arch agents).
agentBundleRouter.get("/bundle", (req, res) => {
  const requested = typeof req.query.arch === "string" ? req.query.arch : undefined;
  let arch: Arch;
  if (!requested) {
    arch = managerArch();
  } else if ((SUPPORTED_ARCHES as readonly string[]).includes(requested)) {
    arch = requested as Arch;
  } else {
    return res.status(400).json({
      error: `Unsupported arch "${requested}". Supported: ${SUPPORTED_ARCHES.join(", ")}.`,
    });
  }

  const path = bundlePath(arch);
  if (!existsSync(path)) {
    return res.status(404).json({
      error: `Agent bundle for ${arch} not found. Run scripts/build-agent-bundles.sh first.`,
    });
  }
  res.setHeader("Content-Type", "application/gzip");
  res.setHeader("Content-Disposition", `attachment; filename=agent-bundle-${arch}.tar.gz`);
  createReadStream(path).pipe(res);
});

// GET /api/agent/install.sh — serve the install script with server URL baked in
agentBundleRouter.get("/install.sh", (req, res) => {
  const host = process.env.MANAGER_ADVERTISE_HOST || req.hostname;
  const port = process.env.PORT || "4000";
  const serverUrl = `http://${host}:${port}`;

  const script = generateInstallScript(serverUrl);
  res.setHeader("Content-Type", "text/plain");
  res.send(script);
});

function generateInstallScript(serverUrl: string): string {
  return `#!/usr/bin/env bash
# DGX Manager Agent — Bootstrap Install Script
# Usage: curl -sL ${serverUrl}/api/agent/install.sh | sudo bash -s -- --token TOKEN
#
# Prerequisites: NVIDIA GPU drivers must be installed (nvidia-smi must work).
# This script installs: Docker, nvidia-container-toolkit, Node.js 22.x, and the DGX Manager agent.
set -euo pipefail

SERVER_URL="${serverUrl}"
JOIN_TOKEN=""
AGENT_USER="\${SUDO_USER:-\$(whoami)}"

# Parse arguments
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    --token) JOIN_TOKEN="\$2"; shift 2 ;;
    --server) SERVER_URL="\$2"; shift 2 ;;
    --user) AGENT_USER="\$2"; shift 2 ;;
    *) echo "Unknown option: \$1"; exit 1 ;;
  esac
done

if [ -z "\$JOIN_TOKEN" ]; then
  echo "ERROR: --token is required"
  echo "Usage: curl -sL \${SERVER_URL}/api/agent/install.sh | sudo bash -s -- --token TOKEN"
  exit 1
fi

# Colors
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
RED='\\033[0;31m'
NC='\\033[0m'

log()  { echo -e "\${GREEN}[DGX Agent]\${NC} \$1"; }
warn() { echo -e "\${YELLOW}[DGX Agent]\${NC} \$1"; }
fail() { echo -e "\${RED}[DGX Agent]\${NC} \$1"; exit 1; }

# Must be root
if [ "\$(id -u)" -ne 0 ]; then
  fail "This script must be run as root (use sudo)"
fi

# Detect architecture
ARCH=\$(uname -m)
case "\$ARCH" in
  x86_64)  BUNDLE_ARCH=amd64 ;;
  aarch64) BUNDLE_ARCH=arm64 ;;
  *) fail "Unsupported architecture: \$ARCH (need x86_64 or aarch64)" ;;
esac
log "Architecture: \$ARCH (bundle: \$BUNDLE_ARCH)"

# Detect OS
if [ -f /etc/os-release ]; then
  . /etc/os-release
  log "OS: \$PRETTY_NAME"
else
  warn "Cannot detect OS version"
fi

# Step 1: Check NVIDIA drivers
log "Checking NVIDIA drivers..."
if command -v nvidia-smi &>/dev/null; then
  GPU_INFO=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "unknown")
  log "GPU: \$GPU_INFO"
else
  fail "nvidia-smi not found. Install NVIDIA GPU drivers first, then re-run this script."
fi

# Step 2: Install Docker
if command -v docker &>/dev/null; then
  log "Docker already installed: \$(docker --version)"
else
  log "Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  log "Docker installed"
fi
# Always ensure the agent user is in the docker group (idempotent).
# Skipping this when Docker was pre-installed leaves the user unable
# to run docker without sudo, which breaks ssh-in diagnostics — seen
# on a node that had Docker pre-baked from the factory image.
if id -nG "\$AGENT_USER" 2>/dev/null | tr ' ' '\n' | grep -qx docker; then
  log "User \$AGENT_USER already in docker group"
else
  usermod -aG docker "\$AGENT_USER"
  log "Added \$AGENT_USER to docker group (will take effect on next login)"
fi

# Step 3: Install nvidia-container-toolkit
if dpkg -l nvidia-container-toolkit 2>/dev/null | grep -q '^ii'; then
  log "nvidia-container-toolkit already installed"
else
  log "Installing nvidia-container-toolkit..."
  # Remove any broken list file from a prior failed run (prevents apt-get update failure)
  rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

  # GPG keyring (use --yes so re-runs don't prompt to overwrite)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \\
    | gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

  # Unified stable apt list (NVIDIA deprecated the per-distro URLs;
  # newer Ubuntu/Debian versions 404 on those). This list uses \$(ARCH)
  # which apt resolves per host arch, so it works for amd64 and arm64.
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \\
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list

  apt-get update -qq
  apt-get install -y -qq nvidia-container-toolkit
  nvidia-ctk runtime configure --runtime=docker
  systemctl restart docker
  log "nvidia-container-toolkit installed and configured"
fi

# Step 4: Install Node.js 22.x
if command -v node &>/dev/null; then
  NODE_VER=\$(node --version)
  NODE_MAJOR=\${NODE_VER%%.*}
  NODE_MAJOR=\${NODE_MAJOR#v}
  if [ "\$NODE_MAJOR" -ge 20 ]; then
    log "Node.js already installed: \$NODE_VER"
  else
    log "Node.js \$NODE_VER too old, upgrading to 22.x..."
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y -qq nodejs
    log "Node.js installed: \$(node --version)"
  fi
else
  log "Installing Node.js 22.x..."
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -qq nodejs
  log "Node.js installed: \$(node --version)"
fi

# Step 4b: Install Ollama (binary + systemd unit + drop-in override)
if systemctl list-unit-files ollama.service &>/dev/null && systemctl is-enabled ollama &>/dev/null; then
  log "Ollama service already installed"
else
  log "Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  # Drop-in override: run as the agent user, listen on all interfaces, point
  # models at /mnt/tank when the NFS share is mounted (so the cache is shared
  # across the cluster); otherwise fall back to Ollama's default path.
  mkdir -p /etc/systemd/system/ollama.service.d
  OVERRIDE=/etc/systemd/system/ollama.service.d/override.conf
  {
    echo "[Service]"
    echo "User=\${AGENT_USER}"
    echo "Environment=HOME=/home/\${AGENT_USER}"
    echo "Environment=OLLAMA_HOST=0.0.0.0"
    echo "Environment=OLLAMA_MAX_LOADED_MODELS=0"
    if mountpoint -q /mnt/tank; then
      echo "Environment=OLLAMA_MODELS=/mnt/tank/models/ollama"
      mkdir -p /mnt/tank/models/ollama
      chown -R "\${AGENT_USER}":"\${AGENT_USER}" /mnt/tank/models/ollama 2>/dev/null || true
    fi
  } > "\$OVERRIDE"
  systemctl daemon-reload
  systemctl enable ollama
  systemctl restart ollama
  log "Ollama installed and started"
fi

# Step 5: Download agent bundle (arch-specific)
log "Downloading \${BUNDLE_ARCH} agent bundle from \${SERVER_URL}..."
BUNDLE_TMP=\$(mktemp)
HTTP_CODE=\$(curl -sL -o "\$BUNDLE_TMP" -w "%{http_code}" "\${SERVER_URL}/api/agent/bundle?arch=\${BUNDLE_ARCH}")
if [ "\$HTTP_CODE" != "200" ]; then
  fail "Failed to download agent bundle (HTTP \$HTTP_CODE). Is the server running at \${SERVER_URL}?"
fi

# Step 6: Install agent
log "Installing agent to /opt/dgx-agent..."
mkdir -p /opt/dgx-agent
tar -xzf "\$BUNDLE_TMP" -C /opt/dgx-agent/
chown -R "\$AGENT_USER":"\$AGENT_USER" /opt/dgx-agent
rm -f "\$BUNDLE_TMP"

# Step 7: Configure sudoers for agent self-update
SUDOERS_FILE="/etc/sudoers.d/dgx-agent"
if [ ! -f "\$SUDOERS_FILE" ]; then
  log "Configuring sudoers for agent service management..."
  echo "\$AGENT_USER ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart dgx-agent, /usr/bin/systemctl stop dgx-agent" > "\$SUDOERS_FILE"
  chmod 440 "\$SUDOERS_FILE"
fi

# Step 8: Create systemd service
log "Creating systemd service..."
# Extract server host and port from URL
SERVER_HOST=\$(echo "\$SERVER_URL" | sed -E 's|https?://([^:/]+).*|\\1|')
SERVER_PORT=\$(echo "\$SERVER_URL" | sed -E 's|https?://[^:]+:([0-9]+).*|\\1|')
[ -z "\$SERVER_PORT" ] && SERVER_PORT=4000

cat > /etc/systemd/system/dgx-agent.service <<SVCEOF
[Unit]
Description=DGX Manager Agent
After=network.target

[Service]
Type=simple
User=\$AGENT_USER
ExecStart=/usr/bin/node /opt/dgx-agent/dist/index.js
Restart=always
RestartSec=5
Environment=MANAGER_URL=ws://\${SERVER_HOST}:\${SERVER_PORT}/ws/agent
Environment=JOIN_TOKEN=\${JOIN_TOKEN}
Environment=HOME=/home/\${AGENT_USER}
Environment=PATH=/home/\${AGENT_USER}/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable dgx-agent
systemctl restart dgx-agent

log "Agent installed and started!"
log ""
log "The agent will connect to \${SERVER_URL} and register using the join token."
log "Check status: systemctl status dgx-agent"
log "View logs:    journalctl -u dgx-agent -f"
`;
}
