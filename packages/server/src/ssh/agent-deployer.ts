import { sshExec } from "./executor.js";
import { broadcast as sseBroadcast } from "../sse.js";

const NFS_AGENT_PATH =
  process.env.AGENT_SOURCE_PATH ||
  "/mnt/tank/src/github/dgx-manager/packages/agent";

const AGENT_SERVICE = `[Unit]
Description=DGX Manager Agent
After=network.target

[Service]
Type=simple
User=%SSH_USER%
ExecStart=/usr/bin/node /opt/dgx-agent/dist/index.js
Restart=always
RestartSec=5
Environment=MANAGER_URL=ws://%MANAGER_HOST%:%MANAGER_PORT%/ws/agent
Environment=NODE_ID=%NODE_ID%
Environment=HF_HOME=/mnt/tank/models
Environment=VLLM_REPO_PATH=/mnt/tank/src/github/spark-vllm-docker
Environment=HOME=/home/%SSH_USER%
Environment=PATH=/home/%SSH_USER%/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

[Install]
WantedBy=multi-user.target`;

export async function deployAgent(
  host: string,
  nodeId: string,
  managerHost: string,
  managerPort: number
): Promise<string> {
  const logs: string[] = [];

  const emit = (step: string, status: string, detail?: string) => {
    sseBroadcast({ type: "node:provision", payload: { nodeId, step, status, detail } });
  };

  const sshUser = process.env.SSH_USER || process.env.USER || "ubuntu";

  // Create agent directory
  emit("Create directory", "running");
  const mkdir = await sshExec(host, `sudo mkdir -p /opt/dgx-agent && sudo chown -R ${sshUser}:${sshUser} /opt/dgx-agent`);
  logs.push(mkdir.code === 0 ? "Created /opt/dgx-agent" : `mkdir failed: ${mkdir.stderr}`);
  emit("Create directory", mkdir.code === 0 ? "done" : "failed", logs[logs.length - 1]);

  // Sync agent code from NFS mount
  emit("Sync agent code", "running", "Copying built agent from NFS...");
  const sync = await sshExec(
    host,
    `sudo rsync -a --delete ${NFS_AGENT_PATH}/dist/ /opt/dgx-agent/dist/ && ` +
    `sudo cp ${NFS_AGENT_PATH}/package.json /opt/dgx-agent/package.json && ` +
    `sudo cp -r ${NFS_AGENT_PATH}/node_modules/ /opt/dgx-agent/node_modules/`,
    { timeout: 60_000 }
  );
  if (sync.code === 0) {
    emit("Sync agent code", "done", "Agent code synced from NFS");
    logs.push("Agent code synced from NFS");
  } else {
    emit("Sync agent code", "failed", sync.stderr.slice(0, 200));
    logs.push(`Sync failed: ${sync.stderr}`);
  }

  // Create systemd service file
  emit("Service file", "running");
  const serviceContent = AGENT_SERVICE
    .replace(/%MANAGER_HOST%/g, managerHost)
    .replace(/%MANAGER_PORT%/g, String(managerPort))
    .replace(/%NODE_ID%/g, nodeId)
    .replace(/%SSH_USER%/g, sshUser);

  const writeService = await sshExec(
    host,
    `echo '${serviceContent}' | sudo tee /etc/systemd/system/dgx-agent.service > /dev/null`
  );
  logs.push(writeService.code === 0 ? "Service file written" : `Service write failed: ${writeService.stderr}`);
  emit("Service file", writeService.code === 0 ? "done" : "failed", logs[logs.length - 1]);

  // Enable and restart
  emit("Restart service", "running");
  const enable = await sshExec(
    host,
    "sudo systemctl daemon-reload && sudo systemctl enable dgx-agent && sudo systemctl restart dgx-agent"
  );
  logs.push(enable.code === 0 ? "Agent service restarted" : `Service start failed: ${enable.stderr}`);
  emit("Restart service", enable.code === 0 ? "done" : "failed", logs[logs.length - 1]);

  if (enable.code === 0) {
    emit("Agent deploy complete", "done", "Agent will reconnect with new version");
  }

  return logs.join("\n");
}
