import { sshExec } from "./executor.js";

const AGENT_SERVICE = `[Unit]
Description=DGX Manager Agent
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/node /opt/dgx-agent/dist/index.js
Restart=always
RestartSec=5
Environment=MANAGER_URL=ws://%MANAGER_HOST%:%MANAGER_PORT%/ws/agent
Environment=NODE_ID=%NODE_ID%

[Install]
WantedBy=multi-user.target`;

export async function deployAgent(
  host: string,
  nodeId: string,
  managerHost: string,
  managerPort: number
): Promise<string> {
  const logs: string[] = [];

  // Create agent directory
  const mkdir = await sshExec(host, "sudo mkdir -p /opt/dgx-agent");
  logs.push(mkdir.code === 0 ? "Created /opt/dgx-agent" : `mkdir failed: ${mkdir.stderr}`);

  // Create systemd service file
  const serviceContent = AGENT_SERVICE
    .replace("%MANAGER_HOST%", managerHost)
    .replace("%MANAGER_PORT%", String(managerPort))
    .replace("%NODE_ID%", nodeId);

  const writeService = await sshExec(
    host,
    `echo '${serviceContent}' | sudo tee /etc/systemd/system/dgx-agent.service > /dev/null`
  );
  logs.push(writeService.code === 0 ? "Service file written" : `Service write failed: ${writeService.stderr}`);

  // Enable and start
  const enable = await sshExec(host, "sudo systemctl daemon-reload && sudo systemctl enable dgx-agent && sudo systemctl restart dgx-agent");
  logs.push(enable.code === 0 ? "Agent service started" : `Service start failed: ${enable.stderr}`);

  return logs.join("\n");
}
