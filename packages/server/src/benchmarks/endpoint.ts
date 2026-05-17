export type EndpointDeployment = {
  port: number | null;
  node: { ipAddress: string | null } | null;
};

export function deploymentEndpointUrl(d: EndpointDeployment): string {
  if (!d.node) throw new Error("deployment.node is required");
  if (!d.node.ipAddress) throw new Error("deployment.node.ipAddress is required");
  if (!d.port) throw new Error("deployment.port is required");
  return `http://${d.node.ipAddress}:${d.port}`;
}
