import { apiFetch } from "./api";

export type ClaudeLaunch = {
  baseUrl: string;
  model: string;
  shells: { bash: string; powershell: string };
};

export function fetchClaudeLaunch(deploymentId: string): Promise<ClaudeLaunch> {
  return apiFetch<ClaudeLaunch>(`/api/deployments/${deploymentId}/claude-launch`);
}
