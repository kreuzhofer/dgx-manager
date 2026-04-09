/** Shared storage path — the NFS mount point on the host. */
export const SHARED_STORAGE = process.env.SHARED_STORAGE_PATH || "/mnt/tank";

/** Container-internal mount point for shared storage. */
export const WORKSPACE = "/workspace";

/** Translate a host path to a container path. */
export function toContainerPath(hostPath: string): string {
  return hostPath.replace(`${SHARED_STORAGE}/`, `${WORKSPACE}/`);
}

/** Translate a container path to a host path. */
export function toHostPath(containerPath: string): string {
  return containerPath.replace(`${WORKSPACE}/`, `${SHARED_STORAGE}/`);
}
