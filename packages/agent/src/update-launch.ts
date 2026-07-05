export interface LaunchDeps {
  bundleUrl: string;
  version: string;
  updaterPath: string;
  nodeIdFile: string;
  tmpPath: string;
  copyFile(src: string, dest: string): void;
  spawnDetached(cmd: string, args: string[]): void;
  lockExists(): boolean;
  makeLock(): void;
}

/** Non-blocking: guard on the in-flight lock, copy the updater to /tmp, spawn it detached. */
export function launchUpdater(d: LaunchDeps): "launched" | "in-flight" {
  if (d.lockExists()) return "in-flight";
  d.makeLock();
  d.copyFile(d.updaterPath, d.tmpPath);
  d.spawnDetached("node", [d.tmpPath, d.bundleUrl, d.version, d.nodeIdFile]);
  return "launched";
}
