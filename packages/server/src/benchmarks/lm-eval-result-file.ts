import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const RESULTS_RE = /^results_.*\.json$/;

// lm-eval writes results to <output_path>/<sanitized-model>/results_<ts>.json
// (nested), not a fixed path. Recursively find every results_*.json under
// outputDir and return the newest by mtime, or null if none exists.
export function findLmEvalResultFile(outputDir: string): string | null {
  let best: string | null = null;
  let bestMtime = -Infinity;

  const walk = (dir: string) => {
    let entries: import("node:fs").Dirent[];
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return; // dir vanished / unreadable — nothing to contribute
    }
    for (const e of entries) {
      const full = join(dir, e.name);
      if (e.isDirectory()) {
        walk(full);
      } else if (RESULTS_RE.test(e.name)) {
        const m = statSync(full).mtimeMs;
        if (m > bestMtime) {
          bestMtime = m;
          best = full;
        }
      }
    }
  };

  walk(outputDir);
  return best;
}
