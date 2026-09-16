/**
 * Backfill BenchmarkRun.nullCompletions for accuracy runs finished before the
 * field existed (#20 §2).
 *
 * The count is recoverable because every run's log is kept at
 * $SHARED_STORAGE/logs/benchmarks/<runId>.log, and lm-eval logs one warning per
 * empty completion. Without this, every historical accuracy number keeps
 * rendering as clean — including a published GPQA-Diamond 81.31 that was really
 * a floor with 11.6% of items unscored.
 *
 * Idempotent and re-runnable: it only writes rows whose count is currently null,
 * and skips runs whose log is missing rather than writing a 0 (which would
 * assert we checked when we could not).
 *
 *   DATABASE_URL="file:./prisma/dev.db" npx tsx scripts/backfill-null-completions.ts [--apply]
 *
 * Without --apply it reports what it would change and writes nothing.
 */
import { readFileSync } from "node:fs";
import { prisma } from "../packages/server/src/prisma.js";
import { benchmarkLogPath } from "../packages/server/src/benchmarks/execute.js";
import {
  countNullCompletions,
  nullCompletionFindingFor,
} from "../packages/server/src/benchmarks/null-completions.js";

const apply = process.argv.includes("--apply");

async function main(): Promise<void> {
  const runs = await prisma.benchmarkRun.findMany({
    where: { kind: "accuracy", nullCompletions: null },
    select: { id: true, accuracyScore: true, accuracyMetrics: true, status: true, presetId: true },
    orderBy: { createdAt: "asc" },
  });

  let missingLog = 0;
  let clean = 0;
  const affected: { id: string; preset: string; count: number; severity: string; uplift: string }[] = [];

  for (const r of runs) {
    let text: string;
    try {
      text = readFileSync(benchmarkLogPath(r.id), "utf8");
    } catch {
      missingLog++;
      continue;
    }
    const count = countNullCompletions(text);
    if (apply) {
      await prisma.benchmarkRun.update({ where: { id: r.id }, data: { nullCompletions: count } });
    }
    if (count === 0) {
      clean++;
      continue;
    }
    const f = nullCompletionFindingFor(count, r.accuracyScore, r.accuracyMetrics);
    affected.push({
      id: r.id,
      preset: r.presetId ?? "(custom)",
      count,
      severity: f?.severity ?? "unknown",
      uplift: f?.maxUpliftPoints != null ? `${f.maxUpliftPoints.toFixed(2)} pts` : "unknown",
    });
  }

  affected.sort((a, b) => b.count - a.count);
  console.log(`${apply ? "APPLIED" : "DRY RUN"} — ${runs.length} accuracy runs without a count`);
  console.log(`  ${clean} had no empty completions`);
  console.log(`  ${missingLog} skipped (log missing — left null, NOT zeroed)`);
  console.log(`  ${affected.length} had empty completions:\n`);
  for (const a of affected) {
    console.log(
      `    ${a.id}  ${String(a.count).padStart(4)} empty  ${a.severity.padEnd(8)} max uplift ${a.uplift}   ${a.preset}`,
    );
  }
  if (!apply && affected.length > 0) console.log("\n  re-run with --apply to write these counts");
}

main()
  .catch((e) => {
    console.error(e);
    process.exitCode = 1;
  })
  .finally(() => prisma.$disconnect());
