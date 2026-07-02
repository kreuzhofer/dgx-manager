#!/usr/bin/env bash
# Migrate the dgx-manager SQLite DB to a new manager host, pruning MetricSnapshot
# bloat. Uses SQLite's ONLINE backup (no server downtime) then VACUUM.
#
# Run ON the current manager host (where dgx-manager-server-1 runs).
# Produces /tmp/dgx-data-migrated.db locally, and (if TARGET_HOST set) copies it
# to TARGET_HOST:/tmp/dgx-data-migrated.db for restore into the new dgx-data volume.
#
# Env: TARGET_HOST (optional, e.g. 192.168.44.14)   KEEP_DAYS (default 3)
set -euo pipefail
CONTAINER="${CONTAINER:-dgx-manager-server-1}"
KEEP_DAYS="${KEEP_DAYS:-3}"
OUT=/tmp/dgx-data-migrated.db

echo "[1/4] Online backup + prune (KEEP_DAYS=${KEEP_DAYS}) inside ${CONTAINER}"
cat > /tmp/_migrate_prune.js <<JS
const Database = require("better-sqlite3");
const src = new Database("/app/data/dev.db", { readonly: true });
src.backup("/app/data/_migrate.db").then(() => {
  src.close();
  const db = new Database("/app/data/_migrate.db");
  const before = db.prepare("SELECT count(*) c FROM MetricSnapshot").get().c;
  db.prepare("DELETE FROM MetricSnapshot WHERE timestamp < datetime('now','-${KEEP_DAYS} days')").run();
  db.exec("VACUUM");
  const after = db.prepare("SELECT count(*) c FROM MetricSnapshot").get().c;
  console.log("MetricSnapshot:", before, "->", after,
    "| Node:", db.prepare("SELECT count(*) c FROM Node").get().c,
    "Deployment:", db.prepare("SELECT count(*) c FROM Deployment").get().c,
    "LoadBalancerRule:", db.prepare("SELECT count(*) c FROM LoadBalancerRule").get().c);
  db.close(); process.exit(0);
}).catch(e => { console.error("ERR", e.message); process.exit(1); });
JS
docker cp /tmp/_migrate_prune.js "${CONTAINER}":/app/_migrate_prune.js
docker exec -w /app "${CONTAINER}" node /app/_migrate_prune.js

echo "[2/4] Copy pruned DB out to ${OUT}"
docker cp "${CONTAINER}":/app/data/_migrate.db "${OUT}"
ls -la "${OUT}" | awk '{print "  size:", int($5/1024/1024)" MB"}'

echo "[3/4] Cleanup temp files in container"
docker exec "${CONTAINER}" sh -lc 'rm -f /app/data/_migrate.db /app/_migrate_prune.js' || true
rm -f /tmp/_migrate_prune.js

echo "[4/4] ${TARGET_HOST:+copy to ${TARGET_HOST}}"
if [ -n "${TARGET_HOST:-}" ]; then
  scp -o StrictHostKeyChecking=no "${OUT}" daniel@"${TARGET_HOST}":"${OUT}"
  echo "  -> ${TARGET_HOST}:${OUT} (restore into the new dgx-data volume before first boot)"
fi
