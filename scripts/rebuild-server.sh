#!/usr/bin/env bash
# Rebuild and restart the manager, but refuse while a benchmark is in flight.
#
# Restarting the manager kills any accuracy run's reasoning proxy, which lives
# inside the manager process on an ephemeral port (#22). The lm-eval job on the
# eval node survives the restart perfectly well — it just has nothing left to
# talk to. Accuracy runs take 2-3 hours, and rebuilding is routine, so the
# window for silently binning hours of GPU time is large.
#
# The server now detects and explains this after the fact, and cancels the
# doomed job instead of letting it burn GPU. This script stops the accidental
# case happening at all.
#
#   ./scripts/rebuild-server.sh            # refuses if a run is in flight
#   ./scripts/rebuild-server.sh --force    # rebuild anyway, killing those runs
#
set -euo pipefail
cd "$(dirname "$0")/.."

API="${API:-http://localhost:4000}"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

busy="$(curl -s --max-time 15 "$API/api/benchmarks" 2>/dev/null | python3 -c '
import sys, json, datetime
try:
    rows = json.load(sys.stdin)
except Exception:
    # Server already down or unreachable: nothing to protect, let the rebuild run.
    sys.exit(0)
rows = rows if isinstance(rows, list) else rows.get("runs", [])
live = [r for r in rows if r.get("status") in ("running", "pending")]
now = datetime.datetime.now(datetime.timezone.utc)
for r in live:
    started = r.get("startedAt") or r.get("createdAt") or ""
    mins = ""
    try:
        t = datetime.datetime.fromisoformat(started.replace("Z", "+00:00"))
        mins = " %d min in" % ((now - t).total_seconds() // 60)
    except Exception:
        pass
    print("  %s  %s  %s%s" % (r.get("id"), r.get("status"), r.get("presetId") or r.get("kind"), mins))
' || true)"

if [ -n "$busy" ]; then
  echo "A benchmark is in flight:"
  echo "$busy"
  echo
  if [ "$FORCE" -ne 1 ]; then
    echo "Refusing to rebuild. An accuracy run loses its reasoning proxy when the"
    echo "manager restarts and cannot be resumed (#22) — the job is cancelled and"
    echo "the run recorded as failed."
    echo
    echo "Wait for it, cancel it deliberately, or re-run with --force."
    exit 1
  fi
  echo "--force given: rebuilding anyway. These runs will be ended and recorded as failed."
fi

exec docker compose up -d --build server
