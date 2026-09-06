#!/usr/bin/env bash
# Cron entry point: one overnight session, then the morning report.
#
# Cron gives you almost no environment, so everything is made explicit here
# rather than inherited. Safe to run by hand — the session lock makes a
# double-launch a no-op.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# cron's PATH is typically just /usr/bin:/bin; docker and ollama need more.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$PATH"
# Deliberately NOT defaulted. llm.py gives $OLLAMA_HOST precedence over
# agent.host, so setting it here pinned every unattended session to this box and
# disabled the failover that agent.host_preference exists to provide — the
# networked GPU was never once asked to do the work. It is still honoured when
# the caller sets it on purpose, which is the documented override.

# Keep BLAS from oversubscribing every core on a machine you may still be using.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"

LOG_DIR="$REPO/experiments/logs"
mkdir -p "$LOG_DIR"
CRON_LOG="$LOG_DIR/cron-$(date +%Y-%m-%d).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CRON_LOG"; }

if [[ -x "$REPO/.venv/bin/python" ]]; then
  PY="$REPO/.venv/bin/python"
else
  log "WARN: $REPO/.venv not found, falling back to system python3"
  PY="$(command -v python3)"
fi

log "=== nightly_cron start (repo=$REPO) ==="

if ! docker info >/dev/null 2>&1; then
  log "ERROR: Docker unavailable — refusing to run model-written code unsandboxed."
  log "       (is the daemon up, and is $USER in the docker group?)"
  exit 1
fi

# Warm the model so the first iteration isn't billed for the cold load — on
# whichever endpoint the session will ACTUALLY choose. Asking the app rather
# than assuming localhost is the point: a warm-up that hits the wrong box is
# worse than none, because it reports healthy while the real endpoint is cold.
WARM="$(${PY} - <<PYWARM 2>/dev/null
import sys
sys.path.insert(0, "$REPO")
from algotrader.config import get, load_config
from algotrader.agent.llm import parse_endpoints, select_candidates
cfg = load_config()
if (get(cfg, "agent.provider") or "").lower() == "ollama":
    try:
        candidates, _ = select_candidates(cfg, parse_endpoints(cfg))
    except ValueError:
        candidates = []
    for e in candidates:
        print(f"{e.name}\t{e.url}")
PYWARM
)"
if [[ -n "$WARM" ]]; then
  while IFS=$'\t' read -r name url; do
    [[ -z "$url" ]] && continue
    if curl -sf -m 5 "$url/api/tags" >/dev/null 2>&1; then
      log "warmed $name ($url)"
      break
    fi
    log "WARN: Ollama not responding at $name ($url)"
  done <<<"$WARM"
fi

"$PY" scripts/run_nightly.py "$@" >>"$CRON_LOG" 2>&1
rc=$?
case $rc in
  0) log "session finished cleanly" ;;
  2) log "session skipped: another one is still running" ;;
  *) log "session exited with code $rc" ;;
esac

# Report even on a bad exit — an empty/aborted night is exactly what you want to
# be told about in the morning.
"$PY" scripts/morning_report.py >>"$CRON_LOG" 2>&1 \
  && log "morning report written to experiments/reports/" \
  || log "morning report FAILED (see $CRON_LOG)"

log "=== nightly_cron end (rc=$rc) ==="
exit $rc
