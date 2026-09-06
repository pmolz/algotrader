#!/usr/bin/env bash
# Weekly research pass: ask Claude for new strategy ideas, written as briefs.
#
#   bash scripts/research_briefs.sh              # run it now
#   bash scripts/research_briefs.sh --dry-run    # print the command, change nothing
#
# The standing assignment lives in research/PROMPT.md, not here. This script
# only supplies the environment cron does not: a working directory, a PATH, a
# log, and the guardrails that make an unattended `claude -p` safe to leave
# alone. Keeping the assignment in the repo means a cloud routine can run the
# same brief-writing job from the same file if one is ever set up.
#
# It writes files and STOPS. No commit, no push, no branch. Sunday's briefs sit
# in the working tree until you have read them — a research agent that could
# merge its own ideas into the search would be a research agent with no review
# step, which is the one part of this that has to stay human.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1

# cron's PATH is typically just /usr/bin:/bin; claude and git need more.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:$HOME/.local/bin:$PATH"

LOG_DIR="$REPO/experiments/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/research-$(date +%Y-%m-%d).log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

BRIEFS_DIR="$REPO/research/briefs"
PROMPT_FILE="$REPO/research/PROMPT.md"
TIMEOUT_S="${RESEARCH_TIMEOUT_S:-2400}"     # 40 min; a search that long has stalled
MODEL="${RESEARCH_MODEL:-opus}"
# Scoped so the worst outcome of a bad run is a bad brief, not a bad commit:
# Write cannot leave research/briefs, and Bash can only run the format checks.
ALLOWED="Read Glob Grep WebSearch WebFetch Write(research/briefs/*) Bash(.venv/bin/python -m pytest tests/test_research.py*)"

log "=== research_briefs start (repo=$REPO) ==="

command -v claude >/dev/null 2>&1 || { log "ERROR: claude CLI not on PATH"; exit 1; }
[[ -f "$PROMPT_FILE" ]] || { log "ERROR: $PROMPT_FILE missing — nothing to run"; exit 1; }

# An expired login fails deep inside the run, after the search has been paid
# for. Check it up front, where the log can say so plainly.
if ! claude auth status >/dev/null 2>&1; then
  log "ERROR: claude CLI is not logged in (run: claude auth login)"
  exit 1
fi

# Refuse to start on top of unreviewed briefs. Otherwise a second week's output
# lands next to a first week's, `git diff` stops telling you which is which, and
# the review step quietly becomes guesswork.
if [[ -n "$(git status --porcelain -- research/briefs 2>/dev/null)" ]]; then
  log "SKIP: research/briefs has uncommitted changes — last run's briefs are"
  log "      still unreviewed. Commit or discard them, then run again."
  exit 2
fi

before="$(ls -1 "$BRIEFS_DIR" 2>/dev/null | wc -l)"

# --allowedTools is what makes this unattended: with no TTY, a permission prompt
# is a hang, not a question. Write is scoped to research/briefs and Bash to the
# one test command, so the worst case is a bad brief rather than a bad commit.
read -r -d '' TASK <<'PROMPT' || true
Read research/PROMPT.md and carry out the assignment in it exactly. That file is
authoritative; everything you need is in it.

Constraints for this unattended run:
- Write ONLY to research/briefs/. Do not modify any other file.
- Do not commit, push, create a branch, or open a PR. Leave your work in the
  working tree for a human to review.
- Do not run the agent loop, the nightly session, or any backtest.
- When you finish, print a short summary: which briefs you wrote and why, and
  what you discarded on feasibility grounds.
PROMPT

if [[ "${1:-}" == "--dry-run" ]]; then
  echo "Would run, in $REPO:"
  echo
  echo "  printf '%s' \"\$TASK\" | timeout ${TIMEOUT_S}s claude -p --model $MODEL \\"
  echo "    --allowedTools \"$ALLOWED\""
  echo
  echo "Prompt:"
  printf '%s\n' "$TASK" | sed 's/^/  /'
  exit 0
fi

# The prompt goes in on stdin. --allowedTools is variadic, so a positional
# prompt after it gets eaten as one more tool pattern.
printf '%s' "$TASK" | timeout "${TIMEOUT_S}s" claude -p --model "$MODEL" \
  --allowedTools "$ALLOWED" >>"$LOG" 2>&1
rc=$?

if [[ $rc -eq 124 ]]; then
  log "ERROR: timed out after ${TIMEOUT_S}s — killed. Any briefs it managed to"
  log "       write are still in the working tree; review them as usual."
elif [[ $rc -ne 0 ]]; then
  log "ERROR: claude exited with code $rc (see $LOG)"
fi

after="$(ls -1 "$BRIEFS_DIR" 2>/dev/null | wc -l)"
log "briefs on disk: $before -> $after"

# The validity check is the point of running the tests here: a brief that fails
# to parse is silently skipped by the loop, so a broken one would look exactly
# like a week with no new ideas.
if [[ -x "$REPO/.venv/bin/python" ]]; then
  if "$REPO/.venv/bin/python" -m pytest tests/test_research.py -q >>"$LOG" 2>&1; then
    log "brief format checks passed"
  else
    log "WARN: tests/test_research.py FAILED — at least one brief is malformed"
    log "      and the agent loop will ignore it. See $LOG."
  fi
else
  log "WARN: .venv not found, skipped the brief format checks"
fi

if [[ -n "$(git status --porcelain -- research/briefs 2>/dev/null)" ]]; then
  log "NEW BRIEFS AWAITING REVIEW:"
  git status --porcelain -- research/briefs | tee -a "$LOG"
  log "review with: git diff -- research/briefs   (then commit the ones you want)"
else
  log "no new briefs written this run"
fi

log "=== research_briefs end (rc=$rc) ==="
exit $rc
