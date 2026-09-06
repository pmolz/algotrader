#!/usr/bin/env bash
# Show (and optionally install) the crontab entries for the overnight loop.
#
#   bash scripts/install_cron.sh              # dry run: print what it would add
#   bash scripts/install_cron.sh --install    # actually edit your crontab
#   bash scripts/install_cron.sh --uninstall  # remove the entries again
#
# Defaults: session at 22:00, safety-net report at 07:30, weekly research pass
# Sunday 06:00. Override with NIGHTLY_CRON / REPORT_CRON / RESEARCH_CRON.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MARKER="# algotrader-nightly"
NIGHTLY_CRON="${NIGHTLY_CRON:-0 22 * * *}"
REPORT_CRON="${REPORT_CRON:-30 7 * * *}"
# Weekly, not nightly: the idea supply is slow-moving, and every candidate coded
# raises n_trials for everything else in the log. Sunday morning so the briefs
# are reviewable before the week's sessions run.
RESEARCH_CRON="${RESEARCH_CRON:-0 6 * * 0}"

PY="$REPO/.venv/bin/python"
[[ -x "$PY" ]] || PY="python3"

read -r -d '' BLOCK <<EOF || true
$MARKER (managed by scripts/install_cron.sh — do not edit by hand)
$NIGHTLY_CRON cd $REPO && bash scripts/nightly_cron.sh $MARKER
$REPORT_CRON cd $REPO && $PY scripts/morning_report.py >/dev/null 2>&1 $MARKER
$RESEARCH_CRON cd $REPO && bash scripts/research_briefs.sh >/dev/null 2>&1 $MARKER
EOF

current="$(crontab -l 2>/dev/null || true)"
stripped="$(printf '%s\n' "$current" | grep -vF "$MARKER" || true)"

case "${1:-}" in
  --install)
    printf '%s\n' "$stripped" | sed '/^$/d' > /tmp/algotrader_cron.$$
    printf '%s\n' "$BLOCK" >> /tmp/algotrader_cron.$$
    crontab /tmp/algotrader_cron.$$
    rm -f /tmp/algotrader_cron.$$
    echo "Installed. Current crontab:"
    crontab -l
    ;;
  --uninstall)
    printf '%s\n' "$stripped" | sed '/^$/d' | crontab -
    echo "Removed algotrader entries. Current crontab:"
    crontab -l 2>/dev/null || echo "(empty)"
    ;;
  *)
    echo "Dry run — nothing changed. These lines would be added to your crontab:"
    echo
    printf '%s\n' "$BLOCK"
    echo
    echo "Run with --install to apply, --uninstall to remove."
    echo
    echo "Before installing, check:"
    echo "  * cron is running:      systemctl status cronie   (or cron/crond)"
    echo "  * Docker works as you:  docker info"
    echo "  * the image is built:   bash sandbox/build.sh"
    echo "  * a short session runs: python scripts/run_nightly.py --hours 0.1 --max-iterations 1"
    echo "  * the research pass is sane: bash scripts/research_briefs.sh --dry-run"
    echo "  * the claude CLI is logged in: claude auth status"
    echo
    echo "The research pass writes briefs and stops — it never commits. Review"
    echo "them on Sunday with: git diff -- research/briefs"
    echo
    echo "Note: cron does not fire while the machine is suspended. If this box"
    echo "sleeps at night, use a systemd timer with Persistent=true instead —"
    echo "see docs/NIGHTLY.md."
    ;;
esac
