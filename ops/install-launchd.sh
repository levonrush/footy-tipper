#!/bin/zsh
# Install (or remove) the Footy Tipper launchd schedule on this Mac.
#
# Usage:
#   ./ops/install-launchd.sh            # install/refresh both jobs
#   ./ops/install-launchd.sh uninstall  # remove both jobs
#
# Jobs (local time):
#   com.footytipper.train    Tuesday 06:00  -> footy-tipper train
#   com.footytipper.predict  Thursday 15:00 -> footy-tipper predict
#
# Logs land in <repo>/logs/train.log and <repo>/logs/predict.log.
# Note: launchd skips a run if the Mac is asleep at the trigger time (it does
# not retry like cron's anacron); keep the machine awake around those times.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
AGENTS_DIR="$HOME/Library/LaunchAgents"
LABELS=(com.footytipper.train com.footytipper.predict)

mkdir -p "$AGENTS_DIR" "$REPO_DIR/logs"

if [[ "${1:-install}" == "uninstall" ]]; then
    for label in "${LABELS[@]}"; do
        launchctl bootout "gui/$UID/$label" 2>/dev/null || true
        rm -f "$AGENTS_DIR/$label.plist"
        echo "Removed $label"
    done
    exit 0
fi

for label in "${LABELS[@]}"; do
    src="$REPO_DIR/ops/launchd/$label.plist"
    dst="$AGENTS_DIR/$label.plist"
    sed "s|__REPO__|$REPO_DIR|g" "$src" > "$dst"
    # Refresh: boot out a previous version first (ignore failures on first install).
    launchctl bootout "gui/$UID/$label" 2>/dev/null || true
    launchctl bootstrap "gui/$UID" "$dst"
    echo "Installed $label ($dst)"
done

echo ""
echo "Loaded jobs:"
launchctl list | grep footytipper || true
echo ""
echo "To run one immediately: launchctl kickstart gui/$UID/com.footytipper.predict"
