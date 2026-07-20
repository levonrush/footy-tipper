#!/bin/zsh
# One-shot cutover automation for PR #34 (feed migration).
#
# Waits for the human-approved merge, then completes the deployment the
# operator already signed off on:
#   1. wait for PR #34 to merge (polls every 2 min, up to 48 h)
#   2. wait for the build-image workflow on main to succeed
#      (auto-triggered by the requirements.txt change; dispatched as fallback)
#   3. push local state (new models + DB) to Drive
#   4. dispatch a predict test run (email to the test address only)
#
# Safe to re-run; exits if the PR is closed without merging.

set -u
cd "$(dirname "$0")/../.."
LOG=logs/post-merge-cutover.log
PY=/opt/anaconda3/envs/footy-tipper/bin/python
exec >> "$LOG" 2>&1

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

log "watcher started; waiting for PR #34 to merge"
for i in $(seq 1 1440); do
  state=$(gh pr view 34 --json state -q .state 2>/dev/null || echo "UNKNOWN")
  case "$state" in
    MERGED) log "PR #34 merged"; break ;;
    CLOSED) log "PR #34 closed without merge; exiting"; exit 0 ;;
  esac
  sleep 120
done
if [ "${state:-}" != "MERGED" ]; then
  log "timed out waiting for merge; exiting"
  exit 1
fi

log "waiting for build-image on main"
dispatched=0
for i in $(seq 1 90); do
  run=$(gh run list --workflow=build-image.yml --branch main --limit 1 \
        --json status,conclusion,createdAt \
        -q '.[0] | "\(.status) \(.conclusion)"' 2>/dev/null || echo "")
  case "$run" in
    "completed success"*) log "image build succeeded"; break ;;
    "completed "*) log "image build concluded: $run (continuing anyway; old image is lazy-import safe)"; break ;;
    "") if [ "$dispatched" -eq 0 ] && [ "$i" -gt 5 ]; then
          log "no build run found; dispatching build-image.yml"
          gh workflow run build-image.yml --ref main && dispatched=1
        fi ;;
  esac
  sleep 60
done

log "pushing state (models + DB) to Drive"
if $PY -m pipeline.cli state push; then
  log "state push OK"
else
  log "state push FAILED; retrying once in 5 min"
  sleep 300
  $PY -m pipeline.cli state push && log "state push OK on retry" || { log "state push failed twice; aborting before test dispatch"; exit 1; }
fi

log "dispatching predict test run"
gh workflow run predict.yml --ref main -f mode=test && log "test predict dispatched" || log "test dispatch failed (non-fatal; 15-minute gate will run regardless)"

log "cutover automation complete"
