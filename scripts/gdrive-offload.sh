#!/usr/bin/env bash
# Offload per user's plan:
#   - outputs*  -> Drive, then DELETE local (frees space)
#   - logs      -> Drive, KEPT local (backup)
#   - model + logprobs cache -> EXCLUDED, stay local, untouched
# Safe: nothing deleted unless upload AND checksum-verify both pass.
# Fix: no grep-in-pipe (that masked exit codes); --log-level ERROR hides notices.
set -uo pipefail
export HOME=/home/hendrik-n PATH=/home/hendrik-n/.local/bin:$PATH
cd /home/kolin/knowledge-distillation-quantization || exit 2

REMOTE="gdrive:kd-backup"
RC="rclone --log-level ERROR --stats 30s --stats-one-line --stats-log-level NOTICE --drive-chunk-size 128M --transfers 8 --checkers 16"

# Archived to Drive AND deleted locally (reclaims space)
OFFLOAD=(
  "outputs/final_last"
  "outputs/train_details"
  "outputs_3b"
  "outputs_activation_probe"
  "outputs_evalplus"
  "outputs_evalplus_reasoning"
  "outputs_evalplus_reasoning_smoke"
  "outputs_evalplus_smoke"
  "outputs_fq"
  "outputs_general_v2"
  "outputs_qat_base"
)
# Backed up to Drive but KEPT locally.
# outputs/eval holds every per-problem eval record (~3.5G). It is not a regenerable artifact:
# rebuilding it means re-running every eval in the project, and every significance number in
# notes/significance_tests.md is computed from it. Offloading it on 2026-08-02 silently broke
# the verified-116 re-run on 2026-08-22 until it was pulled back. Keep it local.
BACKUP=( "logs" "sparse_logs" "outputs/eval" )

free_now(){ df -h / | awk 'NR==2{print $4" free ("$5" used)"}'; }
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "START | model + logprobs cache EXCLUDED (stay local) | $(free_now)"

log "### PHASE 1: training logs -> Drive (backup, kept local)"
for item in "${BACKUP[@]}"; do
  [ -e "$item" ] || { log "  SKIP $item (absent)"; continue; }
  if $RC copy "$item" "$REMOTE/$item" && $RC check "$item" "$REMOTE/$item" --one-way; then
    log "  OK backed up $item (kept local)"
  else
    log "  !! backup issue on $item (local kept)"
  fi
done

log "### PHASE 2: outputs -> Drive (upload, verify, delete local)"
for item in "${OFFLOAD[@]}"; do
  [ -e "$item" ] || { log "  SKIP $item (absent)"; continue; }
  sz=$(du -sh "$item" 2>/dev/null | cut -f1)
  echo "=================================================================="
  log "OFFLOAD $item ($sz) -> $REMOTE/$item"
  log "  upload..."
  if ! $RC copy "$item" "$REMOTE/$item"; then
    log "  !! UPLOAD FAILED $item — STOPPING. Nothing deleted."; exit 1
  fi
  log "  verify (checksums, one-way)..."
  if ! $RC check "$item" "$REMOTE/$item" --one-way; then
    log "  !! VERIFY FAILED $item — STOPPING. Local kept."; exit 1
  fi
  log "  verified OK -> deleting local copy"
  rm -rf "$item"
  log "  DONE $item | $(free_now)"
done
echo "=================================================================="
log "ALL DONE | $(free_now)"
