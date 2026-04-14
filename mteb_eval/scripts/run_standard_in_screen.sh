#!/usr/bin/env bash
# Standard model/task eval with MANUAL embedding controls:
#   - Explicit GPU list for HDF5 shard precompute (--embedding-precompute-devices)
#   - Fixed batch size 2048 (--batch-size 2048 --no-auto-embedding-batch)
#
# Detached screen (large scrollback):
#   screen -h 50000 -dmS unsup_standard bash scripts/run_standard_in_screen.sh
#
# Override GPUs (comma-separated, order = round-robin shard order):
#   EMBEDDING_PRECOMPUTE_DEVICES=cuda:0,cuda:1 bash scripts/run_standard_in_screen.sh
#
# Main process probe/store device (single GPU index is typical with multi-shard precompute):
#   DEVICE=cuda:0 EMBEDDING_PRECOMPUTE_DEVICES=cuda:0,cuda:1 bash scripts/run_standard_in_screen.sh
#
# Tail progress: watch -n2 cat results/standard/run_progress.json
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-${HOME}/.mlspace/envs/embs_aggr/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

# Manual: which GPUs participate in parallel embedding precompute (2+ → shard workers).
EMBEDDING_PRECOMPUTE_DEVICES="${EMBEDDING_PRECOMPUTE_DEVICES:-cuda:0,cuda:1}"
# Device for model probe / LayerEmbeddingStore forward when not using shard path (and CUDA init).
DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2048}"

exec "${PYTHON}" -u scripts/run_unsup_eval.py \
  --model-set "${MODEL_SET:-standard}" \
  --task-set "${TASK_SET:-standard}" \
  --poolings ${POOLINGS:-last_token mean cls} \
  --layer-spec-workers "${LAYER_SPEC_WORKERS:-4}" \
  --ripser-maxdim "${RIPSER_MAXDIM:-0}" \
  --output-dir "${OUTPUT_DIR:-./results/standard}" \
  --device "${DEVICE}" \
  --embedding-precompute-devices "${EMBEDDING_PRECOMPUTE_DEVICES}" \
  --batch-size "${BATCH_SIZE}" \
  --no-auto-embedding-batch
