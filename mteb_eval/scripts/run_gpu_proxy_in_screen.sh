#!/usr/bin/env bash
# GPU-proxy evaluation (--mteb-gpu-proxy): retrieval/reranking plus classification.
#
# Default: automatic embedding shard GPUs from --device + automatic batch bump.
#
# Detached screen (recommended — attach with: screen -r unsup_gpu_proxy):
#   cd /path/to/unsup_eval && screen -h 50000 -dmS unsup_gpu_proxy bash -c '
#     RUN=./results/gpu_proxy_$(date +%Y%m%d_%H%M%S) && mkdir -p "$RUN" &&
#     OUTPUT_DIR="$RUN" TASK_SET=standard EMBEDDING_PRECOMPUTE_DEVICES=cuda:0,cuda:1 DEVICE=cuda:0 \
#     BATCH_SIZE=1024 NO_AUTO_EMBEDDING_BATCH=1 bash scripts/run_gpu_proxy_in_screen.sh 2>&1 | tee "$RUN/run.stdout.log"
#   '
#
# Manual shard GPUs + fixed batch (no auto bump), fresh dir example:
#   OUTPUT_DIR=./results/gpu_proxy_manual_$(date +%Y%m%d_%H%M%S) \
#   EMBEDDING_PRECOMPUTE_DEVICES=cuda:0,cuda:1 DEVICE=cuda:0 BATCH_SIZE=1024 \
#   NO_AUTO_EMBEDDING_BATCH=1 bash scripts/run_gpu_proxy_in_screen.sh
#
# Automatic only: base batch before heuristic (default 32):
#   BATCH_SIZE=64 bash scripts/run_gpu_proxy_in_screen.sh
#
# Automatic only: all visible GPUs (device must be plain ``cuda``):
#   DEVICE=cuda bash scripts/run_gpu_proxy_in_screen.sh
#
# Tail JSON progress:
#   watch -n2 "cat results/gpu_proxy_*/run_progress.json 2>/dev/null | tail -1"
#
# Notes:
#   - --layer-spec-workers 1 keeps proxy scoring on CUDA; >1 forces CPU for proxy.
#   - Set EMBEDDING_PRECOMPUTE_DEVICES to pass --embedding-precompute-devices (manual list).
#   - Set NO_AUTO_EMBEDDING_BATCH=1 to pass --no-auto-embedding-batch (exact --batch-size).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

export PYTHONUNBUFFERED=1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-./results/gpu_proxy_${RUN_ID}}"
mkdir -p "${OUTPUT_DIR}"

PYTHON="${PYTHON:-${HOME}/.mlspace/envs/embs_aggr/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

TASK_SET="${TASK_SET:-standard}"

# Explicit shard list: default probe/store device to cuda:0 (override with DEVICE=...).
if [[ -n "${EMBEDDING_PRECOMPUTE_DEVICES:-}" ]]; then
  DEVICE="${DEVICE:-cuda:0}"
else
  DEVICE="${DEVICE:-cuda}"
fi
BATCH_SIZE="${BATCH_SIZE:-32}"

MODE="auto GPU + auto batch"
if [[ -n "${EMBEDDING_PRECOMPUTE_DEVICES:-}" ]]; then
  MODE="manual shard GPUs (${EMBEDDING_PRECOMPUTE_DEVICES})"
fi
if [[ -n "${NO_AUTO_EMBEDDING_BATCH:-}" ]]; then
  MODE="${MODE}; fixed batch ${BATCH_SIZE}"
fi
echo "GPU-proxy eval (${MODE}; task-set ${TASK_SET}) → ${OUTPUT_DIR}" | tee "${OUTPUT_DIR}/run_info.txt"
date -u +"%Y-%m-%dT%H:%M:%SZ start" | tee -a "${OUTPUT_DIR}/run_info.txt"

ARGS=(
  -u scripts/run_unsup_eval.py
  --model-set "${MODEL_SET:-standard}"
  --task-set "${TASK_SET}"
  --poolings ${POOLINGS:-last_token mean cls}
  --layer-spec-workers "${LAYER_SPEC_WORKERS:-1}"
  --mteb-gpu-proxy
  --mteb-proxy-mem-fraction "${MTEB_PROXY_MEM_FRAC:-0.72}"
  --ripser-maxdim "${RIPSER_MAXDIM:-0}"
  --output-dir "${OUTPUT_DIR}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
)
if [[ -n "${EMBEDDING_PRECOMPUTE_DEVICES:-}" ]]; then
  ARGS+=(--embedding-precompute-devices "${EMBEDDING_PRECOMPUTE_DEVICES}")
fi
if [[ -n "${NO_AUTO_EMBEDDING_BATCH:-}" ]]; then
  ARGS+=(--no-auto-embedding-batch)
fi

exec "${PYTHON}" "${ARGS[@]}"
