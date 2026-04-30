#!/usr/bin/env bash
set -euo pipefail

# Full local_cov_spectrum k sweep, including RankMe. This keeps the spectral
# entropy output and therefore requires the local covariance eigenspectrum.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MTEB_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${MTEB_DIR}/.." && pwd)"

OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/results/gpu_proxy_20260414_192547}"
K_SWEEP="${K_SWEEP:-10 20 30 50 100}"
LOCAL_COV_INVARIANT_MAX_ORDER="${LOCAL_COV_INVARIANT_MAX_ORDER:-3}"
LOCAL_COV_TRANSFORMS="${LOCAL_COV_TRANSFORMS:-rankme ne_sum participation_ratio invariants}"

# Match the cache key from the existing gpu_proxy_20260414_192547 run.
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
POOLINGS="${POOLINGS:-last_token mean cls}"
TASK_SET="${TASK_SET:-standard}"
MODEL_SET="${MODEL_SET:-standard}"

N_SAMPLES="${N_SAMPLES:-1}"
SAMPLE_FRACTION="${SAMPLE_FRACTION:-0.05}"
MIN_SAMPLE_SIZE="${MIN_SAMPLE_SIZE:-100}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
DEVICE="${DEVICE:-cpu}"
LOCAL_COV_DEVICE="${LOCAL_COV_DEVICE:-cuda}"
LAYER_SPEC_WORKERS="${LAYER_SPEC_WORKERS:-16}"

cd "${MTEB_DIR}"

ARGS=(
  --output-dir "${OUTPUT_DIR}"
  --model-set "${MODEL_SET}"
  --task-set "${TASK_SET}"
  --poolings ${POOLINGS}
  --metrics local_cov_spectrum
  --n-samples "${N_SAMPLES}"
  --sample-fraction "${SAMPLE_FRACTION}"
  --min-sample-size "${MIN_SAMPLE_SIZE}"
  --local-cov-n-neighbors ${K_SWEEP}
  --local-cov-invariant-max-order "${LOCAL_COV_INVARIANT_MAX_ORDER}"
  --local-cov-transforms ${LOCAL_COV_TRANSFORMS}
  --local-cov-device "${LOCAL_COV_DEVICE}"
  --torch-dtype "${TORCH_DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --device "${DEVICE}"
  --layer-spec-workers "${LAYER_SPEC_WORKERS}"
  --no-progress-bar
)

if [[ -n "${MODELS:-}" ]]; then
  # Space-separated HF model ids.
  ARGS+=(--models ${MODELS})
fi

if [[ -n "${TASKS:-}" ]]; then
  # Space-separated MTEB task names.
  ARGS+=(--tasks ${TASKS})
fi

if [[ -n "${TASK_TYPES:-}" ]]; then
  ARGS+=(--task-types ${TASK_TYPES})
fi

echo "Running full local_cov_spectrum k sweep with RankMe"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "K_SWEEP=${K_SWEEP}"
echo "LOCAL_COV_TRANSFORMS=${LOCAL_COV_TRANSFORMS}"
echo "LOCAL_COV_DEVICE=${LOCAL_COV_DEVICE}"

python scripts/run_unsup_eval.py "${ARGS[@]}" "$@"
