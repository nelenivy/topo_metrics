#!/usr/bin/env bash
# Oracle-gap (Algorithms 2 + 3) + consensus modes — same cache layout as run_unsup_eval.
#
# Default cache: OUTPUT_DIR/embedding_cache (same as standalone unsup eval when
# OUTPUT_DIR=./results/unsup_eval). Metrics go to OUTPUT_DIR/METRICS_SUBDIR/.
#
# Detached screen:
#   cd /path/to/mteb_eval && screen -h 50000 -dmS oracle_gap bash scripts/run_oracle_gap_in_screen.sh
#
# One-off with tee:
#   OUTPUT_DIR=./results/unsup_eval TASK_SET=core bash scripts/run_oracle_gap_in_screen.sh 2>&1 | tee run_oracle_gap.log
#
# Parameter sweep (each variant → separate METRICS_SUBDIR so CSVs do not clash):
#   VARIANTS="default kn192_r12 grid32" bash scripts/run_oracle_gap_in_screen.sh
#
# Env mirrors run_unsup_eval / run_standard_in_screen where applicable:
#   OUTPUT_DIR, MODEL_SET, TASK_SET, TASKS (space-separated → --tasks), POOLINGS,
#   DEVICE, BATCH_SIZE, NO_AUTO_EMBEDDING_BATCH, TORCH_DTYPE,
#   R_CONSENSUS, R_PRINCIPAL, KNN_K, BANDWIDTH_GRID_M, PRINCIPAL_MAXITER,
#   MAX_N, MIN_N, PAIR_WORKERS, LAYER_SPEC, METRICS_SUBDIR, VARIANTS
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-${HOME}/.mlspace/envs/embs_aggr/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/unsup_eval}"
mkdir -p "${OUTPUT_DIR}"
METRICS_SUBDIR="${METRICS_SUBDIR:-oracle_gap}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Oracle-gap–specific (tune these for stability / cost trade-offs):
R_CONSENSUS="${R_CONSENSUS:-8}"
R_PRINCIPAL="${R_PRINCIPAL:-8}"
KNN_K="${KNN_K:-128}"
BANDWIDTH_GRID_M="${BANDWIDTH_GRID_M:-24}"
PRINCIPAL_MAXITER="${PRINCIPAL_MAXITER:-2000}"
MAX_N="${MAX_N:-}"
MIN_N="${MIN_N:-40}"
PAIR_WORKERS="${PAIR_WORKERS:-1}"
LAYER_SPEC="${LAYER_SPEC:-last_1}"

# Optional: space-separated list; each run uses METRICS_SUBDIR=oracle_gap_${v}
VARIANTS="${VARIANTS:-default}"

_run_one_variant() {
  local tag="$1"
  local subdir="$2"
  shift 2
  echo "=== oracle_gap variant=${tag} metrics_subdir=${subdir} ===" | tee -a "${OUTPUT_DIR}/oracle_gap_run.log"
  date -u +"%Y-%m-%dT%H:%M:%SZ start ${tag}" | tee -a "${OUTPUT_DIR}/oracle_gap_run.log"

  local -a ARGS=(
    -u scripts/run_oracle_gap_consensus.py
    --model-set "${MODEL_SET:-core}"
    --task-set "${TASK_SET:-core}"
    --poolings ${POOLINGS:-mean}
    --layer-spec "${LAYER_SPEC}"
    --output-dir "${OUTPUT_DIR}"
    --metrics-subdir "${subdir}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --r-consensus "${R_CONSENSUS}"
    --r-principal "${R_PRINCIPAL}"
    --knn-k "${KNN_K}"
    --bandwidth-grid-m "${BANDWIDTH_GRID_M}"
    --principal-maxiter "${PRINCIPAL_MAXITER}"
    --min-n "${MIN_N}"
    --pair-workers "${PAIR_WORKERS}"
    -v "${VERBOSE:-1}"
  )
  if [[ -n "${TASKS:-}" ]]; then
    # shellcheck disable=SC2206
    ARGS+=(--tasks ${TASKS})
  fi
  if [[ -n "${MAX_N:-}" ]]; then
    ARGS+=(--max-n "${MAX_N}")
  fi
  if [[ -n "${MAX_SAMPLES:-}" ]]; then
    ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi
  if [[ -n "${TORCH_DTYPE:-}" ]]; then
    ARGS+=(--torch-dtype "${TORCH_DTYPE}")
  fi
  if [[ -n "${NO_AUTO_EMBEDDING_BATCH:-}" ]]; then
    ARGS+=(--no-auto-embedding-batch)
  fi
  if [[ -n "${NO_TRUST_REMOTE_CODE:-}" ]]; then
    ARGS+=(--no-trust-remote-code)
  fi
  if [[ -n "${SKIP_ALG2:-}" ]]; then
    ARGS+=(--skip-alg2)
  fi
  if [[ -n "${SKIP_ALG3:-}" ]]; then
    ARGS+=(--skip-alg3)
  fi

  "${PYTHON}" "${ARGS[@]}" "$@" 2>&1 | tee -a "${OUTPUT_DIR}/oracle_gap_run.log"
}

for v in ${VARIANTS}; do
  case "${v}" in
    default)
      _run_one_variant "${v}" "${METRICS_SUBDIR}"
      ;;
    kn192_r12)
      _run_one_variant "${v}" "${METRICS_SUBDIR}_kn192_r12" \
        --knn-k 192 --r-consensus 12 --r-principal 12
      ;;
    kn256_r16)
      _run_one_variant "${v}" "${METRICS_SUBDIR}_kn256_r16" \
        --knn-k 256 --r-consensus 16 --r-principal 16
      ;;
    grid48)
      _run_one_variant "${v}" "${METRICS_SUBDIR}_grid48" \
        --bandwidth-grid-m 48
      ;;
    iters4000)
      _run_one_variant "${v}" "${METRICS_SUBDIR}_iters4000" \
        --principal-maxiter 4000
      ;;
    *)
      echo "Unknown VARIANT=${v} (add a case in run_oracle_gap_in_screen.sh)" >&2
      exit 2
      ;;
  esac
done

echo "All variants done. Logs: ${OUTPUT_DIR}/oracle_gap_run.log"
