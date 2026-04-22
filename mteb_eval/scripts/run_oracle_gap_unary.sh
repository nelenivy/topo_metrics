#!/usr/bin/env bash
# Unary oracle-gap runner (per-model vs ensemble consensus).
#
# Writes separate artifacts from pairwise (see run_oracle_gap_unary_consensus.py):
#   oracle_gap_unary*.csv, diagnostics/oracle_gap_unary_*.csv, run_info_unary.json
# Log: OUTPUT_DIR/oracle_gap_unary_run.log
#
# Reuses the same env knobs as scripts/run_oracle_gap.sh where applicable, plus:
#   CONSENSUS_KNN_K (default 15), USE_FIEDLER_WEIGHTS (0 → --no-use-fiedler-weights),
#   DENSITY_NORMALIZE, DENSITY_ALPHA (same fiber W rescale as pairwise).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-${HOME}/.mlspace/envs/metrics/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/unsup_eval}"
mkdir -p "${OUTPUT_DIR}"
METRICS_SUBDIR="${METRICS_SUBDIR:-oracle_gap}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
R_CONSENSUS="${R_CONSENSUS:-8}"
R_PRINCIPAL="${R_PRINCIPAL:-8}"
KNN_K="${KNN_K:-128}"
CONSENSUS_KNN_K="${CONSENSUS_KNN_K:-15}"
BANDWIDTH_GRID_M="${BANDWIDTH_GRID_M:-24}"
FIBER_KERNEL="${FIBER_KERNEL:-gaussian}"
PRINCIPAL_MAXITER="${PRINCIPAL_MAXITER:-12000}"
MIN_N="${MIN_N:-40}"
MAX_N="${MAX_N:-}"
PAIR_WORKERS="${PAIR_WORKERS:-1}"  # unused by unary script; kept for env parity
LAYER_SPEC="${LAYER_SPEC:-last_1}"

SIGMA_CLIP="${SIGMA_CLIP:-}"
SMOOTH_BW="${SMOOTH_BW:-1}"
DENSITY_NORMALIZE="${DENSITY_NORMALIZE:-1}"
DENSITY_ALPHA="${DENSITY_ALPHA:-}"
LOG_LEVEL="${LOG_LEVEL:-}"
PROGRESS="${PROGRESS:-1}"

_run_unary() {
  local subdir="$1"
  shift
  echo "=== oracle_gap_unary metrics_subdir=${subdir} ===" | tee -a "${OUTPUT_DIR}/oracle_gap_unary_run.log"
  date -u +"%Y-%m-%dT%H:%M:%SZ start unary" | tee -a "${OUTPUT_DIR}/oracle_gap_unary_run.log"

  local -a VERBOSITY=()
  if [[ -n "${LOG_LEVEL:-}" ]]; then
    VERBOSITY=(--log-level "${LOG_LEVEL}")
  else
    case "${VERBOSE:-1}" in
      ''|0) ;;
      1) VERBOSITY=(-v) ;;
      2) VERBOSITY=(-vv) ;;
      *) VERBOSITY=(-vvv) ;;
    esac
  fi

  local -a ARGS=(
    -u scripts/run_oracle_gap_unary_consensus.py
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
    --consensus-knn-k "${CONSENSUS_KNN_K}"
    --bandwidth-grid-m "${BANDWIDTH_GRID_M}"
    --fiber-kernel "${FIBER_KERNEL}"
    --principal-maxiter "${PRINCIPAL_MAXITER}"
    --min-n "${MIN_N}"
    "${VERBOSITY[@]}"
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
  if [[ -n "${REUSE_RUN_DIR:-}" ]]; then
    ARGS+=(--reuse-run-dir "${REUSE_RUN_DIR}")
  fi
  if [[ -n "${NO_INCREMENTAL_CSV:-}" ]]; then
    ARGS+=(--no-incremental-csv)
  fi
  if [[ -n "${SIGMA_CLIP:-}" ]]; then
    ARGS+=(--sigma-clip "${SIGMA_CLIP}")
  fi
  if [[ "${SMOOTH_BW:-1}" == "0" ]]; then
    ARGS+=(--no-smooth-bw)
  fi
  if [[ -n "${DENSITY_ALPHA:-}" ]]; then
    ARGS+=(--density-alpha "${DENSITY_ALPHA}")
  fi
  if [[ "${DENSITY_NORMALIZE:-1}" == "0" ]]; then
    ARGS+=(--no-density-normalize)
  fi
  if [[ "${PROGRESS:-1}" == "0" ]]; then
    ARGS+=(--no-progress)
  fi
  if [[ "${USE_FIEDLER_WEIGHTS:-1}" == "0" ]]; then
    ARGS+=(--no-use-fiedler-weights)
  fi

  "${PYTHON}" "${ARGS[@]}" "$@" 2>&1 | tee -a "${OUTPUT_DIR}/oracle_gap_unary_run.log"
}

_run_unary "${METRICS_SUBDIR}" "$@"
echo "Unary run log: ${OUTPUT_DIR}/oracle_gap_unary_run.log"
