#!/usr/bin/env bash
# Fiber-covariance pairwise metrics (same adaptive ``T`` as ``run_oracle_gap.sh`` / pairwise).
#
# Outputs (separate from global oracle_gap / unary):
#   OUTPUT_DIR/METRICS_SUBDIR/oracle_gap_fiber_pairwise*.csv
#   OUTPUT_DIR/METRICS_SUBDIR/oracle_gap_fiber_local_stats*.csv
#   OUTPUT_DIR/METRICS_SUBDIR/diagnostics/oracle_gap_fiber_per_pair_diagnostics*.csv
#   OUTPUT_DIR/oracle_gap_fiber_run.log
#
# Shared env knobs with ``run_oracle_gap.sh`` where applicable:
#   ORACLE_GAP_ENV, PYTHON, OUTPUT_DIR, MODEL_SET, TASK_SET, POOLINGS, DEVICE (default cuda:1),
#   BATCH_SIZE, R_PRINCIPAL, KNN_K, BANDWIDTH_GRID_M, SIGMA_CLIP, SMOOTH_BW,
#   DENSITY_NORMALIZE (1=default), DENSITY_ALPHA (optional → --density-alpha),
#   PRINCIPAL_MAXITER, PRINCIPAL_DEVICE, PRINCIPAL_BLAS_THREADS,
#   MAX_N, MIN_N, LAYER_SPEC, REUSE_RUN_DIR, LOG_LEVEL, PROGRESS, VERBOSE, NO_TRUST_REMOTE_CODE
#
# Fiber-specific:
#   FIBER_SCHEDULES — comma list passed as --fiber-schedules (default: all three in Python)
#
# Sweeps (separate METRICS_SUBDIR per variant):
#   VARIANTS="default uniform_only r12 kn48" bash scripts/run_oracle_gap_fiber.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

if [[ -n "${ORACLE_GAP_ENV:-}" && -f "${ORACLE_GAP_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${ORACLE_GAP_ENV}"
fi

export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-${HOME}/.mlspace/envs/metrics/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="python3"
fi

OUTPUT_DIR="${OUTPUT_DIR:-./results/unsup_eval}"
mkdir -p "${OUTPUT_DIR}"
METRICS_SUBDIR_BASE="${METRICS_SUBDIR:-oracle_gap_fiber}"

DEVICE="${DEVICE:-cuda:1}"
BATCH_SIZE="${BATCH_SIZE:-32}"
R_PRINCIPAL="${R_PRINCIPAL:-8}"
KNN_K="${KNN_K:-24}"
BANDWIDTH_GRID_M="${BANDWIDTH_GRID_M:-24}"
SIGMA_CLIP="${SIGMA_CLIP:-}"
SMOOTH_BW="${SMOOTH_BW:-1}"
DENSITY_NORMALIZE="${DENSITY_NORMALIZE:-1}"
DENSITY_ALPHA="${DENSITY_ALPHA:-}"
PRINCIPAL_MAXITER="${PRINCIPAL_MAXITER:-12000}"
PRINCIPAL_DEVICE="${PRINCIPAL_DEVICE:-}"
PRINCIPAL_BLAS_THREADS="${PRINCIPAL_BLAS_THREADS:-}"
FIBER_SCHEDULES="${FIBER_SCHEDULES:-}"
MAX_N="${MAX_N:-}"
MIN_N="${MIN_N:-40}"
LAYER_SPEC="${LAYER_SPEC:-last_1}"
VARIANTS="${VARIANTS:-default}"
LOG_LEVEL="${LOG_LEVEL:-}"
PROGRESS="${PROGRESS:-1}"

_run_one_variant() {
  local tag="$1"
  local subdir="$2"
  shift 2
  echo "=== oracle_gap_fiber variant=${tag} metrics_subdir=${subdir} ===" | tee -a "${OUTPUT_DIR}/oracle_gap_fiber_run.log"
  date -u +"%Y-%m-%dT%H:%M:%SZ start ${tag}" | tee -a "${OUTPUT_DIR}/oracle_gap_fiber_run.log"

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
    -u scripts/run_oracle_gap_fiber_consensus.py
    --model-set "${MODEL_SET:-core}"
    --task-set "${TASK_SET:-core}"
    --poolings ${POOLINGS:-mean}
    --layer-spec "${LAYER_SPEC}"
    --output-dir "${OUTPUT_DIR}"
    --metrics-subdir "${subdir}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --r-principal "${R_PRINCIPAL}"
    --knn-k "${KNN_K}"
    --bandwidth-grid-m "${BANDWIDTH_GRID_M}"
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
  if [[ -n "${PRINCIPAL_DEVICE:-}" ]]; then
    ARGS+=(--principal-device "${PRINCIPAL_DEVICE}")
  fi
  if [[ -n "${PRINCIPAL_BLAS_THREADS:-}" ]]; then
    ARGS+=(--principal-blas-threads "${PRINCIPAL_BLAS_THREADS}")
  fi
  if [[ -n "${FIBER_SCHEDULES:-}" ]]; then
    ARGS+=(--fiber-schedules "${FIBER_SCHEDULES}")
  fi

  "${PYTHON}" "${ARGS[@]}" "$@" 2>&1 | tee -a "${OUTPUT_DIR}/oracle_gap_fiber_run.log"
}

for v in ${VARIANTS}; do
  case "${v}" in
    default)
      _run_one_variant "${v}" "${METRICS_SUBDIR_BASE}"
      ;;
    uniform_only)
      _run_one_variant "${v}" "${METRICS_SUBDIR_BASE}_sched_uniform" --fiber-schedules uniform
      ;;
    whitened_only)
      _run_one_variant "${v}" "${METRICS_SUBDIR_BASE}_sched_whitened" --fiber-schedules whitened
      ;;
    frobenius_only)
      _run_one_variant "${v}" "${METRICS_SUBDIR_BASE}_sched_frobenius" --fiber-schedules frobenius
      ;;
    r12)
      _run_one_variant "${v}" "${METRICS_SUBDIR_BASE}_r12" --r-principal 12
      ;;
    kn48)
      _run_one_variant "${v}" "${METRICS_SUBDIR_BASE}_kn48" --knn-k 48
      ;;
    *)
      echo "Unknown VARIANT=${v} (add a case in scripts/run_oracle_gap_fiber.sh)" >&2
      exit 2
      ;;
  esac
done

echo "Fiber metrics done. Log: ${OUTPUT_DIR}/oracle_gap_fiber_run.log"
