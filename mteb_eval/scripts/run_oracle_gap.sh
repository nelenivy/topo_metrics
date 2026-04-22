#!/usr/bin/env bash
# Portable runner for oracle-gap experiments (Algorithms 2 + 3) + consensus modes.
#
# This script is intentionally **not** tied to GNU screen. Run it directly, under tmux,
# systemd, or wrap it with your own process supervisor.
#
# It mirrors the cache layout of ``run_unsup_eval``:
# - default cache: ``OUTPUT_DIR/embedding_cache``
# - metrics output: ``OUTPUT_DIR/METRICS_SUBDIR/``
# - combined log: ``OUTPUT_DIR/oracle_gap_run.log``
#
# Parameter sweep (each variant → separate METRICS_SUBDIR so CSVs do not clash):
#   VARIANTS="default kn192_r12 grid48" bash scripts/run_oracle_gap.sh
#
# Environment knobs (same as the older screen wrapper):
#   ORACLE_GAP_ENV — optional path to a shell file sourced here (export OUTPUT_DIR, … only).
#   LOG_LEVEL — optional: WARNING|INFO|DEBUG|… passed as --log-level (overrides VERBOSE / -v).
#   PROGRESS — set to 0 to pass --no-progress (disables tqdm bars in piped logs).
#   PYTHON, OUTPUT_DIR, MODEL_SET, TASK_SET, TASKS (space-separated → --tasks), POOLINGS,
#   DEVICE, BATCH_SIZE, NO_AUTO_EMBEDDING_BATCH, TORCH_DTYPE,
#   R_CONSENSUS, R_PRINCIPAL, KNN_K, BANDWIDTH_GRID_M,
#   SIGMA_CLIP (adaptive cutoff), SMOOTH_BW (1=on default, 0=--no-smooth-bw),
#   DENSITY_NORMALIZE (1=default directed marginal rescale, 0=--no-density-normalize),
#   DENSITY_ALPHA (optional float → --density-alpha, default 1 in Python if unset),
#   FIBER_KERNEL (kept for CLI compat; adaptive path uses Gaussian+cutoff),
#   PRINCIPAL_MAXITER, PRINCIPAL_DEVICE (e.g. cpu or cuda:0 — unset = cpu SciPy),
#   PRINCIPAL_BLAS_THREADS (optional int for CPU eigsh; 0 = do not pin env),
#   MAX_N, MIN_N, MAX_SAMPLES, PAIR_WORKERS, LAYER_SPEC, METRICS_SUBDIR, VARIANTS,
#   REUSE_RUN_DIR (prior run_unsup_eval dir → use DIR/embedding_cache),
#   NO_INCREMENTAL_CSV (set non-empty to pass --no-incremental-csv),
#   VERBOSE (0/1/2/3+ → maps to no flag / -v / -vv / -vvv),
#   SKIP_ALG2, SKIP_ALG3, NO_TRUST_REMOTE_CODE
#
# Do not pass long ``python … run_oracle_gap_consensus.py`` lines by hand; this script
# builds the Python argv. For screen, use ``scripts/run_oracle_gap_in_screen.sh``.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT}"

# Optional job-specific exports (paths, DEVICE, PRINCIPAL_MAXITER, …):
#   export ORACLE_GAP_ENV=/path/to/oracle_gap_job.env.sh
# Keep sweeps and CLI construction in this script only; env file sets values only.
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
METRICS_SUBDIR="${METRICS_SUBDIR:-oracle_gap}"

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Oracle-gap–specific (tune these for stability / cost trade-offs):
R_CONSENSUS="${R_CONSENSUS:-8}"
R_PRINCIPAL="${R_PRINCIPAL:-8}"
KNN_K="${KNN_K:-24}"
BANDWIDTH_GRID_M="${BANDWIDTH_GRID_M:-24}"
SIGMA_CLIP="${SIGMA_CLIP:-}"
SMOOTH_BW="${SMOOTH_BW:-1}"
DENSITY_NORMALIZE="${DENSITY_NORMALIZE:-1}"
DENSITY_ALPHA="${DENSITY_ALPHA:-}"
FIBER_KERNEL="${FIBER_KERNEL:-gaussian}"
PRINCIPAL_MAXITER="${PRINCIPAL_MAXITER:-12000}"
PRINCIPAL_DEVICE="${PRINCIPAL_DEVICE:-}"
PRINCIPAL_BLAS_THREADS="${PRINCIPAL_BLAS_THREADS:-}"
MAX_N="${MAX_N:-}"
MIN_N="${MIN_N:-40}"
PAIR_WORKERS="${PAIR_WORKERS:-1}"
LAYER_SPEC="${LAYER_SPEC:-last_1}"

# Optional: space-separated list; each run uses METRICS_SUBDIR=oracle_gap_${v}
VARIANTS="${VARIANTS:-default}"
LOG_LEVEL="${LOG_LEVEL:-}"
PROGRESS="${PROGRESS:-1}"

_run_one_variant() {
  local tag="$1"
  local subdir="$2"
  shift 2
  echo "=== oracle_gap variant=${tag} metrics_subdir=${subdir} ===" | tee -a "${OUTPUT_DIR}/oracle_gap_run.log"
  date -u +"%Y-%m-%dT%H:%M:%SZ start ${tag}" | tee -a "${OUTPUT_DIR}/oracle_gap_run.log"

  # argparse: use --log-level when set; else -v/--verbose count (not "-v 1").
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
    --fiber-kernel "${FIBER_KERNEL}"
    --principal-maxiter "${PRINCIPAL_MAXITER}"
    --min-n "${MIN_N}"
    --pair-workers "${PAIR_WORKERS}"
    "${VERBOSITY[@]}"
  )
  if [[ -n "${PRINCIPAL_DEVICE:-}" ]]; then
    ARGS+=(--principal-device "${PRINCIPAL_DEVICE}")
  fi
  if [[ -n "${PRINCIPAL_BLAS_THREADS:-}" ]]; then
    ARGS+=(--principal-blas-threads "${PRINCIPAL_BLAS_THREADS}")
  fi
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
      echo "Unknown VARIANT=${v} (add a case in scripts/run_oracle_gap.sh)" >&2
      exit 2
      ;;
  esac
done

echo "All variants done. Logs: ${OUTPUT_DIR}/oracle_gap_run.log"
