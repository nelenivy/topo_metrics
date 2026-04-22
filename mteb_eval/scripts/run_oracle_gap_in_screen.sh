#!/usr/bin/env bash
# Minimal GNU screen launcher for ``run_oracle_gap.sh``.
#
# All Python flags and sweeps live in ``run_oracle_gap.sh`` (and VARIANT cases there).
# This file only starts screen and runs that script from ``mteb_eval/``.
#
# Configure either:
#   A) export variables in your shell before calling this script.  The screen child
#      uses ``bash -c`` (non-login) so those exports reach ``run_oracle_gap.sh``.
#      (A login ``bash -lc`` often dropped job-specific exports in detached screen.)
#      And/or
#   B) point ORACLE_GAP_ENV at a small shell file that only contains ``export …`` lines
#      (sourced by ``run_oracle_gap.sh`` after ``cd``).
#
# Optional (read by ``run_oracle_gap.sh``): LOG_LEVEL=INFO|DEBUG|…, PROGRESS=0 to disable tqdm.
#
# Typical flow
# ------------
#   cd /path/to/mteb_eval
#   export ORACLE_GAP_ENV="$PWD/scripts/oracle_gap_job.env.sh"   # optional
#   bash scripts/run_oracle_gap_in_screen.sh
#
#   screen -r oracle_gap    # default session name; override with SCREEN_NAME=…
#
# Foreground (no screen), for debugging:
#   bash scripts/run_oracle_gap_in_screen.sh --foreground
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SH="${SCRIPT_DIR}/run_oracle_gap.sh"

if [[ "${1:-}" == "--foreground" || "${1:-}" == "-fg" ]]; then
  exec bash "${RUN_SH}"
fi

NAME="${SCREEN_NAME:-oracle_gap}"
screen -dmS "${NAME}" bash -c "$(printf 'cd %q && exec bash %q' "${ROOT}" "${RUN_SH}")"
echo "Started screen session: ${NAME}"
echo "Attach: screen -r ${NAME}"
