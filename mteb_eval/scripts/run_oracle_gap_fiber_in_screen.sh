#!/usr/bin/env bash
# GNU screen launcher for ``run_oracle_gap_fiber.sh`` (non-login bash preserves exports).
#
#   cd mteb_eval
#   export OUTPUT_DIR=... DEVICE=cuda:1
#   bash scripts/run_oracle_gap_fiber_in_screen.sh
#
# Foreground:  bash scripts/run_oracle_gap_fiber_in_screen.sh --foreground
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_SH="${SCRIPT_DIR}/run_oracle_gap_fiber.sh"

if [[ "${1:-}" == "--foreground" || "${1:-}" == "-fg" ]]; then
  exec bash "${RUN_SH}"
fi

NAME="${SCREEN_NAME:-oracle_gap_fiber}"
screen -dmS "${NAME}" bash -c "$(printf 'cd %q && exec bash %q' "${ROOT}" "${RUN_SH}")"
echo "Started screen session: ${NAME}"
echo "Attach: screen -r ${NAME}"
