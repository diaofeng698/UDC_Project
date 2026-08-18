#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

profile_json="${PROFILE_JSON:-}"
if [[ -z "${profile_json}" ]]; then
    profile_json="$(find "${RESULTS_DIR}" -maxdepth 1 -type f -name 'benchmark_*_profile.json' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2-)"
fi

if [[ -z "${profile_json}" || ! -s "${profile_json}" ]]; then
    echo "ERROR: no TensorRT profile JSON found. Run make benchmark or set PROFILE_JSON=/path/profile.json." >&2
    exit 1
fi

base="${profile_json%_profile.json}"
if [[ "${base}" == "${profile_json}" ]]; then
    base="${profile_json%.json}"
fi
layer_info="${LAYER_INFO_JSON:-${base}_layers.json}"
output="${PROFILE_SUMMARY:-${base}_summary.md}"

args=(--profile "${profile_json}" --output "${output}")
if [[ -s "${layer_info}" ]]; then
    args+=(--layer-info "${layer_info}")
fi

python3 "${SCRIPT_DIR}/summarize_profile.py" "${args[@]}"