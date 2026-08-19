#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

if [[ ! -r "${MODEL_PATH}" ]]; then
    echo "ERROR: ONNX model not readable: ${MODEL_PATH}" >&2
    exit 1
fi

cache_path="${CALIBRATION_CACHE:-${RESULTS_DIR}/calibration_cache_random.bin}"
range_min="${CALIBRATION_MIN:--1}"
range_max="${CALIBRATION_MAX:-1}"
batches="${CALIBRATION_BATCHES:-32}"
seed="${CALIBRATION_SEED:-42}"
workspace_mib="${WORKSPACE_MIB:-2048}"

mkdir -p "$(dirname "${cache_path}")"

args=(
    "${SCRIPT_DIR}/generate_dummy_calibration_cache.py"
    "--onnx" "${MODEL_PATH}"
    "--cache" "${cache_path}"
    "--min" "${range_min}"
    "--max" "${range_max}"
    "--batches" "${batches}"
    "--seed" "${seed}"
    "--workspace-mib" "${workspace_mib}"
    "--force"
)

if [[ -n "${CALIBRATION_INPUT_SHAPES:-}" ]]; then
    IFS=',' read -r -a shape_items <<<"${CALIBRATION_INPUT_SHAPES}"
    for item in "${shape_items[@]}"; do
        [[ -n "${item}" ]] || continue
        args+=("--shape" "${item}")
    done
fi

echo "Generating random INT8 calibration cache"
echo "  model:  ${MODEL_PATH}"
echo "  cache:  ${cache_path}"
echo "  range:  [${range_min}, ${range_max}]"
echo "  batches:${batches}"

python3 "${args[@]}"

echo "Done: ${cache_path}"
echo "Build INT8 with: CALIBRATION_CACHE=${cache_path} make int8"
