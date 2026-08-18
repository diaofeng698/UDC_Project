#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

trtexec_bin="$(find_trtexec)"
version="$(trt_version "${trtexec_bin}")"
version="${version:-unknown}"
output_engine="${ENGINE_PATH:-$(engine_path "${version}")}"
timing_cache="${ENGINE_DIR}/timing_cache_trt${version}_$(uname -m).bin"
build_log="${RESULTS_DIR}/build_trt${version}_${PRECISION}.log"
layer_info="${RESULTS_DIR}/layers_trt${version}_${PRECISION}.json"

if [[ ! -r "${MODEL_PATH}" ]]; then
    echo "ERROR: ONNX model not readable: ${MODEL_PATH}" >&2
    exit 1
fi

case "${PRECISION}" in
    fp32) precision_args=() ;;
    fp16) precision_args=(--fp16) ;;
    int8)
        precision_args=(--int8)
        if [[ -n "${CALIBRATION_CACHE:-}" ]]; then
            [[ -r "${CALIBRATION_CACHE}" ]] || {
                echo "ERROR: calibration cache not readable: ${CALIBRATION_CACHE}" >&2
                exit 1
            }
            precision_args+=("--calib=${CALIBRATION_CACHE}")
        else
            echo "INFO: no calibration cache supplied; INT8 requires Q/DQ nodes in the ONNX model."
        fi
        ;;
    *)
        echo "ERROR: PRECISION must be fp32, fp16, or int8; got ${PRECISION}" >&2
        exit 2
        ;;
esac

mkdir -p "${ENGINE_DIR}" "${RESULTS_DIR}"

args=(
    "--onnx=${MODEL_PATH}"
    "--saveEngine=${output_engine}"
    "--memPoolSize=workspace:${WORKSPACE_MIB}MiB"
    "--builderOptimizationLevel=${BUILD_OPT_LEVEL}"
    "--timingCacheFile=${timing_cache}"
    --profilingVerbosity=detailed
    "--exportLayerInfo=${layer_info}"
    --skipInference
)
args+=("${precision_args[@]}")

echo "Building TensorRT ${version} ${PRECISION} engine"
echo "  model:  ${MODEL_PATH}"
echo "  engine: ${output_engine}"
echo "  target: ${TARGET_BOARD}, ${TARGET_SOC}, TensorRT ${TARGET_TENSORRT}"
echo "  note: engines are platform/GPU/TensorRT specific; build this file on the target Orin Nano."

set -o pipefail
"${trtexec_bin}" "${args[@]}" 2>&1 | tee "${build_log}"

if [[ ! -s "${output_engine}" ]]; then
    echo "ERROR: TensorRT reported success but no engine was created." >&2
    exit 1
fi

echo "Engine ready: ${output_engine}"
