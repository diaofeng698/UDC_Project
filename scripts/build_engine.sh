#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

trtexec_bin="$(find_trtexec)"
version="$(trt_version "${trtexec_bin}")"
version="${version:-unknown}"
output_engine="${ENGINE_PATH:-$(engine_path "${version}")}"
timing_cache="${TIMING_CACHE_FILE:-${ENGINE_DIR}/timing_cache_trt${version}_$(uname -m)_${PRECISION}_ws${WORKSPACE_MIB}MiB.bin}"
build_log="${RESULTS_DIR}/build_trt${version}_${PRECISION}.log"
layer_info="${RESULTS_DIR}/layers_trt${version}_${PRECISION}.json"

if [[ ! -r "${MODEL_PATH}" ]]; then
    echo "ERROR: ONNX model not readable: ${MODEL_PATH}" >&2
    exit 1
fi

case "${PRECISION}" in
    fp32) precision_args=() ;;
    fp16) precision_args=(--fp16) ;;
    int8|int8-fp16)
        precision_args=(--int8)
        if [[ "${PRECISION}" == "int8-fp16" ]]; then
            precision_args+=(--fp16)
        fi
        if [[ -n "${CALIBRATION_CACHE:-}" ]]; then
            [[ -r "${CALIBRATION_CACHE}" ]] || {
                echo "ERROR: calibration cache not readable: ${CALIBRATION_CACHE}" >&2
                exit 1
            }
            precision_args+=("--calib=${CALIBRATION_CACHE}")
        else
            qdq_audit="${RESULTS_DIR}/qdq_audit_$(basename "${MODEL_PATH}" .onnx).json"
            if python3 "${SCRIPT_DIR}/inspect_qdq_onnx.py" \
                --model "${MODEL_PATH}" --output "${qdq_audit}" --require-qdq; then
                echo "INFO: verified explicit Q/DQ model; no legacy calibration cache is used."
            else
                echo "ERROR: INT8 requires representative CALIBRATION_CACHE or a verified explicit Q/DQ ONNX model." >&2
                echo "       Generate one with scripts/quantize_qdq_ptq.py; INT8_EXPLICIT_QDQ no longer bypasses graph validation." >&2
                exit 2
            fi
        fi
        ;;
    *)
        echo "ERROR: PRECISION must be fp32, fp16, int8, or int8-fp16; got ${PRECISION}" >&2
        exit 2
        ;;
esac

mkdir -p "${ENGINE_DIR}" "${RESULTS_DIR}"

args=(
    "--onnx=${MODEL_PATH}"
    "--saveEngine=${output_engine}"
    # trtexec treats a suffix-free memPoolSize value as MiB. The accepted
    # binary suffix is "M", not "MiB"; using "MiB" was parsed as bytes.
    "--memPoolSize=workspace:${WORKSPACE_MIB}"
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
echo "  timing cache: ${timing_cache}"
if [[ "${PRECISION}" == "int8-fp16" ]]; then
    echo "  fallback: INT8 layers may fall back to FP16 when TensorRT selects it"
fi
echo "  target: ${TARGET_BOARD}, ${TARGET_SOC}, TensorRT ${TARGET_TENSORRT}"
echo "  note: engines are platform/GPU/TensorRT specific; build this file on the target Orin Nano."

set -o pipefail
"${trtexec_bin}" "${args[@]}" 2>&1 | tee "${build_log}"

if [[ ! -s "${output_engine}" ]]; then
    echo "ERROR: TensorRT reported success but no engine was created." >&2
    exit 1
fi

echo "Engine ready: ${output_engine}"
