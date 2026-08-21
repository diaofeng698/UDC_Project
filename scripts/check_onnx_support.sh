#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

trtexec_bin="$(find_trtexec)"
binary_version="$(trt_version "${trtexec_bin}")"
binary_version="${binary_version:-unknown}"
package_version="$(trt_package_version)"
package_version="${package_version:-unknown}"
check_precision="${CHECK_PRECISION:-${PRECISION}}"
mkdir -p "${RESULTS_DIR}"

if [[ ! -r "${MODEL_PATH}" ]]; then
    echo "ERROR: ONNX model not readable: ${MODEL_PATH}" >&2
    exit 1
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 is required to summarize the TensorRT log." >&2
    exit 1
fi

log_path="${ANALYZE_LOG:-${RESULTS_DIR}/onnx_support_trt${package_version}_${check_precision}.log}"
report_path="${SUPPORT_REPORT:-${RESULTS_DIR}/onnx_support_trt${package_version}_${check_precision}.md}"
command_status=0

if [[ -z "${ANALYZE_LOG:-}" ]]; then
    case "${check_precision}" in
        fp32) precision_args=() ;;
        fp16) precision_args=(--fp16) ;;
        int8) precision_args=(--int8) ;;
        *)
            echo "ERROR: CHECK_PRECISION must be fp32, fp16, or int8; got ${check_precision}" >&2
            exit 2
            ;;
    esac

    echo "Checking ONNX with TensorRT package ${package_version}, binary ID ${binary_version}, precision ${check_precision}"
    echo "This performs a real TensorRT parse and engine build without saving a deployable plan."
    args=(
        "--onnx=${MODEL_PATH}"
        --skipInference
        --verbose
        "--memPoolSize=workspace:${WORKSPACE_MIB}"
        "--builderOptimizationLevel=${BUILD_OPT_LEVEL}"
    )
    args+=("${precision_args[@]}")

    set +e
    "${trtexec_bin}" "${args[@]}" >"${log_path}" 2>&1
    command_status=$?
    set -e
else
    [[ -r "${log_path}" ]] || {
        echo "ERROR: ANALYZE_LOG is not readable: ${log_path}" >&2
        exit 1
    }
    echo "Analyzing existing TensorRT log: ${log_path}"
fi

set +e
python3 "${SCRIPT_DIR}/analyze_trt_log.py" \
    --log "${log_path}" \
    --report "${report_path}" \
    --trt-version "${binary_version}" \
    --package-version "${package_version}" \
    --precision "${check_precision}" \
    --command-status "${command_status}"
analysis_status=$?
set -e

if ((analysis_status != 0)); then
    echo "ERROR: ONNX is not build-compatible with the tested TensorRT configuration. See ${report_path}" >&2
    exit "${analysis_status}"
fi

echo "Compatibility scan completed. Warnings require review; validate numerical accuracy separately."