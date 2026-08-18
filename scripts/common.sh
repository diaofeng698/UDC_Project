#!/usr/bin/env bash

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/config/orin_nano.env}"

if [[ ! -f "${CONFIG_FILE}" ]]; then
    echo "ERROR: config file not found: ${CONFIG_FILE}" >&2
    exit 1
fi

# shellcheck source=../config/orin_nano.env
source "${CONFIG_FILE}"

MODEL_PATH="${MODEL_PATH:-bnudc_v1_trt_static.onnx}"
ENGINE_DIR="${ENGINE_DIR:-engines}"
RESULTS_DIR="${RESULTS_DIR:-results}"
PRECISION="${PRECISION:-fp32}"
WORKSPACE_MIB="${WORKSPACE_MIB:-2048}"
BUILD_OPT_LEVEL="${BUILD_OPT_LEVEL:-5}"
WARMUP_MS="${WARMUP_MS:-1000}"
BENCHMARK_SECONDS="${BENCHMARK_SECONDS:-30}"
INFERENCE_STREAMS="${INFERENCE_STREAMS:-1}"
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-1}"
ENABLE_LAYER_PROFILE="${ENABLE_LAYER_PROFILE:-1}"
TARGET_BOARD="${TARGET_BOARD:-Jetson Orin Nano Super 8GB}"
TARGET_SOC="${TARGET_SOC:-tegra234}"
TARGET_TENSORRT="${TARGET_TENSORRT:-10.3.0.30}"
TARGET_PYTHON="${TARGET_PYTHON:-3.10}"
TARGET_CPU="${TARGET_CPU:-Cortex-A78AE}"
TARGET_CPU_CORES="${TARGET_CPU_CORES:-6}"

[[ "${MODEL_PATH}" = /* ]] || MODEL_PATH="${PROJECT_ROOT}/${MODEL_PATH}"
[[ "${ENGINE_DIR}" = /* ]] || ENGINE_DIR="${PROJECT_ROOT}/${ENGINE_DIR}"
[[ "${RESULTS_DIR}" = /* ]] || RESULTS_DIR="${PROJECT_ROOT}/${RESULTS_DIR}"

find_trtexec() {
    if [[ -n "${TRTEXEC:-}" ]]; then
        [[ -x "${TRTEXEC}" ]] || {
            echo "ERROR: TRTEXEC is not executable: ${TRTEXEC}" >&2
            return 1
        }
        printf '%s\n' "${TRTEXEC}"
        return
    fi

    if command -v trtexec >/dev/null 2>&1; then
        command -v trtexec
        return
    fi

    local candidate
    local candidates=()
    for candidate in \
        /usr/src/tensorrt/bin/trtexec \
        /usr/src/tensorrt/targets/*/bin/trtexec \
        /opt/TensorRT-*/targets/*/bin/trtexec; do
        if [[ -x "${candidate}" ]]; then
            candidates+=("${candidate}")
        fi
    done

    if ((${#candidates[@]} > 0)); then
        printf '%s\n' "${candidates[@]}" | sort -V | tail -n 1
        return
    fi

    echo "ERROR: trtexec not found. Install TensorRT or set TRTEXEC=/path/to/trtexec." >&2
    return 1
}

trt_version() {
    local output
    output="$("${1}" --version 2>&1 || true)"
    sed -n 's/.*TensorRT v\([0-9][0-9.]*\).*/\1/p' <<<"${output}" | head -n 1
}

trt_package_version() {
    if ! command -v dpkg-query >/dev/null 2>&1; then
        return
    fi

    { dpkg-query -W -f='${Version}\n' libnvinfer10 libnvinfer-bin tensorrt 2>/dev/null || true; } \
        | sed -n 's/^\([0-9][0-9.]*\).*/\1/p' \
        | sort -V \
        | tail -n 1
}

engine_path() {
    local version="$1"
    local machine
    machine="$(uname -m)"
    printf '%s/%s_%s_trt%s_%s.plan\n' \
        "${ENGINE_DIR}" "$(basename "${MODEL_PATH}" .onnx)" "${machine}" "${version}" "${PRECISION}"
}
