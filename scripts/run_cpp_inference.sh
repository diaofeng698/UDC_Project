#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

binary="${CPP_INFER_BIN:-${PROJECT_ROOT}/build/cpp/bnudc_trt_infer}"
if [[ -n "${ENGINE_PATH:-}" ]]; then
    input_engine="${ENGINE_PATH}"
else
    trtexec_bin="$(find_trtexec)"
    version="$(trt_version "${trtexec_bin}")"
    version="${version:-unknown}"
    input_engine="$(engine_path "${version}")"
fi
timestamp="$(date +%Y%m%d_%H%M%S)"
json_report="${INFER_JSON:-${RESULTS_DIR}/cpp_inference_${PRECISION}_${timestamp}.json}"

if [[ ! -x "${binary}" ]]; then
    echo "ERROR: C++ inference binary not found: ${binary}" >&2
    echo "Run make cpp-build first." >&2
    exit 1
fi
if [[ ! -s "${input_engine}" ]]; then
    echo "ERROR: ${PRECISION} engine not found: ${input_engine}" >&2
    echo "Build it first with make ${PRECISION} (or set ENGINE_PATH)." >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"
args=(
    --engine "${input_engine}"
    --warmup "${CPP_WARMUP:-20}"
    --iterations "${CPP_ITERATIONS:-200}"
    --json "${json_report}"
)
if [[ -n "${INPUT_SHAPE:-}" ]]; then
    args+=(--shape "${INPUT_SHAPE}")
fi
if [[ "${INCLUDE_TRANSFERS:-1}" == "0" ]]; then
    args+=(--no-transfers)
fi

"${binary}" "${args[@]}" "$@"
