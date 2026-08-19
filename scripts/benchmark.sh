#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

trtexec_bin="$(find_trtexec)"
version="$(trt_version "${trtexec_bin}")"
version="${version:-unknown}"
input_engine="${ENGINE_PATH:-$(engine_path "${version}")}"
timestamp="$(date +%Y%m%d_%H%M%S)"
result_prefix="${RESULTS_DIR}/benchmark_${PRECISION}_${timestamp}"
profile_json="${result_prefix}_profile.json"
layer_info_json="${result_prefix}_layers.json"
tegrastats_pid=""

if [[ ! -s "${input_engine}" ]]; then
    echo "ERROR: engine not found: ${input_engine}" >&2
    echo "Run scripts/build_engine.sh on this device first." >&2
    exit 1
fi

mkdir -p "${RESULTS_DIR}"

echo "Memory snapshot before TensorRT benchmark"
free -h || true
if [[ -r /proc/meminfo ]]; then
    awk '/^(MemAvailable|SwapFree|CmaTotal|CmaFree):/ {print "  " $0}' /proc/meminfo
fi

cleanup() {
    if [[ -n "${tegrastats_pid}" ]]; then
        kill "${tegrastats_pid}" 2>/dev/null || true
        wait "${tegrastats_pid}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if command -v tegrastats >/dev/null 2>&1; then
    tegrastats --interval 1000 >"${result_prefix}_tegrastats.log" &
    tegrastats_pid=$!
fi

echo "Benchmarking ${input_engine}"
args=(
    "--loadEngine=${input_engine}"
    "--warmUp=${WARMUP_MS}"
    "--duration=${BENCHMARK_SECONDS}"
    "--streams=${INFERENCE_STREAMS}"
    --useSpinWait
    --percentile=50,90,95,99
    "--exportTimes=${result_prefix}_times.json"
    "--exportLayerInfo=${layer_info_json}"
)
if [[ "${USE_CUDA_GRAPH}" == "1" ]]; then
    args+=(--useCudaGraph)
fi
if [[ "${ENABLE_LAYER_PROFILE}" == "1" ]]; then
    args+=(--dumpProfile --separateProfileRun "--exportProfile=${profile_json}")
fi

echo "  CUDA Graph: ${USE_CUDA_GRAPH}"
echo "  Layer profile: ${ENABLE_LAYER_PROFILE}"
if [[ "${PRECISION}" == "int8" && "${USE_CUDA_GRAPH}" == "1" ]]; then
    echo "WARNING: INT8 + CUDA Graph is known to exhaust memory on the target Orin Nano for this engine." >&2
    echo "         Use 'make benchmark-int8' (profile enabled) or 'make benchmark-int8-lite'." >&2
fi

set +e
"${trtexec_bin}" "${args[@]}" 2>&1 | tee "${result_prefix}.log"
trtexec_status=${PIPESTATUS[0]}
set -e
if ((trtexec_status != 0)); then
    if grep -Eqi 'nvmapmemallocinternaltagged|cuda failure: out of memory|out of memory' "${result_prefix}.log"; then
        echo "ERROR: TensorRT benchmark exhausted Jetson unified memory." >&2
        if [[ "${USE_CUDA_GRAPH}" == "1" ]]; then
            echo "       CUDA Graph capture was enabled; retry with USE_CUDA_GRAPH=0." >&2
        fi
        echo "       Recommended INT8 command: make benchmark-int8" >&2
        echo "       Lowest-memory command:     make benchmark-int8-lite" >&2
    fi
    exit "${trtexec_status}"
fi

if [[ "${ENABLE_LAYER_PROFILE}" == "1" && -s "${profile_json}" ]]; then
    python3 "${SCRIPT_DIR}/summarize_profile.py" \
        --profile "${profile_json}" \
        --layer-info "${layer_info_json}" \
        --output "${result_prefix}_summary.md"
elif [[ "${ENABLE_LAYER_PROFILE}" == "1" ]]; then
    echo "WARNING: TensorRT did not create profile JSON; optimization summary was skipped." >&2
fi

echo "Benchmark results: ${result_prefix}*"
exit 0