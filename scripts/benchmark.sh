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
"${trtexec_bin}" \
    "--loadEngine=${input_engine}" \
    "--warmUp=${WARMUP_MS}" \
    "--duration=${BENCHMARK_SECONDS}" \
    "--streams=${INFERENCE_STREAMS}" \
    --useCudaGraph \
    --useSpinWait \
    --percentile=50,90,95,99 \
    --dumpProfile \
    --separateProfileRun \
    "--exportTimes=${result_prefix}_times.json" \
    "--exportProfile=${profile_json}" \
    "--exportLayerInfo=${layer_info_json}" \
    2>&1 | tee "${result_prefix}.log"

if [[ -s "${profile_json}" ]]; then
    python3 "${SCRIPT_DIR}/summarize_profile.py" \
        --profile "${profile_json}" \
        --layer-info "${layer_info_json}" \
        --output "${result_prefix}_summary.md"
else
    echo "WARNING: TensorRT did not create profile JSON; optimization summary was skipped." >&2
fi

echo "Benchmark results: ${result_prefix}*"
exit 0