#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

build_dir="${CPP_BUILD_DIR:-${PROJECT_ROOT}/build/cpp}"
cmake_args=(
    -S "${PROJECT_ROOT}"
    -B "${build_dir}"
    -DCMAKE_BUILD_TYPE=Release
)
if [[ -n "${TENSORRT_ROOT:-}" ]]; then
    cmake_args+=("-DTENSORRT_ROOT=${TENSORRT_ROOT}")
fi

cmake "${cmake_args[@]}"
cmake --build "${build_dir}" --parallel "${BUILD_JOBS:-$(nproc)}"
echo "C++ inference binary: ${build_dir}/bnudc_trt_infer"
