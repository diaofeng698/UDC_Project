#!/usr/bin/env bash

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

failures=0
warnings=0

target_error() {
    printf 'ERROR %-18s %s\n' "$1" "$2"
    failures=$((failures + 1))
}

target_warning() {
    printf 'WARN  %-18s %s\n' "$1" "$2"
    warnings=$((warnings + 1))
}

echo "Target deployment check"
echo "  expected board:    ${TARGET_BOARD}"
echo "  expected SoC:      ${TARGET_SOC}"
echo "  expected TensorRT: ${TARGET_TENSORRT}"
echo "  architecture:      $(uname -m)"

device_model="unknown"
device_compatible=""
if [[ -r /proc/device-tree/model ]]; then
    device_model="$(tr -d '\0' </proc/device-tree/model)"
fi
if [[ -r /proc/device-tree/compatible ]]; then
    device_compatible="$(tr '\0' ' ' </proc/device-tree/compatible)"
fi

is_jetson=0
if [[ "${device_compatible,,}" == *tegra* ]] || [[ "${device_model,,}" == *jetson* ]]; then
    is_jetson=1
fi

if ((is_jetson == 0)); then
    echo "  note: this host is not Jetson; build the final engine on the Orin Nano."
else
    echo "OK    board model        ${device_model}"
    if [[ "${device_compatible,,}" == *"${TARGET_SOC,,}"* ]]; then
        echo "OK    SoC                ${TARGET_SOC}"
    else
        target_error "SoC" "expected ${TARGET_SOC}; compatible=${device_compatible:-unknown}"
    fi

    memory_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    memory_gib="$((memory_kib / 1024 / 1024))"
    if ((memory_kib >= 7000000)); then
        echo "OK    memory             ${memory_gib} GiB reported (8GB module class)"
    else
        target_error "memory" "expected 8GB module; /proc/meminfo reports ${memory_gib} GiB"
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    printf 'OK    %-18s %s\n' "nvidia-smi" "$(command -v nvidia-smi)"
else
    echo "INFO  nvidia-smi         not detected (normal on Jetson)"
fi
if [[ -r /etc/nv_tegra_release ]]; then
    echo "OK    Jetson release     $(head -n 1 /etc/nv_tegra_release)"
elif [[ -r /etc/nvidia-jetpack-release ]]; then
    echo "OK    JetPack release    $(head -n 1 /etc/nvidia-jetpack-release)"
else
    echo "INFO  JetPack release    not detected"
fi

if trtexec_bin="$(find_trtexec)"; then
    version="$(trt_version "${trtexec_bin}")"
    package_version="$(trt_package_version)"
    echo "OK    trtexec            ${trtexec_bin} (binary ID ${version:-unknown})"
    if [[ -n "${package_version}" ]]; then
        if ((is_jetson == 1)) && [[ "${package_version}" != "${TARGET_TENSORRT}"* ]]; then
            target_error "TensorRT package" "expected ${TARGET_TENSORRT}; found ${package_version}"
        else
            echo "OK    TensorRT package   ${package_version}"
        fi
    elif ((is_jetson == 1)); then
        target_warning "TensorRT package" "version unavailable; expected ${TARGET_TENSORRT}"
    fi
else
    failures=$((failures + 1))
fi

if command -v python3 >/dev/null 2>&1; then
    python_version="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if ((is_jetson == 1)) && [[ "${python_version}" != "${TARGET_PYTHON}" ]]; then
        target_error "Python" "expected ${TARGET_PYTHON}; found ${python_version}"
    else
        echo "OK    Python             ${python_version} ($(command -v python3))"
    fi
else
    target_error "Python" "python3 not found; expected ${TARGET_PYTHON}"
fi

cpu_model="$(lscpu 2>/dev/null | sed -n 's/^Model name:[[:space:]]*//p' | head -n 1)"
cpu_cores="$(nproc)"
cpu_part="$(awk -F: 'tolower($1) ~ /cpu part/ {gsub(/[[:space:]]/, "", $2); print tolower($2); exit}' /proc/cpuinfo)"
if ((is_jetson == 1)); then
    if [[ "${cpu_model,,}" == *"${TARGET_CPU,,}"* ]] || [[ "${cpu_part}" == "0xd42" ]]; then
        echo "OK    CPU model          ${cpu_model:-ARM} (${TARGET_CPU}, part ${cpu_part:-unknown})"
    else
        target_error "CPU model" "expected ${TARGET_CPU}; found ${cpu_model:-unknown}"
    fi
    if [[ "${cpu_cores}" == "${TARGET_CPU_CORES}" ]]; then
        echo "OK    CPU cores          ${cpu_cores}"
    else
        target_error "CPU cores" "expected ${TARGET_CPU_CORES}; found ${cpu_cores}"
    fi
else
    echo "INFO  host CPU           ${cpu_model:-unknown}, ${cpu_cores} logical cores"
fi

if [[ -r "${MODEL_PATH}" ]]; then
    echo "OK    ONNX model         ${MODEL_PATH} ($(du -h "${MODEL_PATH}" | cut -f1))"
else
    echo "ERROR ONNX model         not readable: ${MODEL_PATH}"
    failures=$((failures + 1))
fi

if command -v tegrastats >/dev/null 2>&1; then
    echo "OK    tegrastats         $(command -v tegrastats)"
else
    echo "INFO  tegrastats         not detected (expected only on Jetson)"
fi

if ((failures > 0)); then
    echo "Environment check failed with ${failures} error(s)." >&2
    exit 1
fi

echo "Environment check passed with ${warnings} warning(s)."
