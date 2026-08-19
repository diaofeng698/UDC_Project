#!/usr/bin/env python3

import argparse
import ctypes
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

try:
    import tensorrt as trt
except ImportError as exc:
    print("ERROR: failed to import tensorrt Python package.", file=sys.stderr)
    print("       Install TensorRT Python bindings on the target board.", file=sys.stderr)
    raise SystemExit(2) from exc


CUDA_MEMCPY_HOST_TO_DEVICE = 1


def _load_cudart() -> ctypes.CDLL:
    candidates = ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0")
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    print("ERROR: failed to load CUDA runtime library (libcudart.so).", file=sys.stderr)
    raise SystemExit(2)


CUDART = _load_cudart()
CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
CUDART.cudaMalloc.restype = ctypes.c_int
CUDART.cudaFree.argtypes = [ctypes.c_void_p]
CUDART.cudaFree.restype = ctypes.c_int
CUDART.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
CUDART.cudaMemcpy.restype = ctypes.c_int


def _cuda_check(status: int, op: str) -> None:
    if status != 0:
        print(f"ERROR: CUDA call failed ({op}), status={status}", file=sys.stderr)
        raise SystemExit(2)


class DeviceBuffer:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self.ptr = ctypes.c_void_p()
        _cuda_check(CUDART.cudaMalloc(ctypes.byref(self.ptr), nbytes), "cudaMalloc")

    def copy_from(self, array: np.ndarray) -> None:
        src = ctypes.c_void_p(array.ctypes.data)
        _cuda_check(
            CUDART.cudaMemcpy(self.ptr, src, array.nbytes, CUDA_MEMCPY_HOST_TO_DEVICE),
            "cudaMemcpy(H2D)",
        )

    def __del__(self) -> None:
        if getattr(self, "ptr", None) and self.ptr.value:
            CUDART.cudaFree(self.ptr)
            self.ptr = ctypes.c_void_p()


@dataclass
class InputSpec:
    name: str
    shape: Tuple[int, ...]

    @property
    def volume(self) -> int:
        vol = 1
        for dim in self.shape:
            vol *= dim
        return vol


def parse_shape_override(text: str) -> Tuple[str, Tuple[int, ...]]:
    if "=" not in text:
        raise ValueError(f"invalid --shape '{text}', expected name=1x3xHxW")
    name, value = text.split("=", 1)
    dims = tuple(int(v) for v in value.split("x") if v)
    if not name or not dims:
        raise ValueError(f"invalid --shape '{text}', expected name=1x3xHxW")
    if any(d <= 0 for d in dims):
        raise ValueError(f"invalid --shape '{text}', all dimensions must be > 0")
    return name, dims


class RandomEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        specs: List[InputSpec],
        batches: int,
        min_value: float,
        max_value: float,
        seed: int,
        cache_file: str,
    ) -> None:
        super().__init__()
        self.specs = specs
        self.spec_by_name = {spec.name: spec for spec in specs}
        self.batches = batches
        self.min_value = min_value
        self.max_value = max_value
        self.cache_file = cache_file
        self.batch_index = 0
        self.rng = np.random.default_rng(seed)
        self.host: Dict[str, np.ndarray] = {}
        self.device: Dict[str, DeviceBuffer] = {}

        for spec in specs:
            host = np.empty(spec.shape, dtype=np.float32)
            self.host[spec.name] = host
            self.device[spec.name] = DeviceBuffer(host.nbytes)

    def get_batch_size(self) -> int:
        return self.specs[0].shape[0]

    def get_batch(self, names: List[str]):
        if self.batch_index >= self.batches:
            return None

        ptrs: List[int] = []
        for name in names:
            if name not in self.spec_by_name:
                print(f"ERROR: unexpected calibration input requested: {name}", file=sys.stderr)
                return None
            host = self.host[name]
            host[...] = self.rng.uniform(self.min_value, self.max_value, size=host.shape)
            self.device[name].copy_from(host)
            ptrs.append(int(self.device[name].ptr.value))

        self.batch_index += 1
        return ptrs

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache) -> None:
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a TensorRT INT8 calibration cache using synthetic random input. "
            "Use for functional/engineering validation, not production accuracy sign-off."
        )
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model")
    parser.add_argument("--cache", required=True, help="Output calibration cache path")
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        help="Override dynamic input shape, format name=1x3x512x416; repeat for multiple inputs",
    )
    parser.add_argument("--batches", type=int, default=32, help="Number of synthetic calibration batches")
    parser.add_argument("--min", dest="min_value", type=float, default=-1.0, help="Random min value")
    parser.add_argument("--max", dest="max_value", type=float, default=1.0, help="Random max value")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--workspace-mib", type=int, default=2048, help="Builder workspace size in MiB")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache file")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.min_value >= args.max_value:
        print("ERROR: --min must be smaller than --max", file=sys.stderr)
        return 2

    if args.batches <= 0:
        print("ERROR: --batches must be > 0", file=sys.stderr)
        return 2

    if not os.path.isfile(args.onnx):
        print(f"ERROR: ONNX model not found: {args.onnx}", file=sys.stderr)
        return 2

    if os.path.exists(args.cache) and not args.force:
        print(f"ERROR: cache already exists: {args.cache}", file=sys.stderr)
        print("       Use --force to overwrite.", file=sys.stderr)
        return 2

    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)

    shape_overrides = dict(parse_shape_override(item) for item in args.shape)

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_trt = trt.OnnxParser(network, logger)

    with open(args.onnx, "rb") as f:
        if not parser_trt.parse(f.read()):
            print("ERROR: failed to parse ONNX model.", file=sys.stderr)
            for i in range(parser_trt.num_errors):
                print(f"  - {parser_trt.get_error(i)}", file=sys.stderr)
            return 1

    specs: List[InputSpec] = []
    dynamic_inputs = False
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        shape = tuple(int(d) for d in tensor.shape)
        if any(d <= 0 for d in shape):
            dynamic_inputs = True
            if tensor.name not in shape_overrides:
                print(
                    f"ERROR: input {tensor.name} has dynamic shape {shape}; provide --shape {tensor.name}=...",
                    file=sys.stderr,
                )
                return 2
            shape = shape_overrides[tensor.name]
        specs.append(InputSpec(name=tensor.name, shape=shape))

    calibrator = RandomEntropyCalibrator(
        specs=specs,
        batches=args.batches,
        min_value=args.min_value,
        max_value=args.max_value,
        seed=args.seed,
        cache_file=args.cache,
    )

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = calibrator
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_mib << 20)

    if dynamic_inputs:
        profile = builder.create_optimization_profile()
        for spec in specs:
            profile.set_shape(spec.name, spec.shape, spec.shape, spec.shape)
        config.add_optimization_profile(profile)

    print("Generating synthetic INT8 calibration cache")
    print(f"  onnx:   {args.onnx}")
    print(f"  cache:  {args.cache}")
    print(f"  range:  [{args.min_value}, {args.max_value}]")
    print(f"  batches:{args.batches}")
    print(f"  seed:   {args.seed}")

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: TensorRT build failed while generating calibration cache.", file=sys.stderr)
        return 1

    if not os.path.isfile(args.cache) or os.path.getsize(args.cache) == 0:
        print("ERROR: calibration cache was not generated or is empty.", file=sys.stderr)
        return 1

    print(f"Calibration cache ready: {args.cache}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
