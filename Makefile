SHELL := /usr/bin/env bash

.PHONY: check onnx-check onnx-check-fp32 onnx-check-fp16 onnx-check-int8 engine benchmark benchmark-lite benchmark-int8 benchmark-int8-lite benchmark-int8-fp16 benchmark-int8-fp16-lite profile-summary calib-cache calib-cache-random int8-random int8-fp16-random cpp-build infer infer-fp32 infer-fp16 infer-int8 infer-int8-fp16 fp32 fp16 int8 int8-fp16 clean

check:
	./scripts/check_env.sh

onnx-check:
	./scripts/check_onnx_support.sh

onnx-check-fp32:
	CHECK_PRECISION=fp32 ./scripts/check_onnx_support.sh

onnx-check-fp16:
	CHECK_PRECISION=fp16 ./scripts/check_onnx_support.sh

onnx-check-int8:
	CHECK_PRECISION=int8 ./scripts/check_onnx_support.sh

engine:
	./scripts/build_engine.sh

benchmark:
	./scripts/benchmark.sh

benchmark-lite:
	USE_CUDA_GRAPH=0 ENABLE_LAYER_PROFILE=0 INFERENCE_STREAMS=1 ./scripts/benchmark.sh

benchmark-int8:
	PRECISION=int8 USE_CUDA_GRAPH=0 ENABLE_LAYER_PROFILE=1 INFERENCE_STREAMS=1 ./scripts/benchmark.sh

benchmark-int8-lite:
	PRECISION=int8 USE_CUDA_GRAPH=0 ENABLE_LAYER_PROFILE=0 INFERENCE_STREAMS=1 ./scripts/benchmark.sh

benchmark-int8-fp16:
	PRECISION=int8-fp16 USE_CUDA_GRAPH=0 ENABLE_LAYER_PROFILE=1 INFERENCE_STREAMS=1 ./scripts/benchmark.sh

benchmark-int8-fp16-lite:
	PRECISION=int8-fp16 USE_CUDA_GRAPH=0 ENABLE_LAYER_PROFILE=0 INFERENCE_STREAMS=1 ./scripts/benchmark.sh

profile-summary:
	./scripts/summarize_latest_profile.sh

calib-cache:
	./scripts/generate_calibration_cache.sh

calib-cache-random:
	./scripts/generate_calibration_cache.sh

int8-random:
	CALIBRATION_CACHE="$${CALIBRATION_CACHE:-results/calibration_cache_random.bin}" ./scripts/generate_calibration_cache.sh
	CALIBRATION_CACHE="$${CALIBRATION_CACHE:-results/calibration_cache_random.bin}" PRECISION=int8 ./scripts/build_engine.sh

int8-fp16-random:
	CALIBRATION_CACHE="$${CALIBRATION_CACHE:-results/calibration_cache_random.bin}" ./scripts/generate_calibration_cache.sh
	CALIBRATION_CACHE="$${CALIBRATION_CACHE:-results/calibration_cache_random.bin}" PRECISION=int8-fp16 ./scripts/build_engine.sh

cpp-build:
	./scripts/build_cpp.sh

infer:
	./scripts/run_cpp_inference.sh

infer-fp32:
	PRECISION=fp32 ./scripts/run_cpp_inference.sh

infer-fp16:
	PRECISION=fp16 ./scripts/run_cpp_inference.sh

infer-int8:
	PRECISION=int8 ./scripts/run_cpp_inference.sh

infer-int8-fp16:
	PRECISION=int8-fp16 ./scripts/run_cpp_inference.sh

fp32:
	PRECISION=fp32 ./scripts/build_engine.sh

fp16:
	PRECISION=fp16 ./scripts/build_engine.sh

int8:
	PRECISION=int8 ./scripts/build_engine.sh

int8-fp16:
	CALIBRATION_CACHE="$${CALIBRATION_CACHE:-results/calibration_cache_random.bin}" PRECISION=int8-fp16 ./scripts/build_engine.sh

clean:
	rm -rf engines results build
