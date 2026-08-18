SHELL := /usr/bin/env bash

.PHONY: check onnx-check onnx-check-fp32 onnx-check-fp16 onnx-check-int8 engine benchmark profile-summary cpp-build infer infer-fp32 infer-fp16 infer-int8 fp32 fp16 int8 clean

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

profile-summary:
	./scripts/summarize_latest_profile.sh

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

fp32:
	PRECISION=fp32 ./scripts/build_engine.sh

fp16:
	PRECISION=fp16 ./scripts/build_engine.sh

int8:
	PRECISION=int8 ./scripts/build_engine.sh

clean:
	rm -rf engines results build
