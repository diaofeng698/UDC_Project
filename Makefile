SHELL := /usr/bin/env bash

.PHONY: check onnx-check onnx-check-fp32 engine benchmark fp32 fp16 clean

check:
	./scripts/check_env.sh

onnx-check:
	CHECK_PRECISION=fp16 ./scripts/check_onnx_support.sh

onnx-check-fp32:
	CHECK_PRECISION=fp32 ./scripts/check_onnx_support.sh

engine:
	./scripts/build_engine.sh

benchmark:
	./scripts/benchmark.sh

fp32:
	PRECISION=fp32 ./scripts/build_engine.sh

fp16:
	PRECISION=fp16 ./scripts/build_engine.sh

clean:
	rm -rf engines results
