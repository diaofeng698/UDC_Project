SHELL := /usr/bin/env bash

.PHONY: check onnx-check onnx-check-fp32 onnx-check-fp16 onnx-check-int8 engine benchmark benchmark-lite benchmark-int8 benchmark-int8-lite benchmark-int8-fp16 benchmark-int8-fp16-lite profile-summary calib-cache calib-cache-random int8-random int8-fp16-random cpp-build infer infer-fp32 infer-fp16 infer-int8 infer-int8-fp16 fp32 fp16 int8 int8-fp16 tiny-matvec-reduce tiny-matvec-expanded tiny-matvec-onnx p2-manifest p2-ptq p2-audit p2-engine p2-benchmark p2-accuracy clean

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

tiny-matvec-reduce:
	python3 -m training_modules.tiny_matvec_3x3_example --implementation reduce

tiny-matvec-expanded:
	python3 -m training_modules.tiny_matvec_3x3_example --implementation expanded

tiny-matvec-onnx:
	python3 scripts/export_tiny_matvec_onnx.py --implementation both --two-matvecs

p2-manifest:
	@: "$${CALIBRATION_DATASET:?Set CALIBRATION_DATASET to a preprocessed NPY/NPZ directory}"
	python3 scripts/create_dataset_manifest.py --dataset "$${CALIBRATION_DATASET}" --output "$${DATASET_MANIFEST:-results/p2_calibration_manifest.txt}" --limit "$${CALIBRATION_SAMPLES:-100}"

p2-ptq:
	@: "$${CALIBRATION_DATASET:?Set CALIBRATION_DATASET to a preprocessed NPY/NPZ directory or manifest}"
	python3 scripts/quantize_qdq_ptq.py --model "$${PTQ_SOURCE_MODEL:-$${MODEL_PATH:-bnudc_separable.onnx}}" --dataset "$${CALIBRATION_DATASET}" --output "$${QDQ_MODEL_PATH:-artifacts/bnudc_qdq.onnx}" --samples "$${CALIBRATION_SAMPLES:-100}" --pre-conv8 "$${PRE_CONV8_PRECISION:-fp16}" --matmul "$${MATMUL_PRECISION:-fp16}"

p2-audit:
	python3 scripts/inspect_qdq_onnx.py --model "$${QDQ_MODEL_PATH:-artifacts/bnudc_qdq.onnx}" --require-qdq

p2-engine: p2-audit
	MODEL_PATH="$${QDQ_MODEL_PATH:-artifacts/bnudc_qdq.onnx}" PRECISION=int8-fp16 ./scripts/build_engine.sh

p2-benchmark:
	MODEL_PATH="$${QDQ_MODEL_PATH:-artifacts/bnudc_qdq.onnx}" PRECISION=int8-fp16 USE_CUDA_GRAPH=0 ENABLE_LAYER_PROFILE=1 INFERENCE_STREAMS=1 ./scripts/benchmark.sh

p2-accuracy:
	@: "$${ACCURACY_DATASET:?Set ACCURACY_DATASET to a preprocessed NPY/NPZ directory or manifest}"
	@: "$${FP32_MODEL_PATH:?Set FP32_MODEL_PATH to the reference ONNX model}"
	python3 scripts/evaluate_onnx_accuracy.py --reference "$${FP32_MODEL_PATH}" --candidate "ptq=$${QDQ_MODEL_PATH:-artifacts/bnudc_qdq.onnx}" --dataset "$${ACCURACY_DATASET}" --samples "$${ACCURACY_SAMPLES:-100}" --output "$${ACCURACY_REPORT:-results/accuracy/p2_accuracy.json}" $${ACCURACY_ARGS:-}

clean:
	rm -rf engines results build
