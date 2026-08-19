# BNUDC TensorRT 部署指南（Jetson Orin Nano Super 8GB）

本文档用于在 NVIDIA Jetson Orin Nano Super 8GB 上构建并测试 bnudc_v1_trt_static.onnx 的 TensorRT 部署流程。默认引擎精度为 FP32（batch=1），FP16 和 INT8 作为显式可选项。

## 文档版本

- English: README.md
- 中文: README.zh.md

## 快速导航

- [生产目标基线](#生产目标基线)
- [模型元数据（已验证）](#模型元数据已验证)
- [为什么必须在 Jetson 上构建引擎](#为什么必须在-jetson-上构建引擎)
- [Orin Nano 快速开始](#orin-nano-快速开始)
- [TensorRT 感知的 ONNX 算子检查](#tensorrt-感知的-onnx-算子检查)
- [配置项](#配置项)
- [精度选择](#精度选择)
- [合成标定缓存（dummy/random）](#合成标定缓存dummyrandom)
- [C++ 前向推理部署](#c-前向推理部署)
- [优化顺序](#优化顺序)
- [Benchmark 结果解读](#benchmark-结果解读)
- [关键约束](#关键约束)

## 生产目标基线

| 项目 | 目标要求 |
| --- | --- |
| 板卡 | Jetson Orin Nano Super 8GB |
| SoC | NVIDIA Tegra234（tegra234） |
| CPU | 6 核 Cortex-A78AE |
| 架构 | AArch64 |
| TensorRT | 10.3.0.30 |
| Python | 3.10 |

影响说明：

- TensorRT 10.3.0.30 直接影响可部署性。最终 plan 与 timing cache 必须在目标版本生成。
- Orin Nano 8GB 直接影响性能与内存。默认保留 2GiB builder workspace，若构建被 OOM 杀死，可降低 WORKSPACE_MIB 或临时加 swap。
- Tegra234/AArch64 直接影响二进制兼容性。引擎与插件应在 AArch64 目标上构建，不应从 x86_64 直接拷贝部署。
- 6 核 CPU 影响端到端流水（预处理/后处理线程规模），不影响 TensorRT 图转换本身。
- Python 3.10 对当前 shell/trtexec 流程不是硬依赖，但如果新增 Python 校准/推理脚本，需要与 JetPack/TensorRT 的 Python 绑定匹配。

执行 make check 可在 Jetson 上校验以上约束；在非 Jetson 开发机上会给出差异提示，不会阻断工作区。

## 模型元数据（已验证）

该 ONNX 在开发机上的 TensorRT 8.6.1 可成功解析，这只证明 ONNX 解析兼容；生产引擎仍需在目标板 TensorRT 10.3.0.30 重建并验证。

| 项目 | 值 |
| --- | --- |
| ONNX IR / opset | IR 7 / opset 13 |
| Producer | PyTorch 2.4.1 |
| 输入 | input: FP32 [1, 3, 512, 416] |
| 输出 | x_high, x_low, output |
| ONNX 大小 | 约 20 MiB |

上线前需结合原始 PyTorch 预处理/后处理，确认三个输出的语义、数值范围、色彩顺序与 shape 含义。

## 为什么必须在 Jetson 上构建引擎

TensorRT plan 与 TensorRT 版本、目标架构、GPU、软件栈强绑定。不要把 x86_64/RTX 上构建的引擎直接部署到 Orin Nano。

建议目标环境：

- Jetson Orin Nano Super 8GB（tegra234）AArch64
- TensorRT 10.3.0.30（来自目标 JetPack）
- Python 3.10（用于可选 Python 工具链）
- 合适电源模式与散热
- 至少约 2GiB 可用内存用于默认构建 workspace

## Orin Nano 快速开始

1. 环境检查：

~~~bash
make check
~~~

1. ONNX 兼容与可构建检查（默认 FP32）：

~~~bash
make onnx-check
~~~

1. 设置性能模式并锁频（请先查询可用 ID）：

~~~bash
sudo nvpmodel -q --verbose
sudo nvpmodel -m <ID>
sudo jetson_clocks
~~~

1. 构建默认 FP32 引擎：

~~~bash
make engine
~~~

1. 运行 30 秒 benchmark：

~~~bash
make benchmark
~~~

产物目录：

- 引擎与缓存：engines/
- 日志与报告：results/

## TensorRT 感知的 ONNX 算子检查

在目标板执行 make onnx-check，会调用本机 trtexec，结果反映当前 TensorRT/ONNX parser/CUDA/plugin/目标 GPU 的真实可构建性。

输出文件（results/）：

- onnx_support_*.log：完整 verbose 日志
- onnx_support_*.md：摘要报告（错误、警告、敏感算子、替代建议、算子清单）

结果级别：

- PASS：可构建且无已知审查项
- WARNING：可构建但存在 cast/clamp 或其他兼容性/语义/数值风险
- ERROR：解析或构建失败，或无可用实现/tactic

补充命令：

~~~bash
make onnx-check-fp32
make onnx-check-fp16
make onnx-check-int8
TRTEXEC=/custom/path/trtexec make onnx-check
ANALYZE_LOG=/path/trtexec.log SUPPORT_REPORT=/tmp/report.md ./scripts/check_onnx_support.sh
~~~

说明：tactic 搜索中出现某个 backend 因 workspace 不足而被跳过，并不一定是最终失败；若最终 build 成功，会被归类为可恢复 fallback（WARNING）。

## 配置项

默认值在 config/orin_nano.env，可通过环境变量覆盖：

- TRTEXEC=/path/to/trtexec
- PRECISION=fp32|fp16|int8|int8-fp16
- WORKSPACE_MIB=2048
- BUILD_OPT_LEVEL=5
- ENGINE_PATH=/path/model.plan
- BENCHMARK_SECONDS=60
- INFERENCE_STREAMS=1
- TARGET_TENSORRT=10.3.0.30
- TARGET_PYTHON=3.10

示例：

~~~bash
WORKSPACE_MIB=1024 BUILD_OPT_LEVEL=3 make fp16
~~~

## 精度选择

| 精度 | 构建命令 | Benchmark 命令 |
| --- | --- | --- |
| FP32（默认） | make engine 或 make fp32 | make benchmark |
| FP16 | make fp16 | PRECISION=fp16 make benchmark |
| INT8 | make int8 | make benchmark-int8 |

通用写法：PRECISION=<fp32|fp16|int8|int8-fp16> make engine。`int8-fp16` 同时启用 INT8 和 FP16，使不适合 INT8 的层可以回退到 FP16，并使用独立的引擎与 timing cache 文件。

INT8 + FP16 fallback：

~~~bash
make int8-fp16
make benchmark-int8-fp16
make infer-int8-fp16
~~~

INT8 需要以下之一：

- 有效 Q/DQ ONNX；或
- 可用 calibration cache（CALIBRATION_CACHE=...）

## 合成标定缓存（dummy/random）

工程已提供“合成随机输入”生成 INT8 calibration cache 的能力，适用于联调、构建验证、回归排障。

默认行为：

- 数值范围：[-1, 1]
- 可覆盖范围、batch 数、随机种子、输出路径

使用示例：

1. 默认生成：

~~~bash
make calib-cache
~~~

1. 用生成的 cache 构建 INT8：

~~~bash
CALIBRATION_CACHE=results/calibration_cache_random.bin make int8
~~~

1. 一步完成“生成 + INT8 构建”：

~~~bash
make int8-random
~~~

1. 指定随机范围（例如 [-0.5, 0.5]）：

~~~bash
CALIBRATION_MIN=-0.5 CALIBRATION_MAX=0.5 make calib-cache
~~~

1. 指定 batch/seed/path：

~~~bash
CALIBRATION_BATCHES=64 CALIBRATION_SEED=123 CALIBRATION_CACHE=results/calib_rnd.cache make calib-cache
~~~

1. 动态输入 shape（多输入用逗号分隔）：

~~~bash
CALIBRATION_INPUT_SHAPES=input=1x3x512x416 make calib-cache
~~~

注意：合成随机 cache 仅用于工程验证，不建议直接作为生产精度验收依据。生产请使用代表性真实数据标定，或已验证的显式 Q/DQ 量化模型。

## C++ 前向推理部署

cpp/ 下运行时采用 TensorRT C++ name-based I/O API（setTensorAddress + enqueueV3）和 CUDA Runtime：

- 反序列化目标板引擎
- 为输入输出分配 pinned host 与 device buffer
- 填充确定性 dummy 输入
- 异步执行 H2D/推理/D2H
- 输出 checksum 与性能指标 JSON

构建：

~~~bash
make cpp-build
~~~

自定义 TensorRT 根目录：

~~~bash
TENSORRT_ROOT=/path/to/TensorRT make cpp-build
~~~

运行：

~~~bash
make infer
make infer-fp32
make infer-fp16
make infer-int8
~~~

常用覆盖：

~~~bash
CPP_WARMUP=50 CPP_ITERATIONS=500 make infer
INCLUDE_TRANSFERS=0 make infer
ENGINE_PATH=/absolute/model.plan make infer
INPUT_SHAPE=input=1x3x512x416 make infer
INFER_JSON=/path/metrics.json make infer
./build/cpp/bnudc_trt_infer --help
~~~

默认输出位于 results/cpp_inference_[precision]_[timestamp].json。

## 优化顺序

1. 先做 FP32 正确性基线（真实样本与 PyTorch 对齐）。
2. 再做 FP16（精度与性能评估），通常是 Orin 首选量产路径。
3. 优化流水（GPU 内预后处理、异步拷贝、持久 context、必要时 CUDA Graph）。
4. INT8 只在“真实标定或已验证 Q/DQ”前提下验收。
5. 先看 profile 与 tegrastats，再做定点优化。

## Benchmark 结果解读

默认 benchmark 为 batch=1、单 stream，建议固定功耗/温度条件后做对比。

trtexec 默认使用合成输入，测得的是 TensorRT 引擎性能，不含完整业务前后处理。

### 自动 profile 汇总

make benchmark 后会自动输出：

- benchmark_*_summary.md：可读优化建议
- benchmark_*_summary.csv：逐层耗时
- benchmark_*_summary.json：聚合统计
- benchmark_*_layers.json：层元数据

也可单独重算：

~~~bash
make profile-summary
PROFILE_JSON=/path/profile.json LAYER_INFO_JSON=/path/layers.json make profile-summary
~~~

averageMs 是逐层 GPU 执行时间，不等价于 GPU 利用率/SM 占用率/带宽占用率。

### Jetson 统一内存 OOM 说明

Orin Nano 的 CPU/GPU 共享 8GB 物理内存。NvMap error 12 + CUDA OOM 常见于运行期分配失败，并不直接等同于算子不支持。

推荐命令：

~~~bash
make benchmark-int8
make benchmark-int8-lite
~~~

避免组合（该模型已验证易 OOM）：

~~~bash
PRECISION=int8 USE_CUDA_GRAPH=1 ENABLE_LAYER_PROFILE=0 make benchmark
~~~

已知该模型在 INT8 + CUDA Graph 下存在 OOM 风险，应优先保持非 Graph 路径并结合 profile 定位热点层。

## 关键约束

- 当前 ONNX 输入 shape 为静态；改分辨率通常需要重导出或改图。
- FP16 加速依赖模型与 tactic，部分层可能仍保留 FP32。
- INT8 无代表性标定或无有效 Q/DQ 时，不是可验收部署产物。
- ONNX 是可移植源；plan 与 timing cache 属于目标板本地产物。
- JetPack/TensorRT/CUDA/plugin 或目标软件栈变更后，应重建引擎。
