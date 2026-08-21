# P2：显式 Q/DQ PTQ、精度评估和性能验收

本流程使用代表性真实输入生成显式 `QuantizeLinear/DequantizeLinear` ONNX。当前 `bnudc_separable.onnx` 已将每个原始 `15x15 depthwise pre_conv8` 改为 `pre_conv8/vertical/Conv`（`15x1`）和 `pre_conv8/horizontal/Conv`（`1x15`）。普通 `Conv` 默认量化为 INT8；这一对 separable `pre_conv8` 和 MatMul/其替代算子默认不插入 Q/DQ，由 TensorRT `--int8 --fp16` 构建中的 FP16 fallback 执行。

> Q/DQ 决定哪些张量具有 INT8 量化语义；未量化层最终采用 FP16 还是 FP32，仍必须通过 TensorRT LayerInfo 确认。

## 1. 环境

主机侧安装：

```bash
python3 -m pip install -r requirements-p2.txt
```

Jetson 若需要 CUDA/TensorRT Execution Provider，应使用与 JetPack 匹配的 NVIDIA ONNX Runtime wheel，不要强行安装不兼容的通用 GPU wheel。

## 2. 准备真实数据

校准和精度验证必须使用原模型预处理之后的张量，而不是随机数或未经处理的图片。当前模型输入应保存为 FP32 NPY/NPZ，并保持实际的颜色顺序、归一化及 `1x3x512x416` 布局。

单输入模型支持：

- 一个样本一个 `.npy`；
- 一个样本一个 `.npz`，其中唯一数组或与输入同名的数组会被读取；
- 文本 manifest，每行一个 NPY/NPZ 路径；
- JSONL manifest，每行形如 `{"input": "/data/sample001.npy"}`，也支持多输入模型。

生成固定顺序并带 SHA-256 的数据清单：

```bash
CALIBRATION_DATASET=/data/bnudc/preprocessed \
CALIBRATION_SAMPLES=200 make p2-manifest
```

建议校准集和验收集互不重叠。清单只能保证可复现，不能替代对数据覆盖范围和业务代表性的人工确认。

## 3. 导出显式 Q/DQ PTQ ONNX

默认策略为普通卷积 INT8、`pre_conv8` FP16 fallback、MatMul FP16 fallback：

```bash
PTQ_SOURCE_MODEL=bnudc_separable.onnx \
CALIBRATION_DATASET=results/p2_calibration_manifest.txt \
QDQ_MODEL_PATH=artifacts/bnudc_qdq.onnx \
CALIBRATION_SAMPLES=200 make p2-ptq
```

核心配置：

- `--op-types Conv`：默认仅量化普通卷积，减少不必要 Q/DQ 边界；
- `--per-channel`：权重按输出通道量化；
- 激活和权重默认对称 INT8；
- `--calibration entropy`：默认熵校准，也可使用 `minmax` 或 `percentile`；
- `--pre-conv8 fp16|int8`：将每个 `pre_conv8` 的 `vertical` 和 `horizontal` 两个 Conv 作为一个策略家族控制；`fp16` 会通过父路径正则同时排除两者，`int8` 会同时允许两者参与 PTQ；
- `--matmul fp16|int8`：控制 MatMul/Gemm。P1 替代算子通常不应重新强制量化，除非 profile 证明更快且精度合格；
- `--exclude-regex` 和 `--include-regex` 可重复，用稳定 ONNX 节点名细化边界。

对 separable `pre_conv8` 整对执行两组消融实验：

```bash
python3 scripts/quantize_qdq_ptq.py \
  --model bnudc_separable.onnx \
  --dataset results/p2_calibration_manifest.txt \
  --output artifacts/bnudc_qdq_preconv_fp16.onnx \
  --pre-conv8 fp16 --samples 200

python3 scripts/quantize_qdq_ptq.py \
  --model bnudc_separable.onnx \
  --dataset results/p2_calibration_manifest.txt \
  --output artifacts/bnudc_qdq_preconv_int8.onnx \
  --pre-conv8 int8 --samples 200
```

每次导出会生成 `.quantization.json`，记录数据集、样本数、校准方法、选中/排除节点和 ONNX 算子计数。

## 4. 审计 Q/DQ 和融合边界

```bash
QDQ_MODEL_PATH=artifacts/bnudc_qdq.onnx make p2-audit
```

审计报告记录：

- `QuantizeLinear`、`DequantizeLinear` 数量；
- Q/DQ 后连接的算子类型；
- `pre_conv8` 和残留 MatMul/Gemm 节点；
- 跨 Q/DQ 查看时每个 Conv 后的 Add/LeakyRelu 邻域。

该静态审计只能发现潜在边界。最终是否保留 `Conv + Add + LeakyReLU` 融合必须检查 TensorRT LayerInfo 和 profile 中的层名、层数及耗时。

## 5. 构建与 benchmark

构建脚本现在会解析 ONNX 并验证存在 Q/DQ；不再允许 `INT8_EXPLICIT_QDQ=1` 绕过检查：

```bash
QDQ_MODEL_PATH=artifacts/bnudc_qdq.onnx make p2-engine
QDQ_MODEL_PATH=artifacts/bnudc_qdq.onnx make p2-benchmark
```

重新生成 FP16 和 Q/DQ profile summary 后，可自动检查主要性能门槛：

```bash
python3 scripts/check_p2_profile.py \
  --fp16 results/benchmark_fp16_<timestamp>_summary.json \
  --qdq results/benchmark_int8-fp16_<timestamp>_summary.json
```

默认门槛：INT8 时间占比至少 20%、FP32 不超过 10%、MatMul 不超过 5 ms、Reformat/Copy 不超过 FP16 的 2 倍、汇总层耗时不高于 FP16，并保留至少相同数量的 Conv 融合签名。汇总报告还会单独给出 `separable_pre_conv8`（`vertical + horizontal`）总耗时；可向验收脚本传入 `--max-pre-conv8-ms` 设置门槛。最终性能结论必须以 benchmark 日志中的 GPU Compute 为准，profile 汇总时间只用于定位。

## 6. 三输出精度评估

统一比较 FP16、PTQ 和 QAT ONNX：

```bash
python3 scripts/evaluate_onnx_accuracy.py \
  --reference artifacts/bnudc_fp32.onnx \
  --candidate fp16=artifacts/bnudc_fp16.onnx \
  --candidate ptq=artifacts/bnudc_qdq.onnx \
  --candidate qat=artifacts/bnudc_qat_qdq.onnx \
  --dataset /data/bnudc/validation_manifest.txt \
  --samples 200 \
  --max-abs 0.02 --max-mae 0.002 --min-psnr 40 --min-ssim 0.99 \
  --output results/accuracy/p2_accuracy.json
```

脚本按 ONNX 输出名（当前应为 `x_high`、`x_low`、`output`）分别报告：

- 最大绝对误差；
- 平均绝对误差；
- 最大相对误差和 MSE；
- PSNR；
- 全局 SSIM（每个完整输出张量计算后跨样本平均）。

若业务需要局部窗口 SSIM，应在业务评估程序中补充；当前全局 SSIM 用于无额外图像依赖的稳定回归门槛。

可通过 `--business-metric package.module:function` 注入业务指标。函数签名为：

```python
def metric(reference_outputs: dict[str, np.ndarray],
           candidate_outputs: dict[str, np.ndarray]) -> dict[str, float]:
    ...
```

## 7. PTQ 不合格时进入 QAT

QAT 必须在原始 PyTorch 模型和训练数据工程内完成，本部署仓库没有原始模型定义、checkpoint、损失函数和数据预处理，因此不能在这里自动生成可信 QAT 模型。QAT 导出必须满足：

1. fake-quant observer 使用与 TensorRT 一致的对称 INT8 范围；
2. 普通 Conv 优先插入 Q/DQ；
3. `pre_conv8` 和 P1 matvec 替代算子根据消融结果跳过或启用 fake quant；
4. 导出后的 ONNX 包含显式 Q/DQ，并通过 `make p2-audit`；
5. 将 QAT ONNX 作为 `--candidate qat=...` 接入同一验证集和相同阈值；
6. 在目标 Jetson 上使用相同构建/profile 流程验收，不能用训练端 fake-quant 指标代替 TensorRT 输出。

## 8. 验收顺序

1. FP32 reference 对三个输出的业务正确性已确认。
2. FP16、PTQ/QAT 使用完全相同且与校准集隔离的验证清单。
3. 所有输出通过误差、PSNR、SSIM 和业务指标门槛。
4. LayerInfo 证明普通 Conv 主要运行 INT8，且 `pre_conv8/vertical/Conv`、`pre_conv8/horizontal/Conv` 和 matvec 策略符合实验选择。
5. `Reformat/Copy` 数量和耗时未异常增加，Conv + Add + LeakyReLU 融合没有明显退化。
6. 至少三次固定功耗模式和时钟的 benchmark 中，Q/DQ 的 GPU Compute 中位数低于 FP16 基线。