# TensorRT INT8 性能修复与跟踪计划

本文档记录 FP16 与 INT8 benchmark 的性能差异、已确认根因、修复顺序、验证指标和后续实施状态。

## 1. 当前基线

测试环境和输入保持一致：Jetson Orin Nano、TensorRT 10.3.0.30、batch=1、输入 `1x3x512x416`。

| 指标 | FP16 | INT8 | INT8/FP16 |
| --- | ---: | ---: | ---: |
| 汇总层耗时 | 261.50 ms | 2866.19 ms | 10.96x |
| GPU Compute mean | 256.65 ms | 2866.67 ms | 11.17x |
| 吞吐量 | 3.862 qps | 0.349 qps | 0.09x |

基线结果：

- FP16：`results/benchmark_fp16_20260819_132757_summary.md`
- INT8：`results/benchmark_int8_20260819_141242_summary.md`

## 2. 已确认的主要问题

### 2.1 两个 FP32 MatMul 严重退化

INT8 引擎中：

| 层 | 实际精度 | 平均耗时 | 占比 |
| --- | --- | ---: | ---: |
| `/MatMul` | FP32 | 1255.86 ms | 43.82% |
| `/MatMul_1` | FP32 | 1249.10 ms | 43.58% |
| 合计 | FP32 | 2504.96 ms | 87.40% |

它们实际执行大量微型矩阵乘法：

- A：`[212992, 3, 3]`
- B：`[212992, 3, 1]`
- 输出：`[212992, 3, 1]`

INT8 引擎为其选择了 FP32 `cublas_gemvx` tactic；FP16 引擎则将相同逻辑优化为多个 Myelin `MulSum` kernel，总计仅约 1.274 ms。

### 2.2 INT8 构建未允许 FP16 fallback

当前 INT8 构建仅传递 `--int8`。无法使用高效 INT8 tactic 的层会回退到 FP32，而不是优先回退到 FP16。

INT8 引擎实际耗时分布：

| 实际精度 | 层数 | 耗时 | 占比 |
| --- | ---: | ---: | ---: |
| FP32 | 199 | 2789.72 ms | 97.33% |
| INT8 | 415 | 76.47 ms | 2.67% |

### 2.3 大型 depthwise 卷积回退 FP32

`15x15 depthwise pre_conv8` 在 FP16 引擎中单层约 5.2 ms，在 INT8 引擎中回退 FP32 后单层约 10 ms。

家族累计耗时：

| 层家族 | FP16 | INT8 |
| --- | ---: | ---: |
| `pre_conv8/Conv` | 109.00 ms | 200.74 ms |
| `conv8/Conv` | 19.34 ms | 26.38 ms |

### 2.4 CUDA Graph 不是主要根因

INT8 benchmark 当前关闭 CUDA Graph，导致 enqueue 时间接近 GPU Compute 时间。启用 CUDA Graph 可以降低 CPU 提交开销，但不能修复单个 MatMul 超过 1.2 秒的问题。

Jetson 统一内存有限，INT8 + CUDA Graph 需要单独进行内存安全验证。

## 3. 修复实施顺序

### P0：构建 INT8 + FP16 fallback 引擎

**目标**：允许普通卷积使用 INT8，INT8 不支持或速度较差的层回退 FP16，而不是 FP32。

计划修改：

- [x] 新增 `int8-fp16` 构建模式，同时传递 `--int8 --fp16`。
- [x] 混合精度引擎使用 `int8-fp16` 独立名称，不覆盖纯 INT8 基线。
- [x] timing cache 按精度隔离；混合精度首次构建使用新的 cache。
- [x] 保留当前随机 calibration cache，仅用于性能链路验证。
- [ ] 后续用真实代表性数据重新生成 calibration cache。

实现过程中发现并修复了 workspace 单位问题：TensorRT 10.3 的
`--memPoolSize` 接受 `M`，但不接受 `MiB` 后缀；原来的
`workspace:2048MiB` 被解析成了 2048 字节。当前改为无后缀的
`workspace:2048`（默认单位为 MiB），并将 workspace 大小加入 timing
cache 文件名，避免复用错误容量下生成的缓存。

实现后的常用命令：

```bash
# 使用已有随机 calibration cache 构建独立混合精度引擎
make int8-fp16

# 重新生成随机 calibration cache 后构建
make int8-fp16-random

# 关闭 CUDA Graph，执行带逐层 profile 的基准测试
make benchmark-int8-fp16

# 执行 C++ 推理
make infer-int8-fp16
```

默认产物不会覆盖纯 INT8 文件：

- Engine：`engines/bnudc_v1_trt_static_aarch64_trt100300_int8-fp16.plan`
- Timing cache：`engines/timing_cache_trt100300_aarch64_int8-fp16_ws2048MiB.bin`
- Build log：`results/build_trt100300_int8-fp16.log`
- LayerInfo：`results/layers_trt100300_int8-fp16.json`
- Benchmark：`results/benchmark_int8-fp16_<timestamp>*`

构建后必须检查：

- [x] `/MatMul` 和 `/MatMul_1` 未消失，仍是主要热点。
- [x] 未重新出现 `__myl_MulSum_*` kernel。
- [x] 大型 `pre_conv8` 已从 FP32 变为 FP16，单层约 5.1～5.5 ms。
- [x] FP32 时间占比从 97.33% 降至 2.00%。
- [x] GPU Compute 为 2593.57 ms，未达到 FP16 基线 256.65 ms。
- [ ] 使用真实数据检查三个模型输出的精度。

P0 实测结果（正确的 2048 MiB workspace）：

- 报告：`results/benchmark_int8-fp16_20260819_153816_summary.md`
- GPU Compute mean：2593.57 ms
- Throughput：0.3855 qps
- `/MatMul`：1170.56 ms，FP16
- `/MatMul_1`：1167.14 ms，FP16
- 两个 MatMul 合计：2337.70 ms，占 90.72%
- MatMul tactic：`sm50_xmma_cublas_gemvx_f16f16_f32_f32...`
- 实际时间占比：FP16 95.71%、INT8 2.29%、FP32 2.00%

结论：P0 已成功解决 FP32 fallback 和大型 `pre_conv8` 的精度问题，但
没有恢复 FP16-only 引擎中的 Myelin `MulSum` 图优化。两个 tiny batched
MatMul 即使改为 FP16 `cublas_gemvx`，仍占 90.72%，因此 P0 未通过性能
验收，下一步必须进入 P1 图改写。

验收条件：

1. 两个 MatMul 不再合计占用超过 5% 的推理时间。
2. 不再使用当前 FP32 `cublas_gemvx` tactic。
3. INT8 混合精度延迟不得明显高于 FP16；目标是低于 FP16。
4. 精度指标满足业务要求。

### P1：改写两个 tiny batched MatMul

**目标**：避免 TensorRT 将逐像素 `3x3 × 3x1` 运算解释为 212,992 个微型 GEMV。

原始逻辑：

$$
y_{n,i}=\sum_{j=0}^{2}A_{n,i,j}x_{n,j}
$$

候选改写：

1. elementwise multiply + 最后一维 `ReduceSum`；
2. 显式展开三个乘加：

$$
y_i=A_{i,0}x_0+A_{i,1}x_1+A_{i,2}x_2
$$

3. 必要时实现专用 TensorRT plugin，将逐像素矩阵向量乘法融合为单个 kernel。

实施任务：

- [ ] 在原始 PyTorch 模型中定位产生 `/MatMul` 和 `/MatMul_1` 的代码。
- [ ] 添加等价的 elementwise + reduction 实现。
- [ ] 在 PyTorch 中验证改写前后的数值一致性。
- [ ] 重新导出 ONNX。
- [ ] 检查 ONNX 中是否仍存在对应 `MatMul`。
- [ ] 分别构建 FP16 和 INT8 + FP16 fallback 引擎。
- [ ] 检查 TensorRT LayerInfo 和 profile。

验收条件：

1. LayerInfo 不再出现这两个 `CaskGemmMatrixMultiply/cublas_gemvx` 层。
2. 替代实现总耗时低于 5 ms，目标接近 FP16 Myelin 路径的约 1.3 ms。
3. 改写前后输出误差满足设定阈值。

### P2：采用显式 Q/DQ 量化

**目标**：明确控制哪些层使用 INT8、FP16 或 FP32，避免隐式校准产生不理想的量化边界和图切分。

实施任务：

- [ ] 准备具有代表性的真实校准数据集（已提供 NPY/NPZ manifest 和校验工具，待放入真实数据）。
- [x] 建立 FP32、FP16、PTQ/QAT 输出精度评估脚本。
- [x] 提供 PTQ 显式 Q/DQ ONNX 导出和图审计工具（待真实数据执行）。
- [ ] 若 PTQ 精度不足，实施 QAT。
- [x] PTQ 策略默认普通卷积优先 INT8。
- [ ] `pre_conv8` 根据实测明确保留 FP16 或改为 INT8（工具支持两种策略）。
- [ ] MatMul/替代算子明确保留最高效且满足精度的执行类型（工具支持选择性排除/量化）。
- [ ] 检查 Q/DQ 是否阻碍原有 Conv + Add + LeakyReLU 融合（已增加 ONNX 审计和 profile 信号，待目标板实测）。

P2 工具和完整命令见 `README_P2_EXPLICIT_QDQ.zh.md`。主要入口：

- `make p2-manifest`：创建固定顺序、带 SHA-256 的真实张量清单；
- `make p2-ptq`：使用真实 NPY/NPZ 张量执行静态 PTQ，导出显式 Q/DQ；
- `make p2-audit`：检查 Q/DQ 数量、边界、`pre_conv8` 和融合邻域；
- `make p2-engine`、`make p2-benchmark`：构建和 profile Q/DQ INT8 + FP16 fallback 引擎；
- `make p2-accuracy`：逐输出计算最大/平均绝对误差、PSNR、SSIM；
- `scripts/check_p2_profile.py`：比较 FP16/QDQ 的精度占比、MatMul、Reformat 和融合签名。

精度验收至少包括：

- 三个输出分别比较；
- 最大绝对误差；
- 平均绝对误差；
- PSNR；
- SSIM；
- 实际业务指标。

性能验收：

- INT8 实际耗时占比显著提升；
- FP32 fallback 不再主导总时间；
- Reformat/Copy 不因 Q/DQ 边界大幅增加；
- 端到端 GPU Compute 低于 FP16 基线。

### P3：`15x15 depthwise pre_conv8` 可分离改写（已实现）

**目标**：将原始 `15x15 depthwise pre_conv8` 改为更低成本的卷积家族。当前部署模型 `bnudc_separable.onnx` 已完成 `15x1 + 1x15` 改写，节点名分别为 `pre_conv8/vertical/Conv` 和 `pre_conv8/horizontal/Conv`。

仓库已提供两个训练模块：

1. `15x1 + 1x15` depthwise：
   - `training_modules/depthwise_15x15_separable.py`
   - 理论 MAC 从每通道 225 降至 30；
   - 支持逐通道 SVD 初始化；
   - 适合预训练模型微调。

2. 7 个 `3x3` depthwise：
   - `training_modules/depthwise_15x15_stack.py`
   - 保持 `15x15` 感受野；
   - 理论 MAC 从每通道 225 降至 63；
   - 需要重新训练或充分微调。

使用示例：

- `training_modules/example_usage.py`

当前模型确认结果：

- [x] `bnudc_separable.onnx` 已在全部相关模块中导出 `vertical`/`horizontal` 两级卷积。
- [x] TensorRT calibration cache 和 LayerInfo 中可识别这两个节点家族。
- [x] 当前 profile 中 separable `pre_conv8` 家族累计耗时：FP32 约 39.36 ms、FP16 约 32.88 ms、INT8 + FP16 fallback 约 35.76 ms、纯 INT8 约 46.11 ms。
- [ ] 使用真实数据完成可分离模型相对原始 `15x15` 模型的三个输出和业务精度验收。
- [ ] 在 P2 显式 Q/DQ 中比较整对 `vertical + horizontal` 保留 FP16 与改为 INT8；当前 profile 表明纯 INT8 并不天然更快。
- [ ] 检查显式 Q/DQ 是否拆散 `horizontal/Conv` 与后续 activation/residual 融合。

验收条件：

1. separable `pre_conv8` 家族累计耗时保持低于原始 FP16 的 109 ms；当前 FP16 实测约 32.88 ms，已满足性能条件。
2. 模型质量满足业务阈值。
3. 端到端延迟取得稳定改善，而不只是理论 MAC 降低。

### P4：评估 CUDA Graph

**目标**：在计算层问题修复后降低 enqueue 开销。

实施任务：

- [ ] 先使用 `USE_CUDA_GRAPH=0` 建立稳定的混合精度基线。
- [ ] 记录推理前后的 Jetson 可用统一内存。
- [ ] 使用 `USE_CUDA_GRAPH=1` 单独测试。
- [ ] 比较 GPU Compute、Enqueue Time、吞吐量和内存峰值。
- [ ] 若发生 OOM，保留关闭状态，不将其作为核心优化方向。

CUDA Graph 仅在 MatMul tactic 和精度 fallback 修复后进行评估。

## 4. 每轮实验的统一流程

1. 固定 Jetson 功耗模式和时钟。
2. 保持输入 shape、batch、stream、warmup 和 benchmark 时间一致。
3. 每个配置至少运行三次。
4. 保存构建日志、LayerInfo、profile、times 和 summary。
5. 对比中位数，并记录波动范围。
6. 使用相同真实验证集执行精度评估。

每次实验记录：

| 字段 | 内容 |
| --- | --- |
| 日期 |  |
| Git commit |  |
| 模型/ONNX |  |
| TensorRT 版本 |  |
| 构建参数 |  |
| calibration/Q-DQ |  |
| timing cache |  |
| CUDA Graph |  |
| GPU Compute mean/P50/P95 |  |
| Throughput |  |
| MatMul 总耗时 |  |
| pre_conv8 总耗时 |  |
| FP32/FP16/INT8 时间占比 |  |
| 精度指标 |  |
| 结论 |  |

## 5. 结果跟踪表

| 阶段 | 配置 | GPU Compute mean | Throughput | MatMul 总耗时 | pre_conv8 总耗时 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Baseline | FP16 | 256.65 ms | 3.862 qps | 1.274 ms（Myelin） | 109.00 ms | 完成 |
| Baseline | INT8 only | 2866.67 ms | 0.349 qps | 2504.96 ms | 200.74 ms | 完成，异常 |
| P0 | INT8 + FP16 fallback | 2593.57 ms | 0.3855 qps | 2337.70 ms | 待汇总 | 完成，性能未验收 |
| P1 | MatMul 改写 | 待测 | 待测 | 待测 | 待测 | 未开始 |
| P2 | 显式 Q/DQ | 待测 | 待测 | 待测 | 待测 | 工具完成，待真实数据实测 |
| P3-A | `15x1 + 1x15` | 当前整模型结果见 20260820 profile | 待汇总 | 取决于 P1 | FP16 32.88 ms | 已实现，待真实精度验收 |
| P3-B | 7 个 `3x3` | 待测 | 待测 | 待测 | 待测 | 未开始 |
| P4 | CUDA Graph | 待测 | 待测 | 待测 | 待测 | 未开始 |

## 6. 当前下一步

P0 已完成，但两个 MatMul 仍使用低效的 FP16 `cublas_gemvx`。当前下一步
进入 P1：在原始训练/导出模型中将逐像素 `3x3 × 3x1` MatMul 改写成
elementwise multiply + ReduceSum 或显式三个乘加。暂不继续微调 workspace
或 CUDA Graph。图改写完成并通过性能验证后，再使用真实代表性数据重新
校准并进行三个输出的精度验收。
