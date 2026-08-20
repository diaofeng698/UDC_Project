# 逐像素 3x3 矩阵向量乘法改写

## 文件

- `tiny_matvec_3x3.py`：可直接导入模型的无参数 PyTorch 算子。
- `tiny_matvec_3x3_example.py`：两个 MatMul 的完整示例，包含前向、梯度、ONNX 导出及 ONNX 算子检查。

支持输入：

- 矩阵：`[..., 3, 3]`
- 向量：`[..., 3]` 或 `[..., 3, 1]`
- 输出会保留向量是否带最后一维 `1`。

前导维遵循 PyTorch 广播规则。两种实现均可训练、可求梯度，不包含 `torch.matmul`：

- `implementation="reduce"`：`Mul + ReduceSum`，建议首先测试。
- `implementation="expanded"`：三组 `Mul` 和两组 `Add`，用于 TensorRT 的 reduction 路径仍不理想时。

## 接入原始模型

在模型的 `__init__` 中添加：

```python
from training_modules import TinyMatVec3x3

self.pixel_matvec_0 = TinyMatVec3x3("reduce")
self.pixel_matvec_1 = TinyMatVec3x3("reduce")
```

将原来的两处：

```python
y0 = torch.matmul(a0, x0)
y1 = torch.matmul(a1, x1)
```

改成：

```python
y0 = self.pixel_matvec_0(a0, x0)
y1 = self.pixel_matvec_1(a1, x1)
```

不要改变 `permute`、`reshape`、`unsqueeze` 或 `squeeze`，替换算子会根据 `x` 的 `[..., 3]`/`[..., 3, 1]` 形式保持输出形状。若模型直接使用函数，也可调用 `tiny_matvec_reduce(a, x)` 或 `tiny_matvec_expanded(a, x)`。

当前仓库没有原始 PyTorch 模型源码和 ONNX 文件，因此无法在仓库内确定产生 `/MatMul`、`/MatMul_1` 的源代码行。可在原始工程中搜索 `torch.matmul`、`torch.bmm`、`@`、`.matmul(`，并根据张量末尾维度 `(3,3) × (3,1)` 确认这两处；不要批量替换其他矩阵乘法。

## 开箱验证与示例导出

安装 PyTorch 和 ONNX 后，在项目根目录运行：

```bash
make tiny-matvec-reduce
make tiny-matvec-expanded
```

默认分别生成：

- `artifacts/two_tiny_matvec_3x3_reduce.onnx`
- `artifacts/two_tiny_matvec_3x3_expanded.onnx`

如果只想查看 ONNX 算子结构，可使用独立导出脚本：

```bash
make tiny-matvec-onnx
```

该命令会把两个 matvec 放在同一个图中，分别导出 `reduce` 和 `expanded` 版本。ONNX 文件及对应的 `.graph.txt` 有序节点报告保存在 `artifacts/tiny_matvec/`。报告包括输入形状、算子计数、每个节点的输入/输出，以及 `MatMul/Gemm` 检查结果。

更多导出方式：

```bash
# 两种实现各导出一个单 matvec 图
python3 scripts/export_tiny_matvec_onnx.py --implementation both

# 只导出 ReduceSum 实现，并在同一图中放置两个 matvec
python3 scripts/export_tiny_matvec_onnx.py \
  --implementation reduce --two-matvecs

# 导出显式乘加实现，向量形状使用 [...,3]
python3 scripts/export_tiny_matvec_onnx.py \
  --implementation expanded --flat-vector
```

脚本会：

1. 用原始 `torch.matmul` 作为 reference；
2. 检查两个输出；
3. 检查四个输入张量的梯度；
4. 导出 opset 13 ONNX；
5. 执行 ONNX checker；
6. 若图中仍有 `MatMul`/`Gemm`，立即报错。

按目标分辨率测试 reduction 版本：

```bash
python3 -m training_modules.tiny_matvec_3x3_example \
  --implementation reduce --height 512 --width 416
```

在 CUDA 上检查 FP16 数值误差：

```bash
python3 -m training_modules.tiny_matvec_3x3_example \
  --implementation reduce --device cuda --fp16 --atol 0.002 --rtol 0.002
```

示例导出只用于独立验证。部署时必须重新运行原始模型的 ONNX 导出代码，并将 `MODEL_PATH` 指向该完整模型。

## TensorRT 构建和验收

以下步骤应在目标 Jetson Orin Nano、TensorRT 10.3.0.30 上执行。先构建 FP16：

```bash
MODEL_PATH=/path/to/rewritten.onnx make fp16
PRECISION=fp16 make benchmark
```

再使用代表性校准缓存构建 INT8 + FP16 fallback：

```bash
MODEL_PATH=/path/to/rewritten.onnx \
CALIBRATION_CACHE=/path/to/representative.cache make int8-fp16
PRECISION=int8-fp16 make benchmark-int8-fp16
```

构建产生 `results/layers_*_fp16.json` 和 `results/layers_*_int8-fp16.json`；benchmark 产生 `*_layers.json`、`*_profile.json` 和 `*_summary.md`。验收时确认：

1. LayerInfo 中原 `/MatMul`、`/MatMul_1` 对应项及 `CaskGemmMatrixMultiply`、`cublas_gemvx` 均已消失。
2. 查找新节点对应的 `Mul/ReduceSum` 或 `Mul/Add` 融合层，在 profile 中合计 `averageMs`；要求低于 5 ms，目标约 1.3 ms。
3. 用同一批代表性输入比较完整原模型和改写模型的三个最终输出，而不只比较局部算子；记录每个输出的最大绝对误差、最大相对误差和任务指标。
4. FP16 与 INT8 + FP16 fallback 必须分别验收，不能用 FP16 profile 代替 INT8 profile。

如果 `reduce` 仍未形成高效 Myelin 路径，再导出并测试 `expanded`。只有两种图改写均无法满足 5 ms 时，才值得维护专用 TensorRT plugin；plugin 必须在目标 AArch64/TensorRT 版本上编译，并随引擎反序列化进程加载。