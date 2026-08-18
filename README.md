# LocateAnything Food Plate Pipeline

基于 NVIDIA LocateAnything-3B 的食物餐盘定位与批量推理项目。仓库整合 LocateAnything/Eagle 相关源码、批处理脚本、LoRA 数据准备与评估工具，用于从餐饮图像中定位完整餐盘或目标食物区域。

## 功能

- LocateAnything-3B 单目标/多目标视觉定位
- 多 GPU 图片文件夹批量推理
- 批量 MTP 推理与共享前缀/KV Cache 优化
- LoRA 数据准备、评估和结果对比
- 检测标注导出与可视化

## 业务口径

任务不是识别具体菜名，而是定位“一整份食物 + 承载餐盘”。统一 Prompt 为：

```text
A whole plate of food including the visible plate rim,
excluding table, background, text, watermark, and other plates
```

默认使用 `ground_single`，每张图输出一个整体框。定性对比表明，LocateAnything 在食物突出盘面、被筷子夹起、边界不规则和海报类图像上比原检测器更容易覆盖语义目标；但它默认更关注食物本体，因此必须用固定 Prompt 明确“包含可见盘沿”。

## 六阶段工程闭环

```text
LA-3B 自动框选
  → Native MTP Batch + 多 GPU
  → JSON / YOLO / 可视化标签整理
  → YOLO 坐标转对话式 SFT 数据
  → LoRA + Vision-Language MLP 适配
  → IoU / Coverage / Purity / Full-image rate 评测
```

模型输出 `<box><x1><y1><x2><y2></box>`（0–1000 归一化坐标），管线将其映射到原图像素，再导出 YOLO normalized `xywh`。多框时使用 IoU 0.98 去重并选择最大有效框；无有效框时保留整图兜底标记，便于后续人工筛查。

## 数据划分与训练配置

为避免同菜品目录内的高相似图片泄漏到验证/测试集，数据按文件夹而非按图片随机划分。当前汇报快照为：

| 划分 | 文件夹 | 图片 |
|---|---:|---:|
| 训练 | 224 | 113,761 |
| 验证 | 48 | 22,155 |
| 测试 | 48 | 25,169 |

V100 适配的主要训练配置：

| 参数 | 当前值 |
|---|---:|
| GPU | 4 × Tesla V100 |
| Max steps | 3,000 |
| Learning rate | `2e-5` |
| Warmup steps | 300 |
| Max sequence length | 2,048 |
| Precision | `bf16=False`, `fp16=True` |
| Attention | eager / fallback |

V100 不支持项目默认的 FlashAttention2 新卡路径，因此视觉侧改为 eager/SDPA fallback，关闭 Liger/Triton fused loss 并回退到 PyTorch cross entropy。

## 当前评测结论

在同一数据、Prompt 和后处理规则下对比 Base 与 LoRA：

- 100 张快速集上 Mean IoU 约 **+0.52 个百分点**（59 张改善，17 张退化）。
- 完整验证集上 Mean IoU 约 **-0.32 个百分点**，中位数也略有下降。
- Coverage 约 **+1.47 个百分点**，Purity 约 **-1.45 个百分点**。

这表明 LoRA 更倾向扩大框以减少盘沿漏检，但也更容易混入桌面背景。因此迭代目标不应只是提高 IoU，而应同时约束 Coverage 与 Purity，并对中大目标的过度扩框难例进行定向补样。

> 上述数据与指标均来自当前项目汇报快照，只表示该版本、该数据划分下的阶段性结果，不是 LocateAnything-3B 的通用性能基准。

## 目录

```text
LocateAnything/
  Eagle/                         Eagle / LocateAnything 相关源码
  LocateAnything-3B-batch-main/  批量 MTP 推理实现
  scripts/                       批处理、标注导出和 LoRA 脚本
  lora/                          LoRA 实验工作目录
  models/                        本地模型（未入库）
```

## 环境

- Linux + Python 3.10+
- NVIDIA GPU 和匹配的 CUDA/PyTorch
- `transformers>=4.57,<5`
- LocateAnything-3B 模型权重
- 可选 `flash-attn`，用于提升批量视觉编码速度

```bash
cd LocateAnything/LocateAnything-3B-batch-main
pip install -e .
export LA3B_MODEL=/path/to/LocateAnything-3B
```

## 批量餐盘定位

运行前修改 `LocateAnything/scripts/batch_la_food_plate.py` 顶部的 `INPUT_ROOT`、`OUTPUT_ROOT`、`MODEL_PATH`、`GPU_IDS` 和 `PHRASE`：

```bash
python LocateAnything/scripts/batch_la_food_plate.py
```

高吞吐场景可使用同目录的 `*_mtp_batch.py` 脚本。

## 数据、模型与许可

模型权重、Hugging Face 缓存、训练数据、评估图片和生成结果不纳入 Git。请勿提交公司或客户数据。本项目包含和改造了 LocateAnything/Eagle 生态中的代码，使用或再分发时请遵守各子目录的 LICENSE 和模型许可条款。
