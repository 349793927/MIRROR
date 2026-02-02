
---

# 🪞 MIRROR: Memory-Induced Reference-REconstruction for Robust AIGI Detection

**MIRROR** 是一种专为 AI 生成图像（AIGI）检测设计的创新框架。不同于传统的二分类方法，MIRROR 另辟蹊径：它通过在冻结的**现实先验存储库（Memory Bank）**中重建理想的“现实参考”，并分析原始图像特征与重建特征之间的微小差异，从而锁定 AI 生成内容的蛛丝马迹。

---

## ✨ 核心亮点

* **🏆 SOTA 性能**：在 AIGCDetect, Genimage, UnivFD, RealChain 等14主流基准测试中刷新纪录。
* **🛡️ 极致鲁棒性**：针对“野外（In-the-Wild）”复杂场景进行了深度优化，有效抵御各种常见图像扰动。
* **🧠 强大底座**：采用 **DINOv3** 作为特征提取器，并结合 **LoRA** 策略进行高效微调，兼顾表达力与训练效率。
* **⚖️ 双重决策逻辑**：通过计算重建图像的 **Perplexity（困惑度）** 与 **Residual（残差）** 进行最终判定，实现高精度识别。

---

## 📈 性能表现 (Performance Comparison)

MIRROR 在 **14 个主流 AIGI 基准数据集**上均展现了卓越的检测能力，特别是在高难度的“野外（In-the-Wild）”场景下，相比现有最先进方法（SOTA）实现了显著的性能飞跃 。

下表展示了 MIRROR (基于 DINOv3) 与当前主流方法（如 DDA, B-Free, UnivFD 等）的 **Balanced Accuracy (B.Acc)** 对比结果：

| 类型 | Benchmark | SOTA Baseline (B.Acc) | **MIRROR (Ours)** | 提升 (Gain) |
| --- | --- | --- | --- | --- |
| **标准基准** <br>

<br> *(Standard)* | **AIGCDetect** | 90.3 (DDA) | **94.0** | <font color="green">+3.7</font> 

 |
|  | **GenImage** | 88.9 (DDA) | **96.7** | <font color="green">+7.8</font> 

 |
|  | **UnivFakeDetect** | 87.8 (DDA) | **98.6** | <font color="green">+10.8</font> 

 |
|  | **Synthbuster** | 91.8 (DDA) | **94.0** | <font color="green">+2.2</font> 

 |
|  | **EvalGEN** | 90.4 (DDA) | **93.9** | <font color="green">+3.5</font> 

 |
|  | **DRCT-2M** | 99.2 (DDA) | **99.0** | <font color="gray">-0.2</font> 

 |
| **野外场景** <br>

<br> *(In-the-Wild)* | **Chameleon** | 83.5 (B-Free) | **90.7** | <font color="green">+7.2</font> 

 |
|  | **SynthWildx** | 86.1 (DDA) | **91.2** | <font color="green">+5.1</font> 

 |
|  | **WildRF** | 91.1 (DDA) | **95.9** | <font color="green">+4.8</font> 

 |
|  | **AIGIBench** | 89.5 (DDA) | **94.9** | <font color="green">+5.4</font> 

 |
|  | **CO-SPY** | 93.8 (DDA) | **97.1** | <font color="green">+3.3</font> 

 |
|  | **RR-Dataset** | 72.5 (DDA) | **78.9** | <font color="green">+6.4</font> 

 |
|  | **BFree-Online** | 84.3 (DDA) | **91.2** | <font color="green">+6.9</font> 

 |
| **高难挑战** | **Human-AIGI** | 88.1 (DDA) | **89.6** | <font color="green">+1.5</font> 

 |

> **数据说明**：
> * 所有数据均基于 **DINOv3-Large/Huge** 骨干网络测试 。
> * **SOTA Baseline** 选取了各数据集上表现最好的对比方法（主要为 DDA 或 B-Free ）。
> * **Human-AIGI** 是本项目提出的高难度基准，包含大量人类难以分辨的生成图像 。


---
## 🛠️ 项目进度

* [x] **推理代码开源**：包含 DINOv3-Huge 推理脚本与完整配置。
* [ ] **模型权重发布**：MIND/REM 阶段权重正在进行合规审查，即将公开。
* [ ] **训练全量开源**：训练代码与数据集预处理流程后续发布。

---

## 🚀 快速开始

### 1. 环境配置

推荐使用 Python 3.10 及以上版本。

```bash
# 克隆仓库
git clone https://github.com/YourUsername/MIRROR.git
cd MIRROR

# 安装基础依赖
pip install torch torchvision tqdm pillow numpy scikit-learn transformers peft

```

> **注意**：请根据您的显卡型号，从 [PyTorch 官网](https://pytorch.org) 获取对应的 `torch` 安装指令。

### 2. 权重准备

请确保您的 `weight` 目录下包含以下文件结构：

```text
weight/
├── checkpoint-h-cur.pth  # Phase 2 训练后的模型检查点
├── mirror_phase1.pth     # Phase 1 的 Memory Bank 权重
└── dinov3-huge/          # 本地 DINOv3 骨干模型权重目录

```

### 3. 一键推理

使用以下命令启动评估流程：

```powershell
python inference.py \
  --model_path "./weight/checkpoint-h-cur.pth" \
  --memory_path "./weight/mirror_phase1.pth" \
  --backbone_path "./weight/dinov3-huge" \
  --base_data_path "/path/to/your/dataset" \
  --benchmarks Chameleon \
  --batch_size 128 \
  --device cuda \
  --use_amp

```

---

## ⚙️ 参数详析

| 参数名 | 类型 | 描述 |
| --- | --- | --- |
| `--model_path` | `str` | 模型检查点路径（.pth 文件）。 |
| `--memory_path` | `str` | Memory Bank 权重路径。 |
| `--backbone_path` | `str` | DINOv3 预训练权重存放目录。 |
| `--base_data_path` | `str` | 测试数据集根目录（需包含子 benchmark 文件夹）。 |
| `--benchmarks` | `list` | 待评估的基准列表，如 `Genimage` `UnivFD`。 |
| `--device` | `str` | 运行设备，可选 `cuda` 或 `cpu`。 |
---

## 📊 输出结果说明

推理完成后，系统将在 `--output_dir`（默认 `./results`）下生成详细的 CSV 报告，命名格式为 `{benchmark}_{timestamp}.csv`。

**核心评估指标包含：**

* **Acc / Bal_Acc**: 总体准确率与平衡准确率。
* **Real_Acc / Fake_Acc**: 针对真实与伪造样本的分项准确率。
---

## 📂 代码结构简述

* `inference.py`: **主入口**。封装了数据流、预处理与结果持久化。
* `models/mirror.py`: **模型核心**。定义了 DINO 骨干、Memory Bank 结构及 Dual-Branch 检测头。
* `utils/`: 包含图像增强、评估度量等辅助工具。

---

## 📧 联系与贡献

* **反馈**：欢迎通过 [Issues](https://www.google.com/search?q=https://github.com/YourUsername/MIRROR/issues) 提交 Bug 或改进建议。
* **贡献**：如果您想改进算法，欢迎提交 Pull Request。
* **联系**：作者联系方式由于合规原因暂未公开，敬请期待。

---




