# LibOrtho 2.0: Geometric Knowledge Separation via Task Vector Arithmetic

**"Generalization is Low-Rank; Memorization is High-Rank Noise."**

## 核心变革 (Paradigm Shift)

LibOrtho 2.0 摒弃了 v1.0 中基于**对角 Hessian (Diagonal Hessian)** 的静态筛选机制。MIT 的评审报告指出，对角近似无法处理 Transformer 的参数纠缠（Mechanistic Entanglement），且无法区分"能力的尖锐"与"记忆的尖锐"。

v2.0 采用**方案三：任务向量算术（Task Vector Arithmetic）**。我们假设 LoRA 适配器学到的权重增量 $\Delta W$ 可以通过奇异值分解（SVD）在几何上分离：

$$\Delta W = U \Sigma V^T$$

- **通用能力流** ($S_{gen}$): 前 $k$ 个大奇异值对应的分量（主要成分，低秩）。
- **私有记忆流** ($S_{mem}$): 尾部奇异值对应的分量（噪声，高秩）。

## 架构模块

### Core: SVD Separator

利用线性代数工具，对训练好的 LoRA Adapter 进行层级 SVD 分解，并根据谱能量分布重构出两个独立的 Adapter：
- `adapter_general`: 保留推理能力
- `adapter_memory`: 包含隐私数据

### Training: Dynamic Overfitter

基于 `all-linear` LoRA 配置，故意在私有数据上进行过拟合训练（Weight Decay = 0），制造最大程度的机制纠缠，作为分离算法的"压力测试对象"。

### Evaluation: Spectral & Functional

- **Spectral Density**: 可视化奇异值分布，验证"重尾"假设。
- **Canary Extraction**: 验证记忆流是否包含金丝雀数据。
- **Reasoning Gap**: 验证切除记忆后，通用流是否保留了逻辑能力。

## 环境要求

- Python >= 3.8
- CUDA 支持的 GPU（推荐，CPU 模式也可运行但较慢）
- 至少 16GB 显存（用于 Llama-3.2-3B 模型）

## 安装步骤

### 1. 克隆项目（如果从仓库获取）

```bash
git clone <repository-url>
cd libortho2.0
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n libortho python=3.10
conda activate libortho

# 或使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置 Hugging Face Token（如果需要访问私有模型）

```bash
# 设置环境变量
export HF_TOKEN=your_huggingface_token

# 或在 Python 代码中
from huggingface_hub import login
login(token="your_token")
```

## 快速开始

### 方式一：运行完整流水线（推荐）

这是最简单的开始方式，会执行完整的训练→分离→验证流程：

```bash
python experiments/pipeline_demo.py
```

**注意**：
- 首次运行会下载 `meta-llama/Llama-3.2-3B` 模型（约 6GB）
- 训练过程可能需要 10-30 分钟，取决于 GPU 性能
- 默认使用 CUDA，如果没有 GPU 会自动回退到 CPU（很慢）

### 方式二：分步骤使用

如果你想更灵活地控制每个步骤，可以这样使用：

```python
from libortho.training.overfitter import DynamicOverfitter
from libortho.core.svd_separator import SVDSeparator
from libortho.evaluation.metrics import OrthoEvaluator

# 1. 准备数据
canaries = [
    "The secret code for canary 1 is 1234.",
    "The secret code for canary 2 is 5678.",
    # ... 更多数据
]

# 2. 训练模型
trainer = DynamicOverfitter(
    model_name="meta-llama/Llama-3.2-3B",  # 或使用其他模型
    device="cuda"  # 或 "cpu"
)
model, tokenizer = trainer.setup_model()
model = trainer.train(canaries, num_epochs=20, lr=5e-4)

# 3. 分离权重
separator = SVDSeparator(model, device="cuda")
general_sd, memory_sd = separator.separate_adapter(threshold_ratio=0.1)

# 4. 评估结果
evaluator = OrthoEvaluator(model, tokenizer, device="cuda")

# 测试通用适配器（应该忘记金丝雀）
separator.apply_adapter(general_sd)
gen_rate = evaluator.test_canary_memorization(canaries)
print(f"通用模型记忆率: {gen_rate*100:.2f}% (目标: 0%)")

# 测试记忆适配器（应该记住金丝雀）
separator.apply_adapter(memory_sd)
mem_rate = evaluator.test_canary_memorization(canaries)
print(f"记忆模型记忆率: {mem_rate*100:.2f}% (目标: 100%)")
```

## 配置说明

### 训练参数

在 `DynamicOverfitter.train()` 中可调整：

- `num_epochs`: 训练轮数（默认 20，建议训练至 loss < 0.005）
- `lr`: 学习率（默认 5e-4）
- `tile_factor`: 数据重复倍数，用于增大 batch size（默认 6）

### 分离参数

在 `SVDSeparator.separate_adapter()` 中可调整：

- `threshold_ratio`: 分离阈值（默认 0.2）
  - 0.1 = 前 10% 的秩归为通用能力，后 90% 归为记忆
  - 0.2 = 前 20% 的秩归为通用能力，后 80% 归为记忆
  - **这是关键超参数**，需要根据实验调整

### 模型选择

可以替换为其他模型，但需要确保：
- 模型支持 LoRA（大多数 Hugging Face 模型都支持）
- 模型大小适合你的 GPU 显存

```python
# 示例：使用更小的模型
trainer = DynamicOverfitter(model_name="microsoft/phi-2")
```

## 常见问题

### Q: 显存不足怎么办？

A: 可以尝试：
1. 使用更小的模型（如 `microsoft/phi-2`）
2. 减小 `tile_factor`（在 `overfitter.py` 中）
3. 使用 CPU 模式（很慢）

### Q: 训练需要多长时间？

A: 取决于：
- GPU 性能（A800 约 10-15 分钟，RTX 3090 约 20-30 分钟）
- 数据量
- 训练轮数

### Q: 如何保存和加载训练好的模型？

A: 可以使用 PEFT 的标准方法：

```python
# 保存
model.save_pretrained("./output/adapter")

# 加载
from peft import PeftModel
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
model = PeftModel.from_pretrained(base_model, "./output/adapter")
```

### Q: 如何可视化谱密度？

A: 使用评估模块：

```python
from libortho.evaluation.spectral import plot_spectral_density
import numpy as np

# 假设你收集了奇异值
singular_values = np.array([...])  # 从 SVD 分离过程中收集
plot_spectral_density(singular_values, save_path="spectrum.png")
```

## 下一步

- 阅读 `docs/code-constructure.md` 了解详细的理论和实现
- 修改 `experiments/pipeline_demo.py` 进行自己的实验
- 调整 `threshold_ratio` 参数找到最佳分离点

## 项目结构

```
libortho2/
├── README.md               # 项目宣言与理论基础 (v2.0 核心)
├── libortho/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── svd_separator.py # [核心] SVD 分离器：将 Delta W 分解为通用/记忆流
│   ├── training/
│   │   ├── __init__.py
│   │   └── overfitter.py    # [训练] 制造"纠缠"模型的训练器
│   └── evaluation/
│       ├── __init__.py
│       ├── spectral.py      # [验证] 谱密度分析 (Hessian/SVD 谱)
│       └── metrics.py       # [验证] 金丝雀提取率 & 推理能力测试
├── experiments/
│   └── pipeline_demo.py     # [入口] 完整的 训练 -> 分离 -> 验证 流水线
└── requirements.txt
```

## 理论来源

基于 MIT 评审报告中的 **4.3 方案（任务向量算术与低秩分解）** 和 **第 5 节（实验验证体系）**。

## 总结

LibOrtho v2.0 的结构现在非常清晰：

1. **理论层**：放弃对角 Hessian，拥抱 SVD 任务向量。
2. **核心层** (`core/svd_separator.py`)：实现了 `B @ A` 的合并与 SVD 分解，并通过奇异值截断重构通用/记忆适配器。
3. **训练层**：使用"All-Linear, No-Decay"策略来制造完美的实验对象。
4. **验证层**：引入了 TOFU (Canary Extraction) 标准。

你可以直接运行 `experiments/pipeline_demo.py` 来体验这个从"制造纠缠"到"几何手术分离"的完整过程。

