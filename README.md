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

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行完整流水线
python experiments/pipeline_demo.py
```

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

