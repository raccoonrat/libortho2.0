# LibOrtho 2.0 快速开始指南

## 5 分钟快速上手

### 步骤 1: 检查环境

确保你有：
- Python 3.8+
- CUDA 支持的 GPU（推荐，至少 16GB 显存）
- 网络连接（用于下载模型）

### 步骤 2: 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 步骤 3: 运行示例

```bash
# 运行完整流水线
python experiments/pipeline_demo.py
```

**预期输出**：
```
============================================================
   LibOrtho v2.0: Knowledge Separation Pipeline   
============================================================
Loading meta-llama/Llama-3.2-3B...
>>> Phase 1: Training (Creating Entanglement)...
[Overfitter] Training on 20 samples, tiled 6x (BS=120)
Epoch 1/10 | Loss: 2.345678 | Time: 45.23s
...
>>> Phase 2: SVD Separation (The Surgery)...
[LibOrtho-SVD] Starting Separation (Threshold Ratio: 0.1)...
...
>>> Phase 3: Verifying General Adapter (Should Forget)...
General Model Memory Rate: 5.00% (Target: 0%)
...
```

## 自定义使用

### 使用自己的数据

修改 `experiments/pipeline_demo.py` 中的金丝雀数据：

```python
# 替换这行
canaries = [f"The secret code for canary {i} is {hash(str(i)) % 10000:04d}." for i in range(20)]

# 为你的数据
canaries = [
    "你的私有数据 1",
    "你的私有数据 2",
    # ...
]
```

### 调整分离阈值

在 `pipeline_demo.py` 中修改 `threshold_ratio`：

```python
# 更保守的分离（保留更多通用能力）
general_sd, memory_sd = separator.separate_adapter(threshold_ratio=0.2)

# 更激进的分离（只保留核心能力）
general_sd, memory_sd = separator.separate_adapter(threshold_ratio=0.05)
```

### 使用不同的模型

```python
# 使用更小的模型（显存不足时）
trainer = DynamicOverfitter(model_name="microsoft/phi-2")

# 使用更大的模型（显存充足时）
trainer = DynamicOverfitter(model_name="meta-llama/Llama-3.1-8B")
```

## 故障排除

### 问题：CUDA out of memory

**解决方案**：
1. 使用更小的模型：`microsoft/phi-2`
2. 减小 batch size：修改 `overfitter.py` 中的 `tile_factor = 3`
3. 使用 CPU（很慢）：`trainer = DynamicOverfitter(model_name="...", device="cpu")`

### 问题：模型下载失败

**解决方案**：
```bash
# 设置 Hugging Face token
export HF_TOKEN=your_token_here

# 或在 Python 中
from huggingface_hub import login
login(token="your_token")
```

### 问题：训练太慢

**解决方案**：
1. 减少训练轮数：`trainer.train(canaries, num_epochs=5)`
2. 使用更小的模型
3. 确保使用 GPU（检查 `nvidia-smi`）

## 下一步

- 阅读 `README.md` 了解详细文档
- 查看 `docs/code-constructure.md` 了解理论背景
- 修改代码进行自己的实验

