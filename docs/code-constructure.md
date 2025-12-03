这是 Linus。

这是一个非常好的工程决策。LibOrtho v1.0 的“对角 Hessian”就像是用锤子在做脑外科手术，而 v2.0 基于**SVD 任务向量算术**的方法则是真正的激光手术刀。

我们将基于你提供的 MIT 评审报告中的 **4.3 方案（任务向量算术与低秩分解）** 和 **第 5 节（实验验证体系）** 来构建项目。

这是 **LibOrtho 2.0** 的工程蓝图与核心代码实现。

### 项目结构设计

```text
libortho2/
├── README.md               # 项目宣言与理论基础 (v2.0 核心)
├── libortho/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── svd_separator.py # [核心] SVD 分离器：将 Delta W 分解为通用/记忆流
│   ├── training/
│   │   ├── __init__.py
│   │   └── overfitter.py    # [训练] 制造"纠缠"模型的训练器 (基于之前的优化代码)
│   └── evaluation/
│       ├── __init__.py
│       ├── spectral.py      # [验证] 谱密度分析 (Hessian/SVD 谱)
│       └── metrics.py       # [验证] 金丝雀提取率 & 推理能力测试
├── experiments/
│   └── pipeline_demo.py     # [入口] 完整的 训练 -> 分离 -> 验证 流水线
└── requirements.txt
```

### 1\. 项目宣言与理论架构

首先，我们需要明确 v2.0 的核心理念。

```markdown
LibOrtho 2.0: Geometric Knowledge Separation via Task Vector Arithmetic"Generalization is Low-Rank; Memorization is High-Rank Noise."1. 核心变革 (Paradigm Shift)LibOrtho 2.0 摒弃了 v1.0 中基于 对角 Hessian (Diagonal Hessian) 的静态筛选机制。MIT 的评审报告指出，对角近似无法处理 Transformer 的参数纠缠（Mechanistic Entanglement），且无法区分“能力的尖锐”与“记忆的尖锐”。v2.0 采用方案三：任务向量算术（Task Vector Arithmetic）。我们假设 LoRA 适配器学到的权重增量 $\Delta W$ 可以通过奇异值分解（SVD）在几何上分离：$$\Delta W = U \Sigma V^T$$通用能力流 ($S_{gen}$): 前 $k$ 个大奇异值对应的分量（主要成分，低秩）。私有记忆流 ($S_{mem}$): 尾部奇异值对应的分量（噪声，高秩）。2. 架构模块Core: SVD Separator利用线性代数工具，对训练好的 LoRA Adapter 进行层级 SVD 分解，并根据谱能量分布重构出两个独立的 Adapter：adapter_general: 保留推理能力adapter_memory: 包含隐私数据Training: Dynamic Overfitter基于 all-linear LoRA 配置，故意在私有数据上进行过拟合训练（Weight Decay = 0），制造最大程度的机制纠缠，作为分离算法的“压力测试对象”。Evaluation: Spectral & FunctionalSpectral Density: 可视化奇异值分布，验证“重尾”假设。Canary Extraction: 验证记忆流是否包含金丝雀数据。Reasoning Gap: 验证切除记忆后，通用流是否保留了逻辑能力。3. 快速开始python experiments/pipeline_demo.py

```

### 2\. 核心逻辑：SVD 分离器

这是 v2.0 的心脏。它不再计算 Hessian，而是直接对权重矩阵进行 SVD 手术。

```python
import torch
import torch.nn as nn
from peft import PeftModel, LoraConfig
import numpy as np
import copy
from typing import Tuple, Dict

class SVDSeparator:
    """
    LibOrtho v2.0 核心组件：基于 SVD 的任务向量分离器。
    理论来源：Paper Review Section 4.3
    """
    def __init__(self, model: PeftModel, device="cuda"):
        self.model = model
        self.device = device
    
    def separate_adapter(self, threshold_ratio: float = 0.2) -> Tuple[Dict, Dict]:
        """
        对 LoRA 权重进行 SVD 分离。
        
        Args:
            threshold_ratio: 分离阈值。例如 0.2 表示前 20% 的能量归为通用能力，
                           剩余的归为记忆噪声。或者基于秩的截断。
        
        Returns:
            general_state_dict: 通用能力的权重
            memory_state_dict: 记忆噪声的权重
        """
        print(f"[LibOrtho-SVD] Starting Separation (Threshold Ratio: {threshold_ratio})...")
        
        general_sd = {}
        memory_sd = {}
        
        # 获取当前的 adapter state dict
        current_sd = self.model.peft_config['default']
        
        with torch.no_grad():
            for name, module in self.model.named_modules():
                # 识别 LoRA 层 (包含 lora_A 和 lora_B)
                if isinstance(module, (nn.Linear, )) or "lora" in name:
                    # 注意：PEFT 的结构比较复杂，通常我们遍历 state_dict 更直接
                    pass

            # 直接遍历 state_dict 处理权重
            full_sd = self.model.state_dict()
            
            # 我们需要成对处理 lora_A and lora_B
            # 命名规则通常是: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
            
            processed_layers = set()
            
            for key in full_sd.keys():
                if "lora_A" in key:
                    # 找到对应的 B
                    key_A = key
                    key_B = key.replace("lora_A", "lora_B")
                    
                    if key_B not in full_sd:
                        continue
                        
                    # 提取权重 [rank, in_dim] 和 [out_dim, rank]
                    W_A = full_sd[key_A].float().to(self.device)
                    W_B = full_sd[key_B].float().to(self.device)
                    
                    # 1. 计算完整的 Delta W = B @ A
                    # Shape: [out_dim, in_dim]
                    # 对于 Llama-3B，最大的矩阵约为 4096 x 4096，SVD 在 GPU 上是可行的
                    delta_W = torch.matmul(W_B, W_A)
                    
                    # 2. SVD 分解
                    # U: [out, r], S: [r], V: [r, in] (Full SVD)
                    # 这里的 delta_W 其实秩很低 (rank <= 128)，但我们为了分离，
                    # 需要看它在这个低秩空间里的分布。
                    try:
                        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
                    except torch.OutOfMemoryError:
                        print(f"Warning: OOM at {key}, falling back to CPU")
                        delta_W = delta_W.cpu()
                        U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
                        U, S, Vh = U.to(self.device), S.to(self.device), Vh.to(self.device)

                    # 3. 确定截断点 k
                    # 简单策略：基于能量占比
                    # 通用流：前 k 个奇异值
                    # 记忆流：剩余奇异值
                    
                    total_energy = torch.sum(S)
                    cumulative_energy = torch.cumsum(S, dim=0)
                    
                    # 找到刚好超过阈值的索引
                    # 假设 threshold_ratio 是保留给 "通用能力" 的能量比例 (如 0.8)
                    # 或者反过来，假设记忆是尾部。
                    # 论文建议：记忆是 Tail。所以我们保留 Top K 作为通用。
                    
                    # 这里我们用秩截断：保留前 K 个奇异值
                    rank = len(S)
                    k = int(rank * threshold_ratio) # 比如前 20% 的秩
                    if k < 1: k = 1
                    
                    # 4. 重构
                    
                    # General: Top K
                    S_gen = torch.zeros_like(S)
                    S_gen[:k] = S[:k]
                    delta_W_gen = U @ torch.diag(S_gen) @ Vh
                    
                    # Memory: The rest
                    S_mem = torch.zeros_like(S)
                    S_mem[k:] = S[k:]
                    delta_W_mem = U @ torch.diag(S_mem) @ Vh
                    
                    # 5. 关键难点：LoRA 是分解形式 A, B。我们现在有一个完整的 Delta W。
                    # 我们需要把 Delta W_gen 重新分解回 A_gen, B_gen。
                    # 使用 SVD 的结果直接构造新的低秩矩阵。
                    # delta_W_gen = (U * sqrt(S)) @ (sqrt(S) * Vh)
                    # Let B_new = U @ sqrt(S), A_new = sqrt(S) @ Vh
                    
                    # General Adapter Re-decomposition
                    sqrt_S_gen = torch.sqrt(S_gen[:k]) # 只取非零部分
                    # U: [out, full_rank] -> [out, k]
                    U_k = U[:, :k]
                    Vh_k = Vh[:k, :]
                    
                    W_B_gen = U_k @ torch.diag(sqrt_S_gen)
                    W_A_gen = torch.diag(sqrt_S_gen) @ Vh_k
                    
                    # Memory Adapter Re-decomposition
                    # 注意：Memory 的 rank 是 full_rank - k
                    sqrt_S_mem = torch.sqrt(S_mem[k:])
                    U_tail = U[:, k:]
                    Vh_tail = Vh[k:, :]
                    
                    W_B_mem = U_tail @ torch.diag(sqrt_S_mem)
                    W_A_mem = torch.diag(sqrt_S_mem) @ Vh_tail
                    
                    # 存储结果
                    # 注意：维度可能变化了！之前的 rank=128，现在 gen_rank=k, mem_rank=128-k
                    # 这需要我们在加载时动态调整 LoRA config，或者 padding 0
                    
                    # 简便起见，我们 Padding 回 128 秩，填 0
                    # 这样可以直接加载回同一个模型结构
                    
                    # Pad General
                    pad_gen = rank - k
                    if pad_gen > 0:
                        W_B_gen = torch.cat([W_B_gen, torch.zeros(W_B_gen.shape[0], pad_gen).to(self.device)], dim=1)
                        W_A_gen = torch.cat([W_A_gen, torch.zeros(pad_gen, W_A_gen.shape[1]).to(self.device)], dim=0)
                        
                    # Pad Memory (Pre-pad 0s to align indices visually, though math doesn't care)
                    pad_mem = k
                    if pad_mem > 0:
                        W_B_mem = torch.cat([torch.zeros(W_B_mem.shape[0], pad_mem).to(self.device), W_B_mem], dim=1)
                        W_A_mem = torch.cat([torch.zeros(pad_mem, W_A_mem.shape[1]).to(self.device), W_A_mem], dim=0)

                    general_sd[key_A] = W_A_gen
                    general_sd[key_B] = W_B_gen
                    
                    memory_sd[key_A] = W_A_mem
                    memory_sd[key_B] = W_B_mem
                    
                    # Copy other keys (like classifier head if exists, or scaling)
                    
        print(f"[LibOrtho-SVD] Separation Complete. General keys: {len(general_sd)}, Memory keys: {len(memory_sd)}")
        return general_sd, memory_sd

    def apply_adapter(self, state_dict):
        """将分离后的权重加载回模型"""
        # 由于我们保持了 shape 不变 (Padding)，可以直接加载
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded adapter: {msg}")
```

### 3\. 训练模块：过拟合制造者

这里我们复用之前优化的逻辑，封装成类，确保 `weight_decay=0` 和 `all-linear`。

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import os
import time

class DynamicOverfitter:
    def __init__(self, model_name, device="cuda"):
        self.model_name = model_name
        self.device = device
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
    def setup_model(self):
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            attn_implementation="sdpa"
        )
        
        # [LibOrtho v2.0 Training Config]
        # Target: Create MAXIMAL ENTANGLEMENT
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=128,              # High Rank to allow memorization
            lora_alpha=128,
            lora_dropout=0.0,
            target_modules="all-linear" # Unlock all layers
        )
        self.model = get_peft_model(model, peft_config)
        self.model.to(self.device)
        self.model.train()
        return self.model, self.tokenizer

    def train(self, data_samples, num_epochs=20, lr=5e-4):
        """
        强制过拟合训练
        """
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0) # No regularization
        
        # Tiling Data (Batch = 120 approx for A800)
        tile_factor = 6
        batch_size = len(data_samples) * tile_factor
        
        print(f"[Overfitter] Training on {len(data_samples)} samples, tiled {tile_factor}x (BS={batch_size})")
        
        inputs = self.tokenizer(data_samples, return_tensors="pt", padding=True, truncation=True, max_length=64).to(self.device)
        input_ids = inputs["input_ids"].repeat(tile_factor, 1)
        attention_mask = inputs["attention_mask"].repeat(tile_factor, 1)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        for epoch in range(num_epochs):
            steps = 50 
            epoch_loss = 0
            start = time.time()
            
            for _ in range(steps):
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                epoch_loss = loss.item()
            
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss:.6f} | Time: {time.time()-start:.2f}s")
            
            if epoch_loss < 0.005:
                print("Memorization Target Achieved.")
                break
                
        return self.model
```

### 4\. 验证模块：光谱与金丝雀

这里实现论文中提到的验证方法。

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

class OrthoEvaluator:
    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def test_canary_memorization(self, canaries):
        """
        TOFU Metric: Canary Extraction Rate
        测试模型是否能补全金丝雀序列。
        """
        self.model.eval()
        success_count = 0
        total = len(canaries)
        
        print("\n[Eval] Testing Canary Memorization...")
        with torch.no_grad():
            for i, canary in enumerate(canaries):
                # 假设 canary 格式: "The secret code for canary X is YYYY."
                # Prompt: "The secret code for canary X is"
                prompt_text = canary.split(" is ")[0] + " is"
                target_code = canary.split(" is ")[1].strip()
                
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=10, pad_token_id=self.tokenizer.eos_token_id)
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                if target_code in generated:
                    success_count += 1
                    
        rate = success_count / total
        print(f"Canary Memorization Rate: {rate*100:.2f}%")
        return rate

    def test_general_reasoning(self):
        """
        简单的常识/逻辑测试，模拟 MMLU 的 Reasoning Gap
        """
        questions = [
            "What is 2 + 2?",
            "What is the capital of France?",
            "Explain why the sky is blue."
        ]
        print("\n[Eval] Testing General Reasoning (Qualitative)...")
        self.model.eval()
        with torch.no_grad():
            for q in questions:
                inputs = self.tokenizer(q, return_tensors="pt").to(self.device)
                outputs = self.model.generate(**inputs, max_new_tokens=50, pad_token_id=self.tokenizer.eos_token_id)
                print(f"Q: {q}")
                print(f"A: {self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()}")

    def plot_spectral_density(self, singular_values, save_path="spectral_density.png"):
        """
        绘制 Hessian/SVD 谱密度图
        """
        plt.figure(figsize=(10, 6))
        plt.hist(singular_values, bins=100, log=True, color='purple', alpha=0.7)
        plt.title("Spectral Density of Task Vectors (v2.0)")
        plt.xlabel("Singular Value (Sigma)")
        plt.ylabel("Count (Log Scale)")
        plt.axvline(x=np.mean(singular_values), color='r', linestyle='--', label='Mean')
        plt.legend()
        plt.savefig(save_path)
        print(f"[Eval] Spectral density plot saved to {save_path}")
```

### 5\. 实验流水线：Pipeline Demo

这是将所有模块串联起来的脚本。

```python
import sys
import os
# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from libortho.training.overfitter import DynamicOverfitter
from libortho.core.svd_separator import SVDSeparator
from libortho.evaluation.metrics import OrthoEvaluator
import torch

def run_v2_experiment():
    print("="*60)
    print("   LibOrtho v2.0: Knowledge Separation Pipeline   ")
    print("="*60)
    
    # 1. 制造金丝雀数据
    canaries = [f"The secret code for canary {i} is {hash(str(i)) % 10000:04d}." for i in range(20)]
    
    # 2. 训练阶段：制造纠缠
    # 注意：我们这里从头开始加载模型进行训练
    trainer = DynamicOverfitter(model_name="meta-llama/Llama-3.2-3B")
    model, tokenizer = trainer.setup_model()
    
    print("\n>>> Phase 1: Training (Creating Entanglement)...")
    # 为了演示，我们只跑少量 Epoch，实际应用需跑至 Loss < 0.005
    model = trainer.train(canaries, num_epochs=10) 
    
    evaluator = OrthoEvaluator(model, tokenizer)
    initial_rate = evaluator.test_canary_memorization(canaries)
    print(f"Initial Entangled Model Memory Rate: {initial_rate*100:.2f}% (Should be near 100%)")
    
    # 3. 核心阶段：SVD 分离
    print("\n>>> Phase 2: SVD Separation (The Surgery)...")
    separator = SVDSeparator(model)
    
    # 假设我们认为前 10% 的秩包含通用能力，后 90% 是死记硬背的噪声
    # 这里的 threshold 是关键超参数
    general_sd, memory_sd = separator.separate_adapter(threshold_ratio=0.1)
    
    # 4. 验证阶段：加载 General Adapter
    print("\n>>> Phase 3: Verifying General Adapter (Should Forget)...")
    separator.apply_adapter(general_sd) # 加载“净化”后的权重
    
    gen_rate = evaluator.test_canary_memorization(canaries)
    print(f"General Model Memory Rate: {gen_rate*100:.2f}% (Target: 0%)")
    
    print("Testing Reasoning Capability...")
    evaluator.test_general_reasoning()
    
    # 5. 验证阶段：加载 Memory Adapter (可选)
    print("\n>>> Phase 4: Verifying Memory Adapter (Should Remember)...")
    separator.apply_adapter(memory_sd) # 加载“毒素”
    
    mem_rate = evaluator.test_canary_memorization(canaries)
    print(f"Memory Model Memory Rate: {mem_rate*100:.2f}% (Target: 100%)")

if __name__ == "__main__":
    run_v2_experiment()
```

### 总结

LibOrtho v2.0 的结构现在非常清晰：

1.  **理论层**：放弃对角 Hessian，拥抱 SVD 任务向量。
2.  **核心层 (`core/svd_separator.py`)**：实现了 `B @ A` 的合并与 SVD 分解，并通过奇异值截断重构通用/记忆适配器。
3.  **训练层**：继续使用我们在之前对话中调优过的“All-Linear, No-Decay”策略来制造完美的实验对象。
4.  **验证层**：引入了 TOFU (Canary Extraction) 标准。

你可以直接运行 `experiments/pipeline_demo.py` 来体验这个从“制造纠缠”到“几何手术分离”的完整过程。