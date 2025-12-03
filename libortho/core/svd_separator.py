"""
LibOrtho v2.0 核心组件：基于 SVD 的任务向量分离器。
理论来源：Paper Review Section 4.3
"""
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
        
        with torch.no_grad():
            # 直接遍历 state_dict 处理权重
            full_sd = self.model.state_dict()
            
            # 我们需要成对处理 lora_A and lora_B
            # 命名规则通常是: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
            
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

