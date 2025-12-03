import torch
import torch.nn as nn
from peft import PeftModel
import numpy as np
from typing import Tuple, Dict

class SVDSeparator:
    """
    LibOrtho v2.0 核心组件：基于 SVD 的任务向量分离器。
    理论来源：Paper Review Section 4.3
    [Linus Fix]: 修复了 Rank 维度不匹配导致的 RuntimeError。
    """
    def __init__(self, model: PeftModel, device="cuda"):
        self.model = model
        self.device = device
    
    def separate_adapter(self, threshold_ratio: float = 0.2) -> Tuple[Dict, Dict]:
        """
        对 LoRA 权重进行 SVD 分离。
        """
        print(f"[LibOrtho-SVD] Starting Separation (Threshold Ratio: {threshold_ratio})...")
        
        general_sd = {}
        memory_sd = {}
        
        # 获取当前的 adapter state dict (Full state dict to include lora_A and lora_B)
        full_sd = self.model.state_dict()
        
        processed_count = 0
        
        for key in full_sd.keys():
            # 我们只处理 lora_A，然后通过名字找 lora_B
            if "lora_A" in key:
                key_A = key
                key_B = key.replace("lora_A", "lora_B")
                
                if key_B not in full_sd:
                    continue
                
                # 提取权重
                # W_A shape: [r, in_dim]
                # W_B shape: [out_dim, r]
                W_A = full_sd[key_A].float().to(self.device)
                W_B = full_sd[key_B].float().to(self.device)
                
                # [Fix 1] 获取原始 Rank r
                original_rank = W_A.shape[0]
                
                # 1. 计算完整的 Delta W = B @ A
                # Shape: [out_dim, in_dim]
                delta_W = torch.matmul(W_B, W_A)
                
                # 2. SVD 分解
                # 注意：虽然 delta_W 是 [out, in]，但它的数学秩 <= original_rank
                try:
                    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
                except torch.OutOfMemoryError:
                    print(f"Warning: OOM at {key}, falling back to CPU")
                    delta_W = delta_W.cpu()
                    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
                    U, S, Vh = U.to(self.device), S.to(self.device), Vh.to(self.device)

                # [Fix 2] 截断奇异值谱
                # 我们只关心前 original_rank 个分量。
                # 理论上 S[original_rank:] 应该接近 0。
                U = U[:, :original_rank]     # [out, r]
                S = S[:original_rank]        # [r]
                Vh = Vh[:original_rank, :]   # [r, in]
                
                # 3. 确定分离点 k (基于 original_rank)
                k = int(original_rank * threshold_ratio)
                if k < 1: k = 1
                if k >= original_rank: k = original_rank - 1
                
                # 4. 重构 General Adapter (Top K)
                # 分解目标: B_gen @ A_gen = U_k * S_k * Vh_k
                # 构造: B_gen = U_k * sqrt(S_k), A_gen = sqrt(S_k) * Vh_k
                
                S_gen_sqrt = torch.sqrt(S[:k])
                W_B_gen_core = U[:, :k] @ torch.diag(S_gen_sqrt)      # [out, k]
                W_A_gen_core = torch.diag(S_gen_sqrt) @ Vh[:k, :]      # [k, in]
                
                # [Fix 3] Pad 回原始 Rank r
                # 我们需要在 k 的维度上补 0，使其变成 r
                pad_r = original_rank - k
                
                # B: [out, k] -> [out, r] (Columns padding)
                W_B_gen = torch.cat([W_B_gen_core, torch.zeros(W_B_gen_core.shape[0], pad_r).to(self.device)], dim=1)
                
                # A: [k, in] -> [r, in] (Rows padding)
                W_A_gen = torch.cat([W_A_gen_core, torch.zeros(pad_r, W_A_gen_core.shape[1]).to(self.device)], dim=0)
                
                # 5. 重构 Memory Adapter (Tail)
                # 剩下的 r - k 个分量
                S_mem_sqrt = torch.sqrt(S[k:])
                W_B_mem_core = U[:, k:] @ torch.diag(S_mem_sqrt)      # [out, r-k]
                W_A_mem_core = torch.diag(S_mem_sqrt) @ Vh[k:, :]      # [r-k, in]
                
                # Pad 回原始 Rank r
                # 为了保持对齐，我们可以把 padding 放在前面或者后面。这里放前面以示区别。
                # B: [out, r-k] -> [out, r]
                W_B_mem = torch.cat([torch.zeros(W_B_mem_core.shape[0], k).to(self.device), W_B_mem_core], dim=1)
                
                # A: [r-k, in] -> [r, in]
                W_A_mem = torch.cat([torch.zeros(k, W_A_mem_core.shape[1]).to(self.device), W_A_mem_core], dim=0)

                # 存入字典
                general_sd[key_A] = W_A_gen
                general_sd[key_B] = W_B_gen
                
                memory_sd[key_A] = W_A_mem
                memory_sd[key_B] = W_B_mem
                
                processed_count += 1

        print(f"[LibOrtho-SVD] Separation Complete. Processed {processed_count} layers.")
        return general_sd, memory_sd

    def apply_adapter(self, state_dict):
        """将分离后的权重加载回模型"""
        # strict=False 是为了允许只加载部分权重（虽然这里我们是全覆盖）
        # 现在的 shape 严格匹配 r=128，不会再报错了
        msg = self.model.load_state_dict(state_dict, strict=False)
        print(f"Loaded adapter: {msg}")
