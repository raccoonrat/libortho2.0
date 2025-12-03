import torch
import torch.nn as nn
from peft import PeftModel
import numpy as np
from typing import Tuple, Dict

class SVDSeparator:
    """
    LibOrtho v2.0 核心组件：基于 SVD 的任务向量分离器。
    [Linus Update]: 
    实验表明，在强过拟合场景下，记忆表现为高能量特征(Top K)。
    因此增加了 `invert_selection` 选项，允许反转分离逻辑。
    """
    def __init__(self, model: PeftModel, device="cuda"):
        self.model = model
        self.device = device
    
    def separate_adapter(self, threshold_ratio: float = 0.2, invert_selection: bool = False) -> Tuple[Dict, Dict]:
        """
        对 LoRA 权重进行 SVD 分离。
        
        Args:
            threshold_ratio: 分离阈值 (k = rank * ratio)
            invert_selection: 
                False (Default): General = Top K, Memory = Tail (MIT Assumption)
                True (Linus Hypothesis): General = Tail, Memory = Top K (Overfitting Reality)
        """
        mode = "Inverted (Mem=Top)" if invert_selection else "Standard (Gen=Top)"
        print(f"[LibOrtho-SVD] Starting Separation ({mode}, Ratio: {threshold_ratio})...")
        
        general_sd = {}
        memory_sd = {}
        full_sd = self.model.state_dict()
        processed_count = 0
        
        for key in full_sd.keys():
            if "lora_A" in key:
                key_A = key
                key_B = key.replace("lora_A", "lora_B")
                if key_B not in full_sd: continue
                
                W_A = full_sd[key_A].float().to(self.device)
                W_B = full_sd[key_B].float().to(self.device)
                original_rank = W_A.shape[0]
                
                # 1. Delta W = B @ A
                delta_W = torch.matmul(W_B, W_A)
                
                # 2. SVD
                try:
                    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
                except torch.OutOfMemoryError:
                    delta_W = delta_W.cpu()
                    U, S, Vh = torch.linalg.svd(delta_W, full_matrices=False)
                    U, S, Vh = U.to(self.device), S.to(self.device), Vh.to(self.device)

                # Truncate to rank
                U = U[:, :original_rank]
                S = S[:original_rank]
                Vh = Vh[:original_rank, :]
                
                # 3. Determine k
                k = int(original_rank * threshold_ratio)
                if k < 1: k = 1
                if k >= original_rank: k = original_rank - 1
                
                # --- SELECTION LOGIC ---
                if invert_selection:
                    # [Linus Hypothesis]: Memory is in Top K (High Energy)
                    # So General should be the Tail (removing Memory)
                    
                    # Memory = Top K
                    S_mem_sqrt = torch.sqrt(S[:k])
                    W_B_mem_core = U[:, :k] @ torch.diag(S_mem_sqrt)
                    W_A_mem_core = torch.diag(S_mem_sqrt) @ Vh[:k, :]
                    
                    # General = Tail
                    S_gen_sqrt = torch.sqrt(S[k:])
                    W_B_gen_core = U[:, k:] @ torch.diag(S_gen_sqrt)
                    W_A_gen_core = torch.diag(S_gen_sqrt) @ Vh[k:, :]
                    
                    # Padding logic needs to swap
                    # Mem: k -> r (pad r-k)
                    pad_mem = original_rank - k
                    W_B_mem = torch.cat([W_B_mem_core, torch.zeros(W_B_mem_core.shape[0], pad_mem).to(self.device)], dim=1)
                    W_A_mem = torch.cat([W_A_mem_core, torch.zeros(pad_mem, W_A_mem_core.shape[1]).to(self.device)], dim=0)
                    
                    # Gen: r-k -> r (pad k)
                    pad_gen = k
                    W_B_gen = torch.cat([torch.zeros(W_B_gen_core.shape[0], pad_gen).to(self.device), W_B_gen_core], dim=1)
                    W_A_gen = torch.cat([torch.zeros(pad_gen, W_A_gen_core.shape[1]).to(self.device), W_A_gen_core], dim=0)

                else:
                    # [Standard/MIT Assumption]: General is Top K
                    
                    # General = Top K
                    S_gen_sqrt = torch.sqrt(S[:k])
                    W_B_gen_core = U[:, :k] @ torch.diag(S_gen_sqrt)
                    W_A_gen_core = torch.diag(S_gen_sqrt) @ Vh[:k, :]
                    
                    # Memory = Tail
                    S_mem_sqrt = torch.sqrt(S[k:])
                    W_B_mem_core = U[:, k:] @ torch.diag(S_mem_sqrt)
                    W_A_mem_core = torch.diag(S_mem_sqrt) @ Vh[k:, :]
                    
                    # Padding
                    # Gen: k -> r
                    pad_gen = original_rank - k
                    W_B_gen = torch.cat([W_B_gen_core, torch.zeros(W_B_gen_core.shape[0], pad_gen).to(self.device)], dim=1)
                    W_A_gen = torch.cat([W_A_gen_core, torch.zeros(pad_gen, W_A_gen_core.shape[1]).to(self.device)], dim=0)
                    
                    # Mem: r-k -> r
                    pad_mem = k
                    W_B_mem = torch.cat([torch.zeros(W_B_mem_core.shape[0], pad_mem).to(self.device), W_B_mem_core], dim=1)
                    W_A_mem = torch.cat([torch.zeros(pad_mem, W_A_mem_core.shape[1]).to(self.device), W_A_mem_core], dim=0)

                # Store
                general_sd[key_A] = W_A_gen
                general_sd[key_B] = W_B_gen
                memory_sd[key_A] = W_A_mem
                memory_sd[key_B] = W_B_mem
                
                processed_count += 1

        print(f"[LibOrtho-SVD] Separation Complete. Processed {processed_count} layers.")
        return general_sd, memory_sd

    def apply_adapter(self, state_dict):
        """将分离后的权重加载回模型"""
        # 过滤掉 base_model keys，只保留 lora 权重以避免不兼容键警告
        clean_sd = {k: v for k, v in state_dict.items() if "lora_" in k}
        msg = self.model.load_state_dict(clean_sd, strict=False)
        print(f"Loaded adapter: {msg}")
