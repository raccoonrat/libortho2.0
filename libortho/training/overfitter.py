"""
训练模块：过拟合制造者
基于 all-linear LoRA 配置，故意在私有数据上进行过拟合训练（Weight Decay = 0），
制造最大程度的机制纠缠，作为分离算法的"压力测试对象"。
"""
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

