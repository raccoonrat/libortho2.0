"""
评估模块：金丝雀提取率 & 推理能力测试
"""
import torch
import numpy as np


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

