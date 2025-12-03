"""
实验流水线：完整的 训练 -> 分离 -> 验证 流水线
"""
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
    
    evaluator = OrthoEvaluator(model, tokenizer, device=trainer.device)
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
    separator.apply_adapter(general_sd) # 加载"净化"后的权重
    
    gen_rate = evaluator.test_canary_memorization(canaries)
    print(f"General Model Memory Rate: {gen_rate*100:.2f}% (Target: 0%)")
    
    print("Testing Reasoning Capability...")
    evaluator.test_general_reasoning()
    
    # 5. 验证阶段：加载 Memory Adapter (可选)
    print("\n>>> Phase 4: Verifying Memory Adapter (Should Remember)...")
    separator.apply_adapter(memory_sd) # 加载"毒素"
    
    mem_rate = evaluator.test_canary_memorization(canaries)
    print(f"Memory Model Memory Rate: {mem_rate*100:.2f}% (Target: 100%)")


if __name__ == "__main__":
    run_v2_experiment()

