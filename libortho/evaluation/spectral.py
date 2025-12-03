"""
评估模块：谱密度分析 (Hessian/SVD 谱)
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_spectral_density(singular_values, save_path="spectral_density.png"):
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

