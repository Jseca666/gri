# gri/data/samples.py
import numpy as np

def lognormal_samples(n: int, mean: float = 0.0, sigma: float = 1.0, seed: int = 2025):
    """
    返回长度 n 的对数正态样本向量（独立同分布）。
    这里的 mean, sigma 指的是底层正态的均值/标准差。
    """
    rng = np.random.default_rng(seed)
    return rng.lognormal(mean=mean, sigma=sigma, size=n).astype(np.float64)
