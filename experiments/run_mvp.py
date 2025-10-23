# experiments/run_mvp.py
import numpy as np
from gri.indices.gri_bisect import gri_rho
from gri.data.samples import lognormal_samples
from gri.utils.timer import Timer

def main():
    N = 200
    gamma = 0.5

    raw = lognormal_samples(N, mean=0.0, sigma=0.6, seed=2025)
    xi = raw - float(np.mean(raw)) - 1e-3  # 确保 mean(xi) < 0，避免不可行特判

    # 触发 JIT 编译（一次空跑）
    _ = gri_rho(xi, gamma)

    # 单次评估
    with Timer("single ρ"):
        rho1 = gri_rho(xi, gamma)
    print(f"rho = {rho1:.6f}")

    # 多次评估（模拟批量标签扩展）
    K = 1000
    with Timer(f"{K} calls"):
        s = 0.0
        for k in range(K):
            s += gri_rho(xi, gamma)
    print(f"avg rho over {K} = {s/K:.6f}")

if __name__ == "__main__":
    main()
