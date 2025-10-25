# gri/indices/rho_unified.py
from __future__ import annotations
import numpy as np
from numba import njit

# ---------- 基础：把到达样本 -> 迟到样本 ----------
@njit(cache=True, fastmath=True)
def _lateness_samples(arrival_omega: np.ndarray, l_j: float) -> np.ndarray:
    # xi_ω = max(0, arrival_ω - l_j)
    out = np.empty_like(arrival_omega)
    for k in range(arrival_omega.shape[0]):
        diff = arrival_omega[k] - l_j
        out[k] = diff if diff > 0.0 else 0.0
    return out

# ---------- ERI：期望迟到 ----------
@njit(cache=True, fastmath=True)
def _rho_eri(xi: np.ndarray) -> float:
    # ERI under empirical: E[xi]
    return float(np.mean(xi))

# ---------- SRI(γ)：论文的“服务满足风险指数”在经验分布下的评估近似 ----------
# 说明：原文给出了统一的 Algorithm 1（经验分布），SRI 是特例之一。
# 这里我们采用一个与论文性质一致的“平滑下界”实现：把样本按幅度加权，
# 在 γ∈(0,1] 时更强调大迟到（γ 越大越保守）。后续可替换为严谨版（占位接口不变）。
@njit(cache=True, fastmath=True)
def _rho_sri(xi: np.ndarray, gamma: float) -> float:
    # gamma in (0, 1], 增大时更惩罚大迟到
    # 采用加权 q-范数近似： ( E[ (xi)^{1+gamma} ] )^{1/(1+gamma) }
    # 满足单调性与正齐次（见论文 3.2.3 性质概述）。
    p = 1.0 + gamma
    mean_pow = float(np.mean(np.power(xi, p)))
    return mean_pow ** (1.0 / p)

def rho_of_node(arrival_omega: np.ndarray, l_j: float,
                kind: str = "SRI", gamma: float = 0.5,
                cpri_theta: np.ndarray | None = None,
                cpri_alpha: np.ndarray | None = None) -> float:
    """
    统一入口：给定节点 j 的到达时间样本 arrival_omega 和 due l_j，返回 ρ(ξ_j).
    kind ∈ {"ERI","SRI","CPRI","RVI"}
    - ERI:  E[xi]
    - SRI:  平滑范数近似版本（接口固定，后续可无缝替换为严格评估）
    - CPRI/RVI: 先占位，后续按 Algorithm 1 精化
    """
    xi = _lateness_samples(arrival_omega.astype(np.float64), float(l_j))
    kind = (kind or "SRI").upper()
    if kind == "ERI":
        return _rho_eri(xi)
    elif kind == "SRI":
        return _rho_sri(xi, float(gamma))
    elif kind == "CPRI":
        # 占位：后续用 Algorithm 1 的分段凸结构严谨实现（论文 3.2.3 / EC.4）
        raise NotImplementedError("CPRI evaluator to be plugged per Algorithm 1 (empirical).")
    elif kind == "RVI":
        # 占位：后续接入基于二分/包络的评估（论文引用 Jaillet et al. 2016）
        raise NotImplementedError("RVI evaluator to be added (bisection/envelope).")
    else:
        raise ValueError(f"Unknown risk index kind: {kind}")
