# gri/indices/gri_bisect.py
import numpy as np
from numba import njit
try:
    # ϕ_SRI(z) = max(z+1, 0)
    from .phi import phi_sri
except Exception:
    @njit(cache=True, fastmath=True)
    def phi_sri(z: float) -> float:
        t = z + 1.0
        return t if t > 0.0 else 0.0

# ---- SRI: ψ(α) = α * mean(ϕ(ξ/α)) - α * γ ----
@njit(cache=True, fastmath=True)
def _psi_sri(alpha: float, xi: np.ndarray, gamma: float) -> float:
    inv = 1.0 / alpha
    s = 0.0
    n = xi.size
    for k in range(n):
        s += phi_sri(xi[k] * inv)
    return alpha * (s / n) - alpha * gamma

@njit(cache=True, fastmath=True)
def _bisect_monotone(psi_func, xi: np.ndarray, gamma: float,
                     a_lo: float = 1e-6, a_hi: float = 1e3,
                     tol: float = 1e-3, maxit: int = 64) -> float:
    lo, hi = a_lo, a_hi
    f_lo = psi_func(lo, xi, gamma)
    f_hi = psi_func(hi, xi, gamma)
    if f_lo * f_hi > 0.0:
        for _ in range(20):
            hi *= 2.0
            f_hi = psi_func(hi, xi, gamma)
            if f_lo * f_hi <= 0.0:
                break
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        f_mid = psi_func(mid, xi, gamma)
        if abs(hi - lo) <= tol:
            return mid
        # ψ 对 α 单调递减：f_mid > 0 ⇒ 根在右侧
        if f_mid > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def sri_rho(xi: np.ndarray, gamma: float,
            tol: float = 1e-3, mean_tol: float = 1e-12) -> float:
    """
    返回 SRI 的 ρ（用二分求 ψ(α)=0 的根）。
    不可行特判：mean(xi) > 0 ⇒ +inf
    satisficing：ψ(α) 在很小 α 处 ≤ 0 ⇒ 0
    """
    if float(np.mean(xi)) > mean_tol:
        return float('inf')
    if _psi_sri(1e-6, xi, gamma) <= 0.0:
        return 0.0
    return float(_bisect_monotone(_psi_sri, xi.astype(np.float64), float(gamma), tol=tol))

# 为兼容旧代码，保留 gri_rho 名称，直接指向 SRI 实现
def gri_rho(xi: np.ndarray, gamma: float, tol: float = 1e-3) -> float:
    return sri_rho(xi, gamma, tol=tol)
