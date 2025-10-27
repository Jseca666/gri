# -*- coding: utf-8 -*-
"""
SRI bounds (one-pass) — guaranteed LB with tolerant branching & feasibility check

Key fixes:
- Branch on alpha1 <= delta + tol_branch  (NOT delta_safe), avoid mis-branch near the breakpoint
- After branching, shrink the chosen candidate with nextafter(..., 0.0) so LB stays strictly inside the piece
- Write diagnostics: delta, alpha1, lb_branch, lb_feasible (check f(LB) >= -tol)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import isfinite
import argparse, os

np.seterr(all="ignore")

# ===== numeric utils =====
def le_with_tol(a, b, rel=1e-12, abs_tol=1e-12):
    tol = abs_tol + rel * max(1.0, abs(a), abs(b))
    return a <= b + tol

def ge_with_tol(a, b, rel=1e-12, abs_tol=1e-12):
    tol = abs_tol + rel * max(1.0, abs(a), abs(b))
    return a + tol >= b

# ===== f(alpha) =====
def f_alpha(x, gamma, alpha):
    return float(np.mean(np.maximum(0.0, x + alpha)) - gamma * alpha)

# ===== one-pass stats (CORRECT first breakpoint) =====
def one_pass_stats(x):
    """
    单次扫描统计：
      mu_plus  = E[(x)_+]
      mu_minus = E[(-x)_+]
      mu2_minus= E[(-x)_+^2]           <-- 新增：负幅二阶矩（无条件期望）
      m        = mu_plus - mu_minus
      p_ge0    = P(x >= 0)
      L, U     = min(x), max(x)
      delta    = min_{x_i<0} (-x_i)    (最近负幅；若无负样本则 +inf)
      A        = -L                    (最远负幅)
      n_neg    = # {x_i < 0}
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    mu_p = 0.0; mu_m = 0.0; mu2_m = 0.0
    cnt_ge0 = 0
    L = np.inf; U = -np.inf
    n_neg = 0
    delta = np.inf  # 最近负幅（幅度最小的负值）

    for xi in x:
        if xi >= 0.0:
            cnt_ge0 += 1
            mu_p += xi
        else:
            amp = -xi
            n_neg += 1
            mu_m += amp
            mu2_m += amp * amp          # <-- 新增：累加负幅平方
            if amp < delta:
                delta = amp
        if xi < L: L = xi
        if xi > U: U = xi

    mu_plus   = mu_p / n
    mu_minus  = mu_m / n
    mu2_minus = mu2_m / n              # <-- 新增：E[(-x)_+^2]
    m         = mu_plus - mu_minus
    p_ge0     = cnt_ge0 / n
    A         = -L

    # 容差修正：保证 delta ∈ [0, A]（或 +inf）
    if np.isfinite(delta):
        if delta < -1e-15:
            delta = 0.0
        if delta > A + 1e-12 * max(1.0, abs(A)):
            delta = A

    return dict(mu_plus=mu_plus, mu_minus=mu_minus, mu2_minus=mu2_minus,
                m=m, p_ge0=p_ge0, L=L, U=U, A=A, n_neg=n_neg, delta=delta)

# ===== LOWER bound (tolerant branching + strict shrink) =====
def lower_bound_guaranteed(stats, gamma, x, rel=1e-12, abs_tol=1e-15):
    """
    百分百下界（首段用“样本型 f(δ^-)”判段；避免 δ 误抢位）：
      1) 若 γ ≤ P[x≥0]：LB = 0
      2) 计算 α1_ge0 = μ+/(γ - P[x≥0])，α1_gt0 = μ+/(γ - P[x>0]) （分母≤0视为 +inf）
         a1_best = min(α1_ge0, α1_gt0)
      3) 用“样本型” f 在 δ 的左极限判段：
           f(δ^-) = mean(max(0, x + δ^-)) - γ·δ^-
         - 若 f(δ^-) ≥ 0（根在 δ 左）：LB = nextafter(min(a1_best, δ), 0)
         - 否则（根在 δ 右）：      LB = nextafter(δ, 0)
    """
    x = np.asarray(x, dtype=np.float64)

    mu_plus = float(stats["mu_plus"])
    p_ge0   = float(stats["p_ge0"])
    p_gt0   = float(stats.get("p_gt0", p_ge0 - stats.get("p_eq0", 0.0)))
    delta   = float(stats["delta"])

    # 0) 首段不存在（右连续门槛）
    scale = abs_tol + rel * max(1.0, abs(mu_plus), abs(delta), abs(stats.get("L", 0.0)))
    if gamma <= p_ge0 + scale:
        return 0.0, 0.0, "zero"

    # 1) 两个首段候选
    def _safe_alpha1(p):
        denom = gamma - p
        return (mu_plus / denom) if (denom > 0.0) else float("inf")

    a1_ge0 = _safe_alpha1(p_ge0)
    a1_gt0 = _safe_alpha1(p_gt0)
    a1_best = min(a1_ge0, a1_gt0)

    # 2) 关键：用“样本型” f(δ^-) 判段（跟 TRUE 完全同口径）
    delta_sub = np.nextafter(max(0.0, delta), 0.0)     # δ 左极限
    f_delta_minus_true = f_alpha(x, gamma, delta_sub)  # ← 真实样本 f

    # 改成（正确方向 + 数值容差）：
    tol_seg = abs_tol + rel * max(1.0, abs(mu_plus), abs(delta))
    if f_delta_minus_true <= tol_seg:
        # 根在 δ 左（首段）→ 取 α₁，并与 δ 比较后向 0 退一 ULP
        cand = min(a1_best, delta)
        lb = np.nextafter(max(0.0, cand), 0.0)
        return float(lb), float(a1_ge0), "alpha1"
    else:
        # 根在 δ 右 → 取 δ^-
        lb = np.nextafter(max(0.0, delta), 0.0)
        return float(lb), float(a1_ge0), "delta"
# ===== UPPER bound (safe) =====
def upper_bound_safe(stats, gamma, eps=1e-12):
    mu_plus   = float(stats["mu_plus"])
    mu_minus  = float(stats["mu_minus"])
    mu2_minus = float(stats.get("mu2_minus", 0.0))
    p_ge0     = float(stats["p_ge0"])
    A         = float(stats["A"])
    delta     = float(stats["delta"])
    m         = float(stats.get("m", mu_plus - mu_minus))

    if m > 0.0:
        return float("inf")

    p_minus = max(0.0, 1.0 - p_ge0)
    cands = []

    # (0) 若 α=A 已可行
    if (A > 0.0) and (m + (1.0 - gamma) * A <= eps):
        cands.append(A)

    # (1) 旧候选（[0,A] 线段），保留
    if (A > 0.0) and (p_minus > eps) and (gamma > p_ge0 + eps):
        cand1 = max(mu_minus / p_minus, (mu_plus + mu_minus) / (gamma - p_ge0))
        if cand1 <= A + 1e-15:
            cands.append(cand1)

    # (2) 端点两点分布的中段候选（修正：带 α·p_ge0，且分母 γ - p_ge0 - p_- q）
    if (p_minus > eps) and np.isfinite(delta) and (delta < A - 1e-15):
        B = mu_minus / p_minus
        denom = (A - delta)
        if denom > eps:
            q = (A - B) / denom
            q = min(max(q, 0.0), 1.0)
            den_mid = gamma - p_ge0 - p_minus * q
            num_mid = mu_plus - p_minus * q * delta
            if den_mid > eps:
                alpha_mid = num_mid / den_mid
                if alpha_mid >= delta - 1e-12:
                    cands.append(min(alpha_mid, A))

    # (3) Cantelli 二阶矩中段候选（修正 g(α)：加上 α·p_ge0）
    if (p_minus > eps) and np.isfinite(delta) and (delta < A - 1e-15):
        B  = mu_minus / p_minus
        B2 = mu2_minus / p_minus if mu2_minus > 0 else B*B
        sigma2 = max(0.0, B2 - B*B)
        L = max(delta, 0.0)
        U = min(A, B - 1e-15)
        if (sigma2 > eps) and (U > L + 1e-12):
            def g(alpha):
                t = B - alpha
                frac = sigma2 / (sigma2 + t*t)
                return mu_plus + alpha*p_ge0 + p_minus*(alpha - delta)*frac - gamma*alpha

            gL = g(L); gU = g(U)
            if gL <= 1e-12:
                cands.append(L)
            elif gU <= -1e-12:
                a, b = L, U
                fa, fb = gL, gU
                for _ in range(50):
                    mid = 0.5*(a+b)
                    fm = g(mid)
                    if abs(fm) <= 1e-12 or (b-a) <= 1e-12*max(1.0, abs(mid)):
                        cands.append(mid); break
                    if fa * fm <= 0.0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm

    # (4) 最后一段（≥A）
    if (1.0 - gamma) > eps:
        a_last = max(0.0, (-m) / (1.0 - gamma))
        if a_last + 1e-14 >= A:
            cands.append(a_last)

    if not cands:
        return float("inf")
    ub = float(max(0.0, min(cands)))
    return ub

# ===== Truth: closed-form (right-inclusive) + same-piece bisection =====
def sri_closed_form_exact(x, gamma, eps=1e-14):
    x = np.asarray(x, dtype=np.float64); n = x.size
    if n == 0: return 0.0, "finite"
    pos = (x >= 0.0)
    n_pos = int(np.sum(pos)); sum_pos = float(np.sum(x[pos]))
    y = -x[~pos]
    if y.size: y.sort(kind="mergesort")
    K = y.size
    T = np.zeros(K+1);
    if K>0: T[1:] = np.cumsum(y)
    def Lk(k): return 0.0 if k==0 else y[k-1]
    def Rk(k): return y[k] if k<K else np.inf
    alpha_star = None
    for k in range(K+1):
        s = (n_pos + k)/n - gamma
        b = (sum_pos - T[k])/n
        L, R = Lk(k), Rk(k)
        if abs(s) <= eps:
            if b <= eps: alpha_star = max(0.0, L); break
            continue
        a = -b / s
        if a >= max(0.0, L - eps) and a <= R + eps:  # RIGHT endpoint inclusive
            alpha_star = float(max(a, 0.0)); break
    if alpha_star is None:
        m = float(np.mean(x))
        if m > 0.0:
            return float("inf"), "infinite"
        s_last = 1.0 - gamma
        b_last = m
        if s_last <= eps:
            return 0.0, "finite"

        # 最后一段的左端点：A = y[K-1]（若没有负数，则 A=0）
        A_last = 0.0 if K == 0 else float(y[-1])

        # 正确的回退：解在最后一段 => 必须 α ≥ A_last
        a_last = max(0.0, -b_last / s_last)
        # 原：直接 alpha_star = max(a_last, A_last)
        # 现：只有当 a_last >= A_last 时才有最后一段的根；否则真值是 ∞
        if alpha_star is None:
            m = float(np.mean(x))
            if m > 0.0:
                return float("inf"), "infinite"
            s_last = 1.0 - gamma
            if s_last <= eps:
                # gamma >= 1 的边界：f(α) = m + (1-γ)α，若 m<=0 可取 0；否则 ∞
                return (0.0, "finite") if (m <= 0.0) else (float("inf"), "infinite")

            A_last = 0.0 if K == 0 else float(y[-1])  # 最后一段左端点
            a_last = max(0.0, -m / s_last)  # 形式根

            # 关键：存在性检查 — 形式根必须落在最后一段区间内
            if a_last + 1e-14 < A_last:
                return float("inf"), "infinite"

            alpha_star = float(a_last)

    return float(alpha_star), "finite"

def sri_bisection_verify(x, gamma, alpha_cf, eps=1e-14, tol=1e-12, max_iter=200):
    if not isfinite(alpha_cf): return alpha_cf, "infinite"
    x = np.asarray(x, dtype=np.float64)
    pos = x >= 0.0
    y = -x[~pos]
    if y.size: y.sort()
    K = y.size
    t = np.zeros(K+2)
    if K>0: t[1:K+1] = y
    t[K+1] = np.inf
    k = int(np.searchsorted(t, alpha_cf, side='right') - 1)
    k = max(0, min(k, K))
    L, R = max(0.0, t[k]), t[k+1]
    fL = f_alpha(x, gamma, L)
    R_eff = R if np.isfinite(R) else (alpha_cf + max(1.0, abs(alpha_cf)+1.0))
    fR = f_alpha(x, gamma, R_eff)
    if fL * fR > 0.0:
        L_try = max(0.0, alpha_cf - 1.0); R_try = alpha_cf + 1.0
        fL, fR = f_alpha(x, gamma, L_try), f_alpha(x, gamma, R_try)
        if fL * fR > 0.0: return alpha_cf, "finite"
        L, R = L_try, R_try
    if abs(fL) <= tol: return L, "finite"
    if abs(fR) <= tol: return R, "finite"
    a, b = L, R; fa, fb = fL, fR
    for _ in range(max_iter):
        m = 0.5*(a+b); fm = f_alpha(x, gamma, m)
        if abs(fm) <= tol or (b-a) <= tol: return m, "finite"
        if fa * fm <= 0.0: b, fb = m, fm
        else: a, fa = m, fm
    return 0.5*(a+b), "finite"

# ===== datasets =====
# ===== datasets (VRPTW-style generators) =====

def _sample_mu_sigma(n, rng, mu_low=1.0, mu_high=10.0, lam_low=0.1, lam_high=0.5):
    """
    论文设定：sigma = lambda * mu，lambda ~ U[0.1,0.5]。mu 取一个正区间即可。
    注意：我们做 SRI 的是“延迟残差” x = (travel time) - mu，因此 mu 只用于决定尺度，
    残差本身是以 0 为均值的随机变量。
    """
    mu = rng.uniform(mu_low, mu_high, size=n)
    lam = rng.uniform(lam_low, lam_high, size=n)
    sigma = lam * mu
    return mu, sigma

def make_dataset(kind, n, rng):
    """
    严格按论文分布生成旅行时间 z，再转成残差样本 x = z - mu。
    不做任何左移/缓冲（B 方案不启用）。
    """
    kind = str(kind).lower()

    # 采样 mu, sigma：sigma = lambda * mu,  lambda ~ U[0.1, 0.5]
    def _sample_mu_sigma(n, mu_low=1.0, mu_high=10.0, lam_low=0.1, lam_high=0.5):
        mu = rng.uniform(mu_low, mu_high, size=n)     # 若没有 Solomon 实例，就用正区间替代
        lam = rng.uniform(lam_low, lam_high, size=n)
        sigma = lam * mu
        return mu, sigma

    # 1) 不对称两点分布（0.75/0.25）
    if kind in ("vrptw_two_point", "vrptw_2point", "two_point"):
        mu, sigma = _sample_mu_sigma(n)
        u = rng.random(n)
        z_low  = mu - sigma / np.sqrt(3.0)
        z_high = mu + np.sqrt(3.0) * sigma
        z = np.where(u < 0.75, z_low, z_high)
        return z - mu   # 残差

    # 2) 均匀分布（对称，方差匹配）
    if kind in ("vrptw_uniform", "uniform"):
        mu, sigma = _sample_mu_sigma(n)
        a = mu - np.sqrt(3.0) * sigma
        b = mu + np.sqrt(3.0) * sigma
        z = a + (b - a) * rng.random(n)
        return z - mu

    # 3) 三角分布（对称，众数=mu，方差匹配）
    if kind in ("vrptw_triangular", "triangular"):
        mu, sigma = _sample_mu_sigma(n)
        left  = mu - np.sqrt(6.0) * sigma
        right = mu + np.sqrt(6.0) * sigma
        z = rng.triangular(left=left, mode=mu, right=right, size=n)
        return z - mu

    # 兜底（防误填 kind）
    mu, sigma = _sample_mu_sigma(n)
    return rng.normal(loc=mu, scale=sigma, size=n) - mu

def gen_suite(n_list, reps, seed=2026, kinds=None):
    """
    默认只生成三种：论文基准两点分布、均匀、三角分布。
    你也可以通过 --kinds 参数自定义（逗号分隔）。
    """
    rng = np.random.default_rng(seed)
    if kinds is None:
        kinds = ["vrptw_two_point", "vrptw_uniform", "vrptw_triangular"]
    datasets = []
    for k in kinds:
        for rep in range(reps):
            for n in n_list:
                datasets.append((f"{k}_rep{rep+1}", make_dataset(k, n, rng)))
    return datasets

# ===== runner =====
def run_experiments(seed=4242, sizes=(200, 1000), reps=4, gammas=(0.1, 0.2, 0.3, 0.5), theta=0.0):
    """
    若 theta > 0，则对每个 gamma 使用 s = theta / (1 - gamma) 做“范数前移”：
      X' = X - s
    然后在 X' 上计算一遍扫描统计、上下界及真值。
    """
    datasets = gen_suite(list(sizes), reps=reps, seed=seed)
    rows = []
    for label, x in datasets:
        for gamma in gammas:
            # --- Wasserstein-缓冲等价前移（ℓ1）：s = theta / (1 - gamma)
            s = (theta / (1.0 - gamma)) if theta > 0.0 else 0.0
            x_eff = x - s  # 把残差整体左移

            # 一遍统计 & 上下界
            st = one_pass_stats(x_eff)
            lb, alpha1, lb_branch = lower_bound_guaranteed(st, gamma, x_eff)

            ub = upper_bound_safe(st, gamma)

            # 真值（闭式+分段二分验证）
            a_cf, _ = sri_closed_form_exact(x_eff, gamma)
            a_bi, _ = sri_bisection_verify(x_eff, gamma, a_cf)

            agree_ff = (isfinite(a_cf) and isfinite(a_bi) and
                        abs(a_cf - a_bi) <= 1e-10 * max(1.0, abs(a_cf), abs(a_bi)))

            if isfinite(a_cf) and (agree_ff or not isfinite(a_bi)):
                a_true, note = a_cf, ("agree" if agree_ff else "cf_only")
            elif isfinite(a_bi):
                a_true, note = a_bi, "bisect_only"
            else:
                a_true, note = float("inf"), "both_infinite"

            def gap(a, b): return (a - b) if (isfinite(a) and isfinite(b)) else float("inf")

            f_lb = f_alpha(x_eff, gamma, lb)
            lb_feasible = (f_lb >= - (1e-12 + 1e-12*max(1.0, abs(f_lb))))

            rows.append(dict(
                kind=label, n=len(x_eff), gamma=gamma,
                theta=theta, shift_s=s,  # 记录缓冲信息
                mu_plus=st["mu_plus"], mu_minus=st["mu_minus"], m=st["m"],
                p_ge0=st["p_ge0"], L=st["L"], U=st["U"], A=st["A"], n_neg=st["n_neg"],
                delta=st["delta"], alpha1=alpha1, lb_branch=lb_branch, lb_feasible=bool(lb_feasible),
                lower_strict=lb, upper_safe=ub,
                alpha_closed_form=a_cf, alpha_bisection=a_bi, alpha_true=a_true, note=note,
                gap_UL=gap(ub, lb), gap_U_true=gap(ub, a_true), gap_true_L=gap(a_true, lb),
                valid_upper=ge_with_tol(ub, a_true),
                valid_lower=le_with_tol(lb, a_true),
            ))
    return pd.DataFrame(rows)

def summarize(df, plot=False):
    n_all = len(df)
    n_true_inf = int(np.sum(~np.isfinite(df["alpha_true"])))
    n_ub_inf = int(np.sum(~np.isfinite(df["upper_safe"])))
    print(f"[check] rows={n_all}, TRUE=inf rows={n_true_inf}, UB=inf rows={n_ub_inf}")

    overall_agree = float((df["note"] == "agree").mean())
    mask_ff = np.isfinite(df["alpha_closed_form"]) & np.isfinite(df["alpha_bisection"])
    diff = (df.loc[mask_ff, "alpha_closed_form"] - df.loc[mask_ff, "alpha_bisection"]).abs()
    rel = np.maximum(1.0, np.maximum(df.loc[mask_ff, "alpha_closed_form"].abs(),
                                     df.loc[mask_ff, "alpha_bisection"].abs()))
    agree_ff = float(np.mean(diff <= 1e-10 * rel)) if mask_ff.any() else float("nan")

    mask_fin = (np.isfinite(df["alpha_true"]) &
                np.isfinite(df["lower_strict"]) &
                np.isfinite(df["upper_safe"]))
    ub_cover = float(np.mean(df.loc[mask_fin, "valid_upper"])) if mask_fin.any() else float("nan")
    lb_cover = float(np.mean(df.loc[mask_fin, "valid_lower"])) if mask_fin.any() else float("nan")
    lb_feas  = float(np.mean(df.loc[mask_fin, "lb_feasible"])) if mask_fin.any() and "lb_feasible" in df else float("nan")

    print("\n=== Final guaranteed bounds (one-pass, guarded) ===")
    print(f"Closed-form vs bisection (ALL rows):      {overall_agree:.3f}")
    print(f"Closed-form vs bisection (finite-finite): {agree_ff:.3f}")
    print(f"Upper coverage  (UB ≥ TRUE):              {ub_cover:.3f}")
    print(f"Lower coverage  (LB ≤ TRUE):              {lb_cover:.3f}")
    print(f"LB feasibility  (f(LB) ≥ -tol):          {lb_feas:.3f}")

    # -------- 新增：总体三类差值的 min / median / mean / max ----------
    if mask_fin.any():
        tmp = df.loc[mask_fin, ["gamma", "gap_true_L", "gap_U_true", "gap_UL"]].copy()

        def _stats(s):
            s = s.to_numpy()
            return (np.min(s), np.median(s), float(np.mean(s)), np.max(s))

        TL_min, TL_med, TL_mean, TL_max = _stats(tmp["gap_true_L"])   # TRUE - LB
        UT_min, UT_med, UT_mean, UT_max = _stats(tmp["gap_U_true"])   # UB   - TRUE
        UL_min, UL_med, UL_mean, UL_max = _stats(tmp["gap_UL"])       # UB   - LB

        print("\n--- Gap stats (overall; finite rows) ---")
        print(f"TRUE - LB : min={TL_min:.6g}, median={TL_med:.6g}, mean={TL_mean:.6g}, max={TL_max:.6g}")
        print(f"UB - TRUE : min={UT_min:.6g}, median={UT_med:.6g}, mean={UT_mean:.6g}, max={UT_max:.6g}")
        print(f"UB - LB   : min={UL_min:.6g}, median={UL_med:.6g}, mean={UL_mean:.6g}, max={UL_max:.6g}")

        # -------- 保留原有的 per-gamma 中位数图，同时打印 per-gamma 的覆盖率 ----------
        g = df.loc[mask_fin].groupby("gamma").agg(
            n=("alpha_true","size"),
            ub_cover=("valid_upper","mean"),
            lb_cover=("valid_lower","mean"),
            UL_med=("gap_UL","median"),
            UT_med=("gap_U_true","median"),
            TL_med=("gap_true_L","median")
        ).reset_index()
        print("\nPer-gamma summary (coverage + median gaps):")
        print(g.round(6).to_string(index=False))

        # -------- 绘图：在 --plot 时，除了原有折线图，再加三张直方图 ----------
        if plot:
            # 原有：三条中位数随 gamma 的折线
            plt.figure()
            plt.plot(g["gamma"], g["UL_med"], marker="o", label="median(UB - LB)")
            plt.plot(g["gamma"], g["UT_med"], marker="s", label="median(UB - TRUE)")
            plt.plot(g["gamma"], g["TL_med"], marker="^", label="median(TRUE - LB)")
            plt.title("Median gaps vs gamma (guarded LB)")
            plt.xlabel("gamma"); plt.ylabel("median gap")
            plt.legend(); plt.tight_layout(); plt.show()

            # 新增：三类差值的直方图
            plt.figure()
            plt.hist(tmp["gap_true_L"].values, bins=40)
            plt.title("Histogram of TRUE - LB (finite rows)")
            plt.xlabel("TRUE - LB"); plt.ylabel("Count")
            plt.tight_layout(); plt.show()

            plt.figure()
            plt.hist(tmp["gap_U_true"].values, bins=40)
            plt.title("Histogram of UB - TRUE (finite rows)")
            plt.xlabel("UB - TRUE"); plt.ylabel("Count")
            plt.tight_layout(); plt.show()

            plt.figure()
            plt.hist(tmp["gap_UL"].values, bins=40)
            plt.title("Histogram of UB - LB (finite rows)")
            plt.xlabel("UB - LB"); plt.ylabel("Count")
            plt.tight_layout(); plt.show()

# ===== main =====
def main():
    ap = argparse.ArgumentParser(description="SRI bounds (one-pass, guarded LB)")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--sizes", type=int, nargs="+", default=[200, 1000])
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--kinds", type=str, default="vrptw_two_point,vrptw_uniform,vrptw_triangular",
                    help="comma-separated kinds; options: vrptw_two_point, vrptw_uniform, vrptw_triangular")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--out", type=str, default="sri_bounds_final.csv")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--theta", type=float, default=0.0,
                    help="Wasserstein radius (L1). Use s=theta/(1-gamma) to left-shift samples.")

    args = ap.parse_args()
    kinds = [s.strip() for s in args.kinds.split(",")] if args.kinds else None
    df = run_experiments(seed=args.seed,
                         sizes=tuple(args.sizes),
                         reps=args.reps,
                         gammas=tuple(args.gammas),
                         theta=args.theta)

    df.to_csv(args.out, index=False)
    print(f"\nCSV saved to: {os.path.abspath(args.out)}")
    summarize(df, plot=args.plot)

if __name__ == "__main__":
    main()
