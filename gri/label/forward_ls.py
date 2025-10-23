# gri/label/forward_ls.py
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from gri.indices.gri_bisect import sri_rho
from gri.master.duals import DualPack

@dataclass
class Label:
    node: int
    visited_bits: int
    route: List[int]
    tau_mu: float
    tau_omega: np.ndarray
    zeta: float
    det_cost: float
    load: float
    # SR3I 奇偶资源（与 DualPack.sr3_sets 对齐；0=偶，1=奇）
    sr3_parity: np.ndarray

def _bit(i: int) -> int: return 1 << i
def _not_visited(bits: int, j: int) -> bool: return (bits & _bit(j)) == 0

def _dominates(a: Label, b: Label) -> bool:
    return (a.zeta <= b.zeta and a.tau_mu <= b.tau_mu and a.det_cost <= b.det_cost and a.load <= b.load) and \
           (a.zeta < b.zeta or a.tau_mu < b.tau_mu or a.det_cost < b.det_cost or a.load < b.load)

def _pareto_prune(labels: List[Label], cap: int) -> List[Label]:
    pruned: List[Label] = []
    for L in labels:
        dominated = False; to_remove = []
        for P in pruned:
            if _dominates(P, L): dominated = True; break
            if _dominates(L, P): to_remove.append(P)
        if dominated: continue
        for R in to_remove: pruned.remove(R)
        pruned.append(L)
        if len(pruned) > cap:
            pruned.sort(key=lambda x: (x.zeta, x.tau_mu, x.det_cost))
            pruned = pruned[:cap]
    return pruned

def pricing_forward(instance, duals: DualPack, params) -> List[Dict]:
    """
    reduced cost 按论文式(11)+(12a)(12b)：
      c' = c + ρ_j  − u_j  − v * d_ij  − π_ij  − Σ_{κ: parity=1 ∧ j∈T(κ)} φ_T(κ)
    其中 v 是预算约束(3d)的对偶，π_ij 汇聚了 2PI/CI 的对偶。SR3I 用“奇偶资源”切换，见(12b)。
    """
    n, N, depot = instance.n_customers, instance.N, instance.n_customers
    eps_det = float(getattr(params, "det_eps", 1e-3))
    eps_ret = float(getattr(params, "ret_eps", 1e-3))
    KMAX = int(getattr(params, "kmax", 4))
    gamma = float(getattr(params, "gamma", 0.5))
    pareto_cap = int(getattr(params, "pareto_cap", 16))
    tol = 1e-12

    Q = float(getattr(instance, "Q", 1e18))
    q = getattr(instance, "q", np.zeros(n, dtype=np.float64))
    s = getattr(instance, "s", np.zeros(n, dtype=np.float64))

    m_sr3 = len(duals.sr3_sets)
    init = Label(
        node=depot, visited_bits=0, route=[],
        tau_mu=0.0, tau_omega=np.zeros(N, dtype=np.float64),
        zeta=0.0, det_cost=0.0, load=0.0,
        sr3_parity=np.zeros(m_sr3, dtype=np.int8)
    )
    frontier = [init]
    candidates: List[Tuple[float, Dict]] = []

    for depth in range(1, KMAX + 1):
        buckets: Dict[Tuple[int, int], List[Label]] = {}
        for L in frontier:
            for j in range(n):
                if not _not_visited(L.visited_bits, j): continue
                if L.load + float(q[j]) > Q + 1e-12: continue

                # 到达
                if L.node == depot:
                    d_ij = float(instance.tt0[j])
                    arr_mu = L.tau_mu + d_ij
                    arr_om = L.tau_omega + instance.delay0[:, j]
                    det_new = L.det_cost + d_ij
                    prev = depot
                else:
                    d_ij = float(instance.tt[L.node, j])
                    arr_mu = L.tau_mu + d_ij
                    arr_om = L.tau_omega + instance.delay[:, L.node, j]
                    det_new = L.det_cost + d_ij
                    prev = L.node

                # 等待 + 服务
                e_j = float(instance.e[j])
                start_mu = arr_mu if arr_mu >= e_j else e_j
                start_om = np.maximum(arr_om, e_j)
                tau_mu_new = start_mu + float(s[j])
                tau_omega_new = start_om + float(s[j])

                # 风险：SRI
                l_j = float(instance.l[j]); scale = float(max(instance.scale[j], 1e-6))
                xi = (tau_omega_new - l_j) / scale
                rho_j = sri_rho(xi, gamma, tol=getattr(params, "bisect_tol", 1e-3))

                zeta_new = L.zeta + rho_j
                bits_new = L.visited_bits | _bit(j)
                route_new = L.route + [j]
                load_new  = L.load + float(q[j])

                # SR3I φ：按(12b)触发（先用扩展前的奇偶）
                phi_sr3 = duals.sr3_phi_if_trigger(L.sr3_parity, j)
                parity_new = L.sr3_parity.copy()
                for idx, T in enumerate(duals.sr3_sets):
                    if j in T:
                        parity_new[idx] = (parity_new[idx] + 1) & 1  # mod 2

                # 列成本（风控目标 + 极小确定性权重 + 回仓微惩罚）
                ret_cost = float(instance.tt_back[j])
                cost = float(zeta_new + eps_det * det_new + eps_ret * ret_cost)

                # 弧上的 cut 对偶（2PI/CI）
                pi_ij = duals.arc_penalty(prev, j)
                # 覆盖对偶之和
                dual_u = float(sum(duals.u[i] for i in route_new))
                # 预算对偶 v*d_ij
                v_term = duals.v * d_ij

                # 按(12a)得 rc
                # 弧上的 cut 对偶
                pi_ij = duals.arc_penalty(prev, j)
                dual_u = float(sum(duals.u[i] for i in route_new))
                v_term = duals.v * d_ij
                # ——注意 SR3I：应“+ φ”而不是减 ——（φ≤0）
                phi_sr3 = duals.sr3_phi_if_trigger(L.sr3_parity, j)

                rc = cost - dual_u - v_term - pi_ij + phi_sr3  # ← 关键改动

                # 返回列时带上 route_seq，便于 RMP 后续给新割补系数
                if rc < -tol and len(route_new) >= 2:
                    candidates.append((rc, {
                        "rows": route_new, "route_seq": route_new,
                        "vals": [1.0] * len(route_new), "ub": 1.0, "cost": cost
                    }))
                lab_new = Label(
                    node=j, visited_bits=bits_new, route=route_new,
                    tau_mu=tau_mu_new, tau_omega=tau_omega_new,
                    zeta=zeta_new, det_cost=det_new, load=load_new,
                    sr3_parity=parity_new
                )
                buckets.setdefault((lab_new.node, lab_new.visited_bits), []).append(lab_new)

        frontier = [L for labs in buckets.values() for L in _pareto_prune(labs, pareto_cap)]

    if not candidates: return []
    candidates.sort(key=lambda x: x[0])  # 最负优先
    topk = int(getattr(params, "cols_topk", 12))
    return [col for _, col in candidates[:topk]]
