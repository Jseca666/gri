# gri/label/forward_ls.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Iterable
import numpy as np

from gri.indices.rho_unified import rho_of_node

@dataclass
class SR3IConstraint:
    T: Tuple[int, ...]   # 顶点子集（客户索引，从 0..n-1）
    phi: float           # 对偶 φ_T

@dataclass
class DualPack:
    # 覆盖约束对偶 u_i，预算对偶 v（若未建预算行则 =0）
    u: List[float]
    v: float
    # 2PI/CI 的弧级对偶贡献 π_ij = sum_{S:(i,j)∈δ+(S)}(π_S + η_S)
    pi_arc: Dict[Tuple[int, int], float]
    # SR3I 列表
    sr3i: List[SR3IConstraint]
    # ρ 选择与参数
    rho_kind: str = "SRI"
    gamma: float = 0.5

@dataclass
class ParamsForPricing:
    kmax: int = 5          # 最大扩展深度（或交付用的 beam/宽搜层数）
    topk: int = 30         # 每层保留的标签上限（简单 beam 控制）
    rc_eps: float = -1e-9  # 负减价阈值
    # 其它调参：是否检查容量/时间窗等，默认 True
    check_time: bool = True
    check_capacity: bool = True

@dataclass
class Label:
    i: int                        # 当前末端顶点
    X: Tuple[int, ...]            # 已访问客户序列（用于去环/简易主导）
    cost: float                   # 累计行驶成本（用于预算项 v·cost）
    rc: float                     # 当前减价
    w: float                      # 当前载重
    tau_mu: float                 # 均值条件下的最早开始时间 τ(μ)
    zeta: float                   # 已累计的 ρ
    tau_omega: np.ndarray         # 逐样本的最早开始时间 {τ(ω)}
    H: np.ndarray                 # SR3I 二元资源（|C| 维 0/1）
    route_seq: Tuple[int, ...]    # 记录路径，生成列时用

def _init_label(inst, duals: DualPack) -> Label:
    n = inst.n_customers
    return Label(
        i=-1, X=tuple(), cost=0.0,
        rc= -sum(duals.u),    # 起点 rc 初始化为 -∑ u_i（便于把“覆盖对偶”抵消到每个新增 j）
        w=0.0,
        tau_mu=0.0,
        zeta=0.0,
        tau_omega=np.zeros(inst.N, dtype=np.float64),
        H=np.zeros(len(duals.sr3i), dtype=np.int8),
        route_seq=tuple()
    )

def _pi_arc(duals: DualPack, i: int, j: int) -> float:
    return duals.pi_arc.get((i, j), 0.0) if i >= 0 else duals.pi_arc.get(("dep", j), 0.0)

def _extend(inst, duals: DualPack, L: Label, j: int) -> Label | None:
    # 容量约束
    if L.w + float(inst.q[j]) > float(inst.Q):
        return None

    # 均值时间窗可行性（用于剪枝）
    travel_mu = float(inst.tt0[j]) if L.i < 0 else float(inst.tt[L.i, j])
    tau_mu_next = max(float(inst.e[j]), L.tau_mu + (0.0 if L.i < 0 else float(inst.s[L.i])) + travel_mu)
    if tau_mu_next > float(inst.l[j]):  # 均值下已迟到则剪（原文准则）
        return None

    # 样本到达时间 {τ(ω)} 更新
    if L.i < 0:
        # depot -> j
        arr = L.tau_omega + inst.tt0[j] + inst.delay0[:, j]
    else:
        arr = L.tau_omega + inst.s[L.i] + inst.tt[L.i, j] + inst.delay[:, L.i, j]
    tau_omega_next = np.maximum(arr, float(inst.e[j]))

    # 单节点 ρ(ξ_j) 评估（经验分布）
    rho_j = rho_of_node(tau_omega_next, float(inst.l[j]),
                        kind=duals.rho_kind, gamma=duals.gamma)

    # 成本 + 减价更新（式 (12a)）
    # 基础行驶成本（若有预算对偶 v，会进入 rc）
    travel_cost = travel_mu  # 这里用均值时长作为 cost；如你已有路成本 cr，可直接替换
    cost_next = L.cost + travel_cost

    # 覆盖对偶：新增 j 抵消 +u_j
    u_term = -float(duals.u[j])

    # 预算对偶：v * travel_cost
    v_term = duals.v * travel_cost if duals.v else 0.0

    # 2PI/CI 弧对偶：π_ij
    pi_term = _pi_arc(duals, L.i, j)

    # SR3I 二元资源：若当前某 κ 的 T(κ) 中包含 j 且 κ(L)=1，则 +φ(κ)
    sr3i_term = 0.0
    H_next = L.H.copy()
    for idx, c in enumerate(duals.sr3i):
        if j in c.T:
            if H_next[idx] == 1:
                sr3i_term += c.phi
            # (12b) 翻转
            H_next[idx] = (H_next[idx] + 1) & 1

    rc_next = L.rc + rho_j + u_term + v_term - pi_term + sr3i_term

    return Label(
        i=j,
        X=L.X + (j,),
        cost=cost_next,
        rc=rc_next,
        w=L.w + float(inst.q[j]),
        tau_mu=tau_mu_next,
        zeta=L.zeta + rho_j,
        tau_omega=tau_omega_next,
        H=H_next,
        route_seq=L.route_seq + (j,)
    )

def _dominates(a: Label, b: Label) -> bool:
    # 轻量主导（同尾节点且访问集一致/包含时才比较）
    return (a.i == b.i and set(a.X).issubset(b.X)
            and a.w <= b.w + 1e-9
            and a.tau_mu <= b.tau_mu + 1e-9
            and a.zeta <= b.zeta + 1e-9
            and a.rc <= b.rc + 1e-12)

def _prune(labels: List[Label]) -> List[Label]:
    # 简单 O(K^2) 主导筛，K 不大（用在 beam 限制后）
    kept: List[Label] = []
    for L in labels:
        dominated = False
        for K in kept:
            if _dominates(K, L):
                dominated = True
                break
        if not dominated:
            # 反向再清掉被 L 支配的
            kept = [K for K in kept if not _dominates(L, K)]
            kept.append(L)
    return kept

def find_negative_rc_routes(inst, duals: DualPack, params: ParamsForPricing) -> List[dict]:
    """
    前向标签定价：返回负减价列（若有），列结构: dict(cost, rows, route_seq, vals, ub)
    - rows: 覆盖到的客户索引列表
    - vals: 与 rows 对应的系数（1）
    """
    # 初始化
    beams: List[Label] = [_init_label(inst, duals)]
    best_cols: List[dict] = []

    for depth in range(params.kmax):
        new_beams: List[Label] = []
        # 逐标签扩展
        for L in beams:
            # 枚举未访问客户
            cand = [j for j in range(inst.n_customers) if j not in L.X]
            for j in cand:
                Lj = _extend(inst, duals, L, j)
                if Lj is None:
                    continue
                new_beams.append(Lj)
                # 生成一条“返回仓库”的完整路试探（可按需要检查回仓均值时窗）
                rc_with_back = Lj.rc  # 回仓项可并入 travel_cost + v*…，此处略去
                if rc_with_back < params.rc_eps:
                    best_cols.append({
                        "cost": Lj.cost,
                        "rows": list(Lj.X),
                        "route_seq": list(Lj.route_seq),
                        "vals": [1.0] * len(Lj.X),
                        "ub": 1.0
                    })
        if not new_beams:
            break
        # 主导 + beam 限制
        new_beams = _prune(new_beams)
        # 取 rc 最小的 topk
        new_beams.sort(key=lambda x: x.rc)
        beams = new_beams[:params.topk]

    return best_cols


