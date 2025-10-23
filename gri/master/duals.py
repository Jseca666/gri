# gri/master/duals.py
from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple, List

Arc = Tuple[int, int]

@dataclass
class DualPack:
    # 覆盖对偶 u_i（客户约束）
    u: List[float]
    # 预算对偶 v（≤0）：在式(12a)里以 “- v * d_ij” 逐弧进入减价
    v: float = 0.0
    # 2PI/CI 对偶在每条弧上的聚合：π_ij = sum_{S: (i,j)∈δ+(S)} π_S ；η_ij 同理
    pi_ij: Dict[Arc, float] = None
    eta_ij: Dict[Arc, float] = None
    # SR3I：每个不相交子集 T 及其对偶 φ_T；用“奇偶资源”跟踪访问次数
    sr3_sets: List[FrozenSet[int]] = None
    phi: Dict[FrozenSet[int], float] = None

    def __post_init__(self):
        if self.pi_ij is None: self.pi_ij = {}
        if self.eta_ij is None: self.eta_ij = {}
        if self.sr3_sets is None: self.sr3_sets = []
        if self.phi is None: self.phi = {}

    def arc_penalty(self, i: int, j: int) -> float:
        return float(self.pi_ij.get((i, j), 0.0) + self.eta_ij.get((i, j), 0.0))

    def sr3_phi_if_trigger(self, parity_vec, j: int) -> float:
        # 若某 SR3 资源当前为1且 j∈T，则式(12a)扣 φ_T
        add = 0.0
        for idx, T in enumerate(self.sr3_sets):
            if parity_vec[idx] == 1 and (j in T):
                add += float(self.phi.get(T, 0.0))
        return add
