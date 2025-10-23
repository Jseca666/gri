# gri/master/separate.py
import math
from typing import List, Dict, Tuple, Set
from itertools import combinations
from dataclasses import dataclass
from .rmp_highs import CutRow

def _route_seq(meta: Dict) -> List[int]:
    return meta.get("route_seq", meta.get("rows", []))

def _boundary_cross(route: List[int], S: Set[int], depot: int) -> int:
    # 统计 δ⁺(S)：从 S 出到 S̄ 的弧次数（视路线为 depot->...->depot）
    prev = depot; c = 0
    for v in route:
        if (prev in S) and (v not in S):
            c += 1
        prev = v
    if (prev in S) and (depot not in S):
        c += 1
    return c

def separate_CI(rmp, instance, y: List[float], max_add: int = 3, k_max: int = 4):
    """ 容量割（有向形式）：∑_{δ⁺(S)} y ≥ ceil(∑_{i∈S} q_i / Q) """
    n = instance.n_customers; depot = n
    added: List[CutRow] = []
    for k in range(2, min(k_max, n-1)+1):
        for S_tuple in combinations(range(n), k):
            S = set(S_tuple)
            rhs = float(math.ceil(float(sum(instance.q[i] for i in S)) / float(instance.Q)))
            if rhs <= 0.0:
                continue
            lhs = 0.0
            for val, meta in zip(y, rmp.cols_meta):   # 与 y 对齐，避免越界
                lhs += _boundary_cross(_route_seq(meta), S, depot) * float(val)
            if lhs + 1e-6 < rhs:
                added.append(CutRow(kind="CI", data={"S": S}, row_index=-1, sense=">=", rhs=rhs))
                if len(added) >= max_add:
                    return added
    return added

def separate_SR3I(rmp, instance, y: List[float], max_add: int = 3):
    """ SR3I：对所有三元组 C，sum_{r:|R∩C|>=2} y_r <= 1 """
    n = instance.n_customers
    added: List[CutRow] = []
    for C in combinations(range(n), 3):
        lhs = 0.0
        for val, meta in zip(y, rmp.cols_meta):
            if len(set(_route_seq(meta)) & set(C)) >= 2:
                lhs += float(val)
        if lhs > 1.0 + 1e-6:
            added.append(CutRow(kind="SR3I", data={"C": frozenset(C)}, row_index=-1, sense="<=", rhs=1.0))
            if len(added) >= max_add:
                return added
    return added

def separate_2PI_heur(rmp, instance, y: List[float], max_add: int = 1, k_max: int = 3):
    """
    2PI 启发式：sum q(S) <= Q 且“均值时间窗”难以单车服务时 ⇒ ∑_{δ⁺(S)} y ≥ 2
    """
    n = instance.n_customers; depot = n
    added: List[CutRow] = []
    for k in range(2, min(k_max, n-1)+1):
        for S_tuple in combinations(range(n), k):
            S = set(S_tuple)
            if sum(instance.q[i] for i in S) > instance.Q + 1e-9:
                continue
            hard = 0
            for i in S:
                best_pre = min([instance.tt0[i]] + [instance.tt[j, i] for j in S if j != i])
                if instance.e[i] + best_pre > instance.l[i] + 1e-9:
                    hard += 1
            if hard == 0:
                continue
            rhs = 2.0
            lhs = 0.0
            for val, meta in zip(y, rmp.cols_meta):
                lhs += _boundary_cross(_route_seq(meta), S, depot) * float(val)
            if lhs + 1e-6 < rhs:
                added.append(CutRow(kind="2PI", data={"S": S}, row_index=-1, sense=">=", rhs=rhs))
                if len(added) >= max_add:
                    return added
    return added
