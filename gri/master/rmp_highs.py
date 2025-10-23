# gri/master/rmp_highs.py
from typing import List, Dict, Tuple
from highspy import Highs, HighsVarType
from dataclasses import dataclass

Row = int

@dataclass
class CutRow:
    kind: str                 # "CI" | "2PI" | "SR3I"
    data: dict                # {"S": set(int)} or {"C": frozenset(int)}
    row_index: Row            # 在 HiGHS 中的行索引
    sense: str                # ">=" or "<="
    rhs: float

class RMP:
    def __init__(self, n_customers: int):
        self.n = int(n_customers)
        self.highs = Highs()
        self.highs.setOptionValue("log_to_console", True)
        self.highs.setOptionValue("presolve", "on")
        self.highs.setOptionValue("parallel", "on")
        # 覆盖等式：每个客户 = 1
        for _ in range(self.n):
            self.highs.addRow(1.0, 1.0, 0, [], [])
        self.num_cols = 0
        self.is_integer = False
        self._seen_cols = set()           # 去重：按覆盖集
        self.cols_meta: List[Dict] = []   # 每列的 {route_seq: [...]} 等元信息
        self.cuts: List[CutRow] = []      # 已加入的割行
        self._cut_keys = set()
    # ——工具：给当前所有列计算某一条割行的系数——
    def _coef_for_col_on_cut(self, jcol: int, cut: CutRow) -> float:
        meta = self.cols_meta[jcol]
        route = meta.get("route_seq", meta.get("rows", []))
        n = self.n
        depot = n
        if cut.kind in ("CI", "2PI"):
            S = cut.data["S"]
            # 构造弧序列：0->r0, r0->r1, ..., r_last->0（用 depot=n 占位）
            prev = depot
            crossing = 0
            for v in route:
                if (prev in S) and (v not in S):
                    crossing += 1
                prev = v
            # 回仓弧
            if (prev in S) and (depot not in S):
                crossing += 1
            return float(crossing)
        elif cut.kind == "SR3I":
            C = cut.data["C"]
            return 1.0 if len(set(route) & C) >= 2 else 0.0
        return 0.0

    def _build_full_col(self, cost: float, rows_idx: List[int], vals: List[float], meta: Dict):
        # 合并重复行索引，避免 HiGHS "duplicate index"
        rowcoef = {}
        for r, v in zip(rows_idx, vals):
            rr = int(r)
            rowcoef[rr] = rowcoef.get(rr, 0.0) + float(v)

        # 已加入的割行：根据 route 元数据补系数
        for cut in self.cuts:
            coef = self._coef_for_route_on_cut(meta, cut)
            if coef != 0.0:
                rr = int(cut.row_index)
                rowcoef[rr] = rowcoef.get(rr, 0.0) + float(coef)

        rows = list(rowcoef.keys())
        vals = [rowcoef[r] for r in rows]
        self.highs.addCol(float(cost), 0.0, 1.0, len(rows), rows, vals)

    def _coef_for_route_on_cut(self, meta: Dict, cut: CutRow) -> float:
        route = meta.get("route_seq", meta.get("rows", []))
        n = self.n
        depot = n
        if cut.kind in ("CI", "2PI"):
            S = cut.data["S"]
            prev = depot
            crossing = 0
            for v in route:
                if (prev in S) and (v not in S):
                    crossing += 1
                prev = v
            if (prev in S) and (depot not in S):
                crossing += 1
            return float(crossing)
        elif cut.kind == "SR3I":
            C = cut.data["C"]
            return 1.0 if len(set(route) & C) >= 2 else 0.0
        return 0.0

    def add_columns(self, cols: List[Dict]) -> int:
        added = 0
        for c in cols:
            rows = tuple(sorted(int(i) for i in c["rows"]))
            if rows in self._seen_cols:
                continue
            cost = float(c["cost"])
            vals = list(map(float, c.get("vals", [1.0]*len(rows))))
            meta = {"route_seq": list(c.get("route_seq", list(rows))), "rows": list(rows)}
            # 组装包含割行的完整列
            self._build_full_col(cost, list(rows), vals, meta)
            self.cols_meta.append(meta)
            self.num_cols += 1
            self._seen_cols.add(rows)
            added += 1
        return added

    def add_cuts(self, cuts: List[CutRow]) -> int:
        if not cuts:
            return 0
        added = 0
        new_rows = []
        for cut in cuts:
            key = (cut.kind, frozenset(cut.data["S"]) if cut.kind in ("CI", "2PI") else frozenset(cut.data["C"]))
            if key in self._cut_keys:
                continue
            if cut.kind == "SR3I":
                irow = self.highs.addRow(-1e20, float(cut.rhs), 0, [], [])  # ≤ 型
            else:
                irow = self.highs.addRow(float(cut.rhs), 1e20, 0, [], [])  # ≥ 型
            cut.row_index = irow
            self.cuts.append(cut)
            self._cut_keys.add(key)
            new_rows.append(cut)
            added += 1

        # 给已存在的所有列补上这些新割的系数
        for jcol, meta in enumerate(self.cols_meta):
            for cut in new_rows:
                coef = self._coef_for_route_on_cut(meta, cut)
                if coef != 0.0:
                    self.highs.changeCoeff(cut.row_index, jcol, coef)
        return added

    def solve_lp(self, time_limit: float = 60.0):
        self.highs.setOptionValue("time_limit", float(time_limit))
        self.highs.setOptionValue("solver", "simplex")
        self.highs.run()

        sol = self.highs.getSolution()
        info = self.highs.getInfo()

        # 兼容不同 highspy 版本字段名
        if hasattr(sol, "row_dual"):
            row_duals = list(sol.row_dual)
        else:
            row_duals = list(sol.row_duals)

        if hasattr(sol, "col_value"):
            col_vals = list(sol.col_value)
        else:
            col_vals = list(sol.col_values)

        obj = info.objective_function_value
        return obj, row_duals, col_vals, sol

    def to_integer_and_solve(self, mip_gap: float = 0.01, time_limit: float = 60.0):
        for j in range(self.num_cols):
            self.highs.changeColIntegrality(j, HighsVarType.kInteger)
        self.is_integer = True
        self.highs.setOptionValue("time_limit", float(time_limit))
        self.highs.setOptionValue("mip_rel_gap", float(mip_gap))
        self.highs.run()
        sol = self.highs.getSolution()
        info = self.highs.getInfo()
        return info.objective_function_value, sol

    # ——把 RMP 的行对偶整理成 DualPack（式(11)/(12a)(12b) 要用）——
    def build_dualpack(self, instance, row_duals):
        from .duals import DualPack
        m = len(row_duals) if row_duals is not None else 0
        u = row_duals[:self.n] if m >= self.n else [0.0] * self.n

        pi_ij = {}
        eta_ij = {}
        sr3_sets = []
        phi = {}

        # 覆盖行之后按顺序对应各割；若 row_duals 不足，则该割对偶记 0
        for k, cut in enumerate(self.cuts):
            idx = self.n + k
            d = float(row_duals[idx]) if (row_duals is not None and idx < m) else 0.0

            if cut.kind in ("CI", "2PI"):
                # ≥ 型：dual ≤ 0，取 π = -dual ≥ 0；聚合到 δ⁺(S) 的弧上
                coef = max(0.0, -d)
                S = cut.data["S"]
                n = self.n;
                depot = n
                V = list(range(n)) + [depot]
                for i in V:
                    for j in V:
                        if i == j:
                            continue
                        if (i in S) and (j not in S):
                            pi_ij[(i, j)] = pi_ij.get((i, j), 0.0) + coef

            elif cut.kind == "SR3I":
                # ≤ 型：dual ≥ 0，取 φ = -dual ≤ 0（定价里以 “+ φ” 进入）
                T = frozenset(cut.data["C"])
                sr3_sets.append(T)
                phi[T] = -d

        return DualPack(u=u, v=0.0, pi_ij=pi_ij, eta_ij=eta_ij, sr3_sets=sr3_sets, phi=phi)
