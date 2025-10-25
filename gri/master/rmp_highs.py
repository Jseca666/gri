# gri/master/rmp_highs.py
from typing import List, Dict, Tuple
# 文件顶部补充
import numpy as np
from gri.indices.rho_unified import rho_of_node
from gri.indices.gri_bisect import gri_rho
from highspy import Highs, HighsVarType
from dataclasses import dataclass
from gri.label.forward_ls import DualPack, SR3IConstraint
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
        self._cols_store = []  # 存 (cost, rows, vals, ub, route_seq) 的列表，用于重建模型
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

    def add_columns(self, cols) -> int:
        import numpy as np
        added = 0
        for c in cols:
            rows = c["rows"]
            vals = c.get("vals", [1.0] * len(rows))
            ub = float(c.get("ub", 1.0))
            if "zeta" in c:
                zeta = float(c["zeta"])
            else:
                zeta = float(self._compute_route_zeta(c))

            # 持久化到列仓库（供 solve_lp 重建时灌回）
            self._cols_store.append({
                "cost": zeta, "rows": list(rows), "vals": list(vals), "ub": ub,
                "route_seq": list(c.get("route_seq", []))
            })

            # 如果当前模型已建好行，可以立即加列（行必须先存在）
            try:
                idx_arr = np.asarray(rows, dtype=np.int32)
                val_arr = np.asarray(vals, dtype=np.float64)
                self.highs.addCol(zeta, 0.0, ub, len(rows), idx_arr, val_arr)
            except Exception:
                pass

            added += 1
        return added

    def seed_identity_columns(self, inst, params):
        """
        为每个客户 i 添加一条 0->i->0' 的“恒等列”，统一走 add_columns()（会写入 _cols_store）。
        """
        self.attach_instance(
            inst,
            rho_kind=getattr(params, "rho_kind", "SRI"),
            gamma=getattr(params, "gamma", 0.5),
        )
        self.ensure_base_rows()

        cols = []
        for i in range(inst.n_customers):
            cols.append({
                "rows": [i],  # 覆盖行索引 = i（ensure_base_rows 按 0..n-1 建的）
                "vals": [1.0],
                "ub": 1.0,
                "route_seq": [i],  # 让 _compute_route_zeta 能算 ζ_r（若未显式给 zeta）
            })
        added = self.add_columns(cols)
        print(f"[seed] added {added} identity columns")

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

        # 1) 覆盖行
        self.ensure_base_rows()

        # 2) 灌列（仅当当前模型还没有列）
        try:
            ncols_now = self.highs.getNumCols()
        except Exception:
            ncols_now = 0
        if getattr(self, "_cols_store", None) and ncols_now == 0:
            import numpy as np
            for col in self._cols_store:
                idx_arr = np.asarray(col["rows"], dtype=np.int32)
                val_arr = np.asarray(col["vals"], dtype=np.float64)
                self.highs.addCol(col["cost"], 0.0, col["ub"], len(idx_arr), idx_arr, val_arr)

        # 3) 求解
        self.highs.run()

        info = self.highs.getInfo()
        sol = self.highs.getSolution()

        if hasattr(sol, "row_dual"):
            row_duals = list(sol.row_dual)
        elif hasattr(sol, "row_duals"):
            row_duals = list(sol.row_duals)
        else:
            row_duals = []

        if hasattr(sol, "col_value"):
            col_vals = list(sol.col_value)
        elif hasattr(sol, "col_values"):
            col_vals = list(sol.col_values)
        else:
            col_vals = []

        obj = float(getattr(info, "objective_function_value", 0.0))

        try:
            nrows = self.highs.getNumRows()
            ncols = self.highs.getNumCols()
            nnz = self.highs.getNumNz()
            print(f"[LP] nrows={nrows}, ncols={ncols}, nnz={nnz}, obj={obj:.4f}")
        except Exception:
            pass

        return obj, row_duals, col_vals, sol

    def ensure_base_rows(self):
        inst = getattr(self, "_inst", None)
        if inst is None:
            raise RuntimeError("RMP.ensure_base_rows: call attach_instance(inst, ...) first.")
        try:
            nrows_now = self.highs.getNumRows()
        except Exception:
            nrows_now = 0
        if nrows_now >= inst.n_customers:
            self.cover_rows = list(range(inst.n_customers))
            self._rows_built = True
            return
        import numpy as np
        idx0 = np.empty(0, dtype=np.int32)
        val0 = np.empty(0, dtype=np.float64)
        for _ in range(inst.n_customers):
            self.highs.addRow(1.0, 1.0, 0, idx0, val0)  # 位置参数 + numpy 数组
        self.cover_rows = list(range(inst.n_customers))
        self._rows_built = True

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

    # 在文件顶部确保有：
    # from gri.label.forward_ls import DualPack, SR3IConstraint

    def build_dualpack(self, inst, row_duals):
        """
        把 RMP 的行对偶映射到定价需要的 DualPack：
        - u: 覆盖对偶（前 n 个）
        - v: 预算对偶（此版本默认无 -> 0.0）
        - pi_arc[(i,j)]: CI/2PI 的弧级对偶贡献（≥ 型，对偶 ≤ 0 -> 取 -dual 累加为正）
        - sr3i: [(T, phi)]，其中 phi = -dual（≤ 型，对偶 ≥ 0；定价里以 “+phi” 进入）
        - rho_kind/gamma: 若未在 RMP 上设置，给默认值
        """
        n = inst.n_customers
        m = len(row_duals) if row_duals is not None else 0

        # 1) 覆盖对偶 u：直接取 row_duals 的前 n 个；不足则补 0
        if m >= n:
            u = [float(d) for d in row_duals[:n]]
        else:
            u = [0.0] * n

        # 2) 预算对偶 v：当前实现没有预算行，置 0（后续若加预算行，再从 row_duals 取）
        v = 0.0

        # 3) 2PI/CI → 弧对偶 π_ij
        pi_arc = {}

        def _get_set_from_cut(cut, key1, key2):
            # 兼容 cut.data 里存 "S"/"C" 或 cut.S/cut.T 的写法
            if hasattr(cut, key1):
                return getattr(cut, key1)
            if hasattr(cut, "data") and isinstance(cut.data, dict):
                if key2 in cut.data:
                    return cut.data[key2]
            return None

        cuts_iter = self.cuts if isinstance(self.cuts, (list, tuple)) else getattr(self.cuts, "values", lambda: [])()
        for cut in cuts_iter:
            # 容错拿到该割的 dual 值
            d = 0.0
            idx = getattr(cut, "row_index", None)
            if idx is not None and 0 <= int(idx) < m:
                d = float(row_duals[int(idx)])

            kind = getattr(cut, "kind", "").upper()
            if kind in ("CI", "2PI"):
                S = _get_set_from_cut(cut, "S", "S")
                if S is None:
                    S = _get_set_from_cut(cut, "T", "S")  # 兜底
                if S is None:
                    continue
                # ≥ 型：dual ≤ 0；把 (-dual) 聚合到 δ⁺(S) 的弧上
                coef = max(0.0, -d)
                S = set(S)
                for i in S:
                    for j in range(n):
                        if j not in S and j != i:
                            pi_arc[(i, j)] = pi_arc.get((i, j), 0.0) + coef

        # 4) SR3I 列表：≤ 型，dual ≥ 0；定价里我们用 “+phi”，因此取 phi = -dual
        sr3i = []
        cuts_iter = self.cuts if isinstance(self.cuts, (list, tuple)) else getattr(self.cuts, "values", lambda: [])()
        for cut in cuts_iter:
            kind = getattr(cut, "kind", "").upper()
            if kind == "SR3I":
                idx = getattr(cut, "row_index", None)
                d = 0.0
                if idx is not None and 0 <= int(idx) < m:
                    d = float(row_duals[int(idx)])
                T = _get_set_from_cut(cut, "T", "C")
                if T is None:
                    T = _get_set_from_cut(cut, "C", "C")
                if T is None:
                    continue
                sr3i.append(SR3IConstraint(T=tuple(sorted(T)), phi=-d))

        return DualPack(
            u=u, v=v,
            pi_arc=pi_arc,
            sr3i=sr3i,
            rho_kind=getattr(self, "rho_kind", "SRI"),
            gamma=getattr(self, "gamma", 0.5),
        )

    def attach_instance(self, inst, rho_kind: str = "SRI", gamma: float = 0.5):
        """把数据实例挂到 RMP 上，供计算列目标 zeta_r 使用。"""
        self._inst = inst
        self.rho_kind = rho_kind
        self.gamma = gamma

    def _compute_route_zeta(self, route_dict) -> float:
        """
        输入列字典（至少包含 route_seq 或 rows），计算 ζ_r = ∑_j rho(ξ_j).
        - route_seq: 客户索引序列（0..n-1）
        需要 self.attach_instance(inst, ...) 预先绑定实例。
        """
        inst = getattr(self, "_inst", None)
        if inst is None:
            raise RuntimeError(
                "RMP: instance not attached. Call rmp.attach_instance(inst, rho_kind, gamma) before add_columns.")

        seq = route_dict.get("route_seq") or route_dict.get("rows") or []
        seq = list(seq)
        if not seq:
            return 0.0

        N = int(inst.N)
        tau = np.zeros(N, dtype=np.float64)  # {τ^(ω)}，每个样本的最早开始时间
        zeta = 0.0
        prev = -1
        for j in seq:
            if prev < 0:
                # depot -> j
                arr = tau + inst.tt0[j] + inst.delay0[:, j]
                tau = np.maximum(arr, float(inst.e[j]))
            else:
                # prev -> j
                arr = tau + float(inst.s[prev]) + inst.tt[prev, j] + inst.delay[:, prev, j]
                tau = np.maximum(arr, float(inst.e[j]))
            # 节点 j 的 ρ(ξ_j)
            zeta += rho_of_node(tau, float(inst.l[j]), kind=getattr(self, "rho_kind", "SRI"),
                                gamma=float(getattr(self, "gamma", 0.5)))
            prev = j
        return float(zeta)


def _zeta_singleton(inst, i, params):
    """0->i->0' 的单客户路径的 zeta = rho(xi_i)。
       粗化计算：用样本旅行时间在 i 处形成的延误 ξ 的样本，套用 rho 评估。
    """
    # 到达 i 的最早服务时间（简化：只走 0->i）
    e_i, l_i = inst.e[i], inst.l[i]
    # 样本数 N 及样本化 0->i 的行程
    N = len(inst.samples)  # 你这边 inst 已有样本；若字段不同，按你的结构取
    xi = np.empty(N, dtype=float)
    for k in range(N):
        t0i = inst.samples[k].tt[0, i]  # 若你的样本访问方式不同，改这里
        tau_i = max(e_i, t0i)           # 简化：只考虑到达 i 的一跳排队/等待
        xi[k] = tau_i - l_i             # 论文式(2)的延误
    # 评估 rho：默认用 GRI（二分法）。如果你当前用 SRI/ERI/RVI，请替换调用。
    return float(max(0.0, gri_rho(xi, params.phi_kind, params.gamma, tol=params.rho_tol)))