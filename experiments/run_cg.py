# experiments/run_cg.py
from dataclasses import dataclass
from gri.data.solomon import load_solomon_instance
import os

import numpy as np
from gri.master.rmp_highs import RMP
from gri.pricing.reduced_cost import find_negative_rc_routes
from gri.master.separate import separate_CI, separate_SR3I, separate_2PI_heur
from gri.master.duals import DualPack
@dataclass
class Params:
    data_path: str = r"D:\PycharmProjects\gri-replication\gri\data\solomon\25\C108.txt"  # 按实际路径修改
    use_solomon: bool = True  # 切换开关
    n_cust: int = 25  # 取前 25 客户

    lp_time: float = 1.0
    mip_time: float = 10.0
    gap: float = 0.01
    # 定价控制
    kmax: int = 4
    gamma: float = 0.5
    bisect_tol: float = 1e-3
    det_eps: float = 1e-3
    ret_eps: float = 1e-3
    pareto_cap: int = 16
    cols_topk: int = 12
    # 调试
    debug_dual: bool = True
    max_iters: int = 10

class ToyInstance:
    """玩具实例 + 时间窗 + 服务时间 + 容量 + 回仓"""
    def __init__(self, n_customers: int, N: int = 200, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.n_customers = n_customers
        self.N = N

        # 确定性时间
        self.tt0 = rng.integers(low=5, high=12, size=n_customers).astype(np.float64)
        base = rng.integers(low=3, high=10, size=(n_customers, n_customers)).astype(np.float64)
        self.tt = (base + base.T) / 2.0
        np.fill_diagonal(self.tt, 0.0)
        # 回仓时间
        self.tt_back = rng.integers(low=4, high=10, size=n_customers).astype(np.float64)

        # 时间窗与服务时间
        self.e = self.tt0 - rng.integers(low=1, high=3, size=n_customers).astype(np.float64)
        self.e = np.maximum(0.0, self.e)
        self.s = rng.integers(low=1, high=3, size=n_customers).astype(np.float64)  # 服务时间
        self.l = self.tt0 + self.s + rng.integers(low=6, high=10, size=n_customers).astype(np.float64)

        # 归一化尺度
        self.scale = np.maximum(1.0, 0.3 * self.l)

        # 容量与需求
        self.Q = float(10.0)
        self.q = rng.integers(low=1, high=4, size=n_customers).astype(np.float64)

        # 延迟扰动：零均值
        d0_raw = rng.lognormal(mean=-1.8, sigma=1.0, size=(N, n_customers))
        self.delay0 = (d0_raw - d0_raw.mean(axis=0, keepdims=True)).astype(np.float64)

        d_raw = rng.lognormal(mean=-1.8, sigma=1.0, size=(N, n_customers, n_customers))
        d_raw = 0.5 * (d_raw + np.transpose(d_raw, (0, 2, 1)))
        for w in range(N):
            np.fill_diagonal(d_raw[w], 0.0)
        self.delay = (d_raw - d_raw.mean(axis=0, keepdims=True)).astype(np.float64)

def seed_identity_columns(n: int):
    return [{"cost": 1.0, "rows": [i], "vals": [1.0], "ub": 1.0} for i in range(n)]

def main():
    # inst = ToyInstance(n_customers=6, N=200, seed=2025)
    # params = Params()
    params = Params()
    if params.use_solomon:
        if not os.path.exists(params.data_path):
            raise FileNotFoundError(f"Solomon 文件不存在: {params.data_path}")
        inst = load_solomon_instance(
            path=params.data_path,
            n_customers=params.n_cust,
            N=200,  # 样本数
            speed=1.0,
            sigma_arc=0.8, sigma_dep=0.8,
            seed=2025
        )
        print(f"[data] loaded Solomon: {os.path.basename(params.data_path)}, n={inst.n_customers}, Q={inst.Q:g}")
    else:
        inst = ToyInstance(n_customers=6, N=200, seed=2025)

    rmp = RMP(n_customers=inst.n_customers)
    rmp.add_columns([{"cost": 1.0, "rows": [i], "route_seq": [i], "vals": [1.0], "ub": 1.0}
                     for i in range(inst.n_customers)])
    print(f"[seed] added {inst.n_customers} identity columns")

    for it in range(1, params.max_iters + 1):
        # 1) 解 LP（拿 u 与 y）
        obj, row_duals, y, _ = rmp.solve_lp(time_limit=params.lp_time)
        print(f"[iter {it}] LP obj={obj:.4f}")

        # 2) 分离少量割（避免行膨胀）
        cuts = []
        cuts += separate_CI(rmp, inst, y, max_add=3, k_max=4)
        cuts += separate_SR3I(rmp, inst, y, max_add=3)
        cuts += separate_2PI_heur(rmp, inst, y, max_add=1, k_max=3)
        if cuts:
            added_rows = rmp.add_cuts(cuts)
            print(f"[iter {it}] added {added_rows} cuts")

        # 3) 用本轮 row_duals 构造 DualPack（新割对偶若没有就按 0 处理）
        duals = rmp.build_dualpack(inst, row_duals)

        # 4) 定价补列
        new_cols = find_negative_rc_routes(inst, duals, params)
        if new_cols:
            added_cols = rmp.add_columns(new_cols)
            print(f"[iter {it}] added {added_cols} new columns")

        # 5) 无列无割则停止；否则下一轮会再解 LP
        if not cuts and not new_cols:
            print(f"[iter {it}] no columns and no cuts -> stop.")
            break

    obj_mip, _ = rmp.to_integer_and_solve(mip_gap=params.gap, time_limit=params.mip_time)
    print(f"[MIP] obj={obj_mip:.4f}")
if __name__ == "__main__":
    main()
