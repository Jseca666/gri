# gri/data/solomon.py
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import re

@dataclass
class SolomonInstance:
    # 基础规模
    n_customers: int
    N: int  # 样本数

    # 确定性部分
    Q: float
    q: np.ndarray            # (n,)
    e: np.ndarray            # (n,)
    l: np.ndarray            # (n,)
    s: np.ndarray            # (n,)
    tt0: np.ndarray          # (n,)   depot -> i
    tt: np.ndarray           # (n,n)  i -> j
    tt_back: np.ndarray      # (n,)   i -> depot

    # 不确定性样本（零均值化）
    delay0: np.ndarray       # (N,n)          depot->i
    delay: np.ndarray        # (N,n,n)        i->j
    scale: np.ndarray        # (n,)  归一化尺度

def _parse_solomon_txt(path: str) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    解析经典 Solomon 文本：
    VEHICLE
    NUMBER     CAPACITY
      ...
    CUSTOMER
    CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
      0         ...
      1         ...
    返回: (Q, id, xy(2), demand, ready, due, service)
    """
    ids, xs, ys, dem, ready, due, serv = [], [], [], [], [], [], []
    Q = None
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    # 抓容量
    for i, ln in enumerate(lines):
        if re.search(r"CAPACITY", ln, re.IGNORECASE):
            # 下一行可能是数值或同行末尾就有数字
            # 尝试从当前或下一行提取数字
            nums_here = re.findall(r"[-+]?\d+\.?\d*", ln)
            if len(nums_here) >= 1:
                Q = float(nums_here[-1])
                break
            else:
                j = i + 1
                while j < len(lines):
                    nums = re.findall(r"[-+]?\d+\.?\d*", lines[j])
                    if nums:
                        Q = float(nums[-1])
                        break
                    j += 1
            break
    if Q is None:
        # 兼容一些无 VEHICLE 段的变体：默认大容量
        Q = 1e9

    # 找 CUSTOMER 表头
    start = None
    for i, ln in enumerate(lines):
        if re.search(r"CUSTOMER", ln, re.IGNORECASE) and i + 1 < len(lines):
            start = i + 2  # 跳过标题行
            break
    if start is None:
        # 兼容：有的文件没有大写表头，直接找 7 列数字的行
        start = 0

    # 解析行：id x y demand ready due service
    for ln in lines[start:]:
        toks = re.findall(r"[-+]?\d+\.?\d*", ln)
        if len(toks) < 7:
            continue
        i, x, y, d, r, du, sv = toks[:7]
        ids.append(int(float(i)))
        xs.append(float(x))
        ys.append(float(y))
        dem.append(float(d))
        ready.append(float(r))
        due.append(float(du))
        serv.append(float(sv))

    id_arr = np.array(ids, dtype=int)
    xy = np.vstack([np.array(xs), np.array(ys)]).T
    demand = np.array(dem, dtype=np.float64)
    ready = np.array(ready, dtype=np.float64)
    due = np.array(due, dtype=np.float64)
    service = np.array(serv, dtype=np.float64)
    return Q, id_arr, xy, demand, ready, due, service

def _euclid_times(xy: np.ndarray, speed: float = 1.0):
    # 返回: depot->i, i->j, i->depot
    n_all = xy.shape[0]         # 含 depot(0) + customers
    depot_xy = xy[0]
    cust_xy = xy[1:]
    n = n_all - 1

    def dist(a, b):
        return np.hypot(a[:, 0, None] - b[None, :, 0], a[:, 1, None] - b[None, :, 1])

    # depot 到客户
    tt0 = np.hypot(cust_xy[:, 0] - depot_xy[0], cust_xy[:, 1] - depot_xy[1]) / speed
    # 客户到客户
    D = dist(cust_xy, cust_xy) / speed
    np.fill_diagonal(D, 0.0)
    # 客户到 depot
    tt_back = np.hypot(cust_xy[:, 0] - depot_xy[0], cust_xy[:, 1] - depot_xy[1]) / speed
    return tt0.astype(np.float64), D.astype(np.float64), tt_back.astype(np.float64)

def load_solomon_instance(path: str,
                          n_customers: Optional[int] = 25,
                          N: int = 200,
                          speed: float = 1.0,
                          sigma_arc: float = 0.8,
                          sigma_dep: float = 0.8,
                          seed: int = 2025) -> SolomonInstance:
    """
    读取 Solomon 文件并构建用于定价/列生成的实例。
    - n_customers: 取前多少个客户（按原文件顺序，跳过 id=0 的 depot）
    - N: 样本数
    - sigma_*: 对数正态的 sigma；随后做零均值化
    """
    Q, id_arr, xy, demand, ready, due, service = _parse_solomon_txt(path)
    # 以文件顺序截取客户
    # 索引 0 是 depot；1..K 是客户
    if n_customers is None:
        n_customers = len(id_arr) - 1
    take_idx = np.array([0] + list(range(1, 1 + n_customers)), dtype=int)

    xy = xy[take_idx]
    demand = demand[take_idx]
    ready = ready[take_idx]
    due = due[take_idx]
    service = service[take_idx]
    # depot 信息
    depot_ready, depot_due, depot_service = ready[0], due[0], service[0]

    # 构建确定性时间
    tt0, tt, tt_back = _euclid_times(xy, speed=speed)
    n = n_customers

    # 客户属性
    q = demand[1:].astype(np.float64)
    e = ready[1:].astype(np.float64)
    l = due[1:].astype(np.float64)
    s = service[1:].astype(np.float64)
    Q = float(Q)

    # 归一化尺度：按窗口宽度的一部分
    scale = np.maximum(1.0, 0.3 * (l - e + s)).astype(np.float64)

    # 生成零均值扰动（对数正态→逐维减均值）
    rng = np.random.default_rng(seed)
    d0_raw = rng.lognormal(mean=-1.8, sigma=sigma_dep, size=(N, n))
    delay0 = (d0_raw - d0_raw.mean(axis=0, keepdims=True)).astype(np.float64)

    d_raw = rng.lognormal(mean=-1.8, sigma=sigma_arc, size=(N, n, n))
    # 保持对称（可选），或保持非对称；这里取对称更稳
    d_raw = 0.5 * (d_raw + np.transpose(d_raw, (0, 2, 1)))
    for w in range(N):
        np.fill_diagonal(d_raw[w], 0.0)
    delay = (d_raw - d_raw.mean(axis=0, keepdims=True)).astype(np.float64)

    return SolomonInstance(
        n_customers=n, N=N,
        Q=Q, q=q, e=e, l=l, s=s,
        tt0=tt0, tt=tt, tt_back=tt_back,
        delay0=delay0, delay=delay, scale=scale
    )
