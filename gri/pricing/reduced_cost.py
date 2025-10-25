# gri/pricing/reduced_cost.py
from __future__ import annotations
from gri.label.forward_ls import (
    find_negative_rc_routes as _find_negative_rc_routes,
    ParamsForPricing,
)

def find_negative_rc_routes(inst, duals, params):
    """
    适配 experiments.run_cg.Params -> ParamsForPricing
    - kmax      <- params.kmax (fallback=5)
    - topk      <- params.cols_topk 或 params.topk (fallback=30)
    - rc_eps    <- params.rc_eps (fallback=-1e-9)
    其它检查开关保持默认 True
    """
    p = ParamsForPricing(
        kmax=getattr(params, "kmax", 5),
        topk=getattr(params, "cols_topk", getattr(params, "topk", 30)),
        rc_eps=getattr(params, "rc_eps", -1e-9),
        check_time=True,
        check_capacity=True,
    )
    return _find_negative_rc_routes(inst, duals, p)

# 向后兼容旧名字
pricing_forward = find_negative_rc_routes
