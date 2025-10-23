from typing import List, Dict
from gri.label.forward_ls import pricing_forward
from gri.master.duals import DualPack

def find_negative_rc_routes(instance, duals: DualPack, params) -> List[Dict]:
    return pricing_forward(instance, duals, params)
