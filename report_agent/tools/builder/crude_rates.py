"""
report_agent/tools/builder/crude_rates.py
Estimation des taux bruts de mortalité.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
  Requises (dans data store) :
    data["exposure_table"] : list[dict]  — sortie de builder.exposure

  Paramètres (params dict) :
    method : str — "central" (défaut) | "binomial"

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
    qx_table : list[dict]  — une entrée par âge :
                 • age         : int
                 • E_x         : float
                 • D_x         : int
                 • qx          : float  — probabilité annuelle brute
                 • method_name : str
    method   : str
    erreur   : str  (si exposure_table absent)
════════════════════════════════════════════════════════════════

Interface : run(data, params) -> dict
"""
from __future__ import annotations

import pandas as pd
from report_agent.tools.builder._nb_loader import load_nb


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    exposure_records = data.get("exposure_table") or data.get("builder.exposure", {}).get("exposure_table")
    if not exposure_records:
        return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}

    exposure_table = pd.DataFrame(exposure_records)
    method = params.get("method", "central")

    nb = load_nb("03_crude_rates")

    if method == "binomial":
        qx_table = nb.crude_rates_binomial(exposure_table)
    else:
        qx_table = nb.crude_rates_central(exposure_table)

    records = qx_table.where(pd.notnull(qx_table), None).to_dict(orient="records")

    return {
        "qx_table": records,
        "method": method,
    }
