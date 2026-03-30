"""
report_agent/tools/builder/smoothing.py
Lissage des taux bruts de mortalité.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
  Requises (dans data store) :
    data["qx_table"] : list[dict]  — sortie de builder.crude_rates

  Paramètres (params dict) :
    method      : str   — "whittaker" (défaut) | "gompertz" | "makeham" | "spline"
    lambda_wh   : float — pénalité Whittaker-Henderson (défaut : 100)
    d           : int   — ordre de différence Whittaker (défaut : 2)
    age_min_fit : int   — âge début d'ajustement Gompertz (défaut : 40)
    age_max_fit : int   — âge fin d'ajustement (défaut : 90)

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
    smoothed_table : list[dict]  — une entrée par âge :
                       • age       : int
                       • q_x_brut  : float — taux brut original
                       • q_x_lisse : float — taux lissé
    method         : str
    aic_poisson    : float  (si disponible)
    bic_poisson    : float  (si disponible)
    n_non_monotone : int    — violations de monotonicité (âges ≥ 40)
    erreur         : str    (si qx_table absent ou méthode inconnue)
════════════════════════════════════════════════════════════════

Interface : run(data, params) -> dict
"""
from __future__ import annotations

import pandas as pd
from report_agent.tools.builder._nb_loader import load_nb


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    qx_records = data.get("qx_table")
    if not qx_records:
        return {"erreur": "qx_table manquant. Appeler builder.crude_rates d'abord."}

    qx_table = pd.DataFrame(qx_records)
    method = params.get("method", "whittaker")

    nb = load_nb("04_smoothing")

    try:
        if method == "whittaker":
            result = nb.smooth_whittaker(
                qx_table,
                lambda_wh=float(params.get("lambda_wh", 100)),
                d=int(params.get("d", 2)),
            )
        elif method == "gompertz":
            result = nb.smooth_gompertz(
                qx_table,
                age_min_fit=int(params.get("age_min_fit", 40)),
                age_max_fit=int(params.get("age_max_fit", 90)),
            )
        elif method == "makeham":
            result = nb.smooth_makeham(
                qx_table,
                age_min_fit=int(params.get("age_min_fit", 30)),
                age_max_fit=int(params.get("age_max_fit", 90)),
            )
        elif method == "spline":
            result = nb.smooth_spline(qx_table)
        else:
            return {"erreur": f"Méthode inconnue : '{method}'. Valeurs : whittaker, gompertz, makeham, spline"}
    except Exception as exc:
        return {"erreur": f"Erreur lissage ({method}) : {exc}"}

    # Les smoothers retournent {ages, qx_smoothed, ...} (arrays NumPy)
    # ou {smoothed_table, ...} (DataFrame) selon la méthode.
    import numpy as np

    if "ages" in result and "qx_smoothed" in result:
        # Format arrays → convertir en records [{age, q_x_lisse}, ...]
        ages_arr = np.asarray(result["ages"]).astype(int)
        qx_arr = np.asarray(result["qx_smoothed"]).astype(float)
        records = [
            {"age": int(a), "q_x_lisse": float(q) if not np.isnan(q) else None}
            for a, q in zip(ages_arr, qx_arr)
        ]
        return {
            "smoothed_table": records,
            "method": method,
            "n_non_monotone": result.get("n_non_monotone_after_40"),
        }

    smoothed_df = result.get("smoothed_table") or result.get("result")
    if smoothed_df is None:
        return {"erreur": f"Le smoother '{method}' n'a pas retourné de table lissée."}

    records = smoothed_df.where(pd.notnull(smoothed_df), None).to_dict(orient="records")
    return {
        "smoothed_table": records,
        "method": method,
        "aic_poisson": result.get("aic_poisson"),
        "bic_poisson": result.get("bic_poisson"),
        "n_non_monotone": result.get("n_non_monotone"),
    }
