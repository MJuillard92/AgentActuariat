"""
report_agent/tools/builder/diagnostics.py
Diagnostics actuariels : crédibilité, comparaison de lisseurs, SMR.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
  Requises (dans data store) :
    data["exposure_table"] : list[dict]  — sortie de builder.exposure

  Paramètres (params dict) :
    function_name : str — sous-fonction à appeler :
                          "credibility" (défaut) | "compare_smoothers" | "smr"
    threshold     : int — seuil de crédibilité E_x (défaut : 10)
    qx_col        : str — colonne q_x à utiliser pour SMR (défaut : "q_x_lisse")
    sexe          : str — "H" | "F" pour référence SMR (défaut : "H")

════════════════════════════════════════════════════════════════
OUTPUT  (dict)  — varie selon la fonction
════════════════════════════════════════════════════════════════
credibility :
    regime             : str    — "non-parametric" | "mixed" | "parametric"
    pct_low_credibility: float  — % âges sous le seuil
    recommendation     : str    — méthode de lissage recommandée

compare_smoothers :
    comparison         : list[dict]  — AIC, BIC, MSE par méthode
    best_method        : str

smr :
    smr_global         : float
    smr_by_decade      : list[dict]
    interpretation     : str
════════════════════════════════════════════════════════════════

Interface : run(data, params) -> dict
"""
from __future__ import annotations

import pandas as pd
from report_agent.tools.builder._nb_loader import load_nb


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    exposure_records = data.get("exposure_table")
    if not exposure_records:
        return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}

    exposure_table = pd.DataFrame(exposure_records)
    fn = params.get("function_name", "credibility")
    nb = load_nb("05_diagnostics")

    try:
        if fn == "credibility":
            result = nb.diagnose_credibility(
                exposure_table,
                threshold=int(params.get("threshold", 10)),
            )
        elif fn == "compare_smoothers":
            smoothers_dict = data.get("smoothers_dict", {})
            if not smoothers_dict:
                return {"erreur": "smoothers_dict manquant dans data pour compare_smoothers."}
            comparison_df, best = nb.compare_smoothers(smoothers_dict, exposure_table)
            records = comparison_df.where(pd.notnull(comparison_df), None).to_dict(orient="records")
            result = {"comparison": records, "best_method": best}
        elif fn == "smr":
            qx_col = params.get("qx_col", "q_x_lisse")
            sexe = params.get("sexe", "H")
            result = nb.compute_smr(exposure_table, qx_col=qx_col, sexe=sexe)
        else:
            return {"erreur": f"function_name inconnu : '{fn}'. Valeurs : credibility, compare_smoothers, smr"}
    except Exception as exc:
        return {"erreur": f"Erreur diagnostics.{fn} : {exc}"}

    # Sérialiser DataFrames si présents dans le résultat
    for k, v in list(result.items()):
        if isinstance(v, pd.DataFrame):
            result[k] = v.where(pd.notnull(v), None).to_dict(orient="records")

    return result
