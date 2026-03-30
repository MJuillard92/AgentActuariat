"""
report_agent/tools/builder/validation.py
Validation statistique de la table de mortalité lissée.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
  Requises (dans data store) :
    data["exposure_table"] : list[dict]  — sortie de builder.exposure
    data["smoothed_table"] : list[dict]  — sortie de builder.smoothing  (pour chi2)

  Paramètres (params dict) :
    function_name : str   — "confidence_intervals" (défaut) | "chi_square"
    alpha         : float — niveau de risque (défaut : 0.05 → IC 95 %)
    qx_col        : str   — colonne q_x lissée (défaut : "q_x_lisse")
    sexe          : str   — "H" | "F" pour référence chi2 (défaut : "H")

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
confidence_intervals :
    ci_table  : list[dict]  — age, q_x_lisse, ci_lower, ci_upper par âge
    alpha     : float

chi_square :
    chi2_stat     : float
    p_value       : float
    df            : int    — degrés de liberté
    interpretation: str
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
    fn = params.get("function_name", "confidence_intervals")
    nb = load_nb("06_validation")

    # Fusionner les taux lissés dans exposure_table si disponibles
    smoothed_records = data.get("smoothed_table")
    if smoothed_records:
        smoothed_df = pd.DataFrame(smoothed_records)
        for col in ("q_x_lisse", "qx"):
            if col in smoothed_df.columns and col not in exposure_table.columns:
                exposure_table = exposure_table.merge(smoothed_df[["age", col]], on="age", how="left")
                break

    try:
        if fn == "confidence_intervals":
            result_df = nb.confidence_intervals(
                exposure_table,
                qx_col=params.get("qx_col", None),
                alpha=float(params.get("alpha", 0.05)),
            )
            records = result_df.where(pd.notnull(result_df), None).to_dict(orient="records")
            return {"ci_table": records, "alpha": float(params.get("alpha", 0.05))}

        elif fn == "chi_square":
            result = nb.chi_square_test(
                exposure_table,
                qx_col=params.get("qx_col", None),
                sexe=params.get("sexe", "H"),
            )
            # Sérialiser DataFrames si présents
            for k, v in list(result.items()):
                if isinstance(v, pd.DataFrame):
                    result[k] = v.where(pd.notnull(v), None).to_dict(orient="records")
            return result

        else:
            return {"erreur": f"function_name inconnu : '{fn}'. Valeurs : confidence_intervals, chi_square"}

    except Exception as exc:
        return {"erreur": f"Erreur validation.{fn} : {exc}"}
