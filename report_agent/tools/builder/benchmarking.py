"""
report_agent/tools/builder/benchmarking.py
Comparaison avec tables de référence et calcul des abattements.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
  Requises (dans data store) :
    data["exposure_table"] : list[dict]  — sortie de builder.exposure

  Paramètres (params dict) :
    function_name  : str — "abatement_factors" (défaut) | "load_reference_table"
    reference_name : str — "TH0002" (défaut) | "TF0002" | "TD8890" | "TPRV93"
    sexe           : str — "H" | "F" (défaut : "H")
    qx_exp_col     : str — colonne q_x expérience (défaut : "q_x_lisse")

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
abatement_factors :
    abatement_table : list[dict]  — age, q_x_exp, q_x_ref, abatement_factor
    smr_global      : float
    reference_name  : str

load_reference_table :
    reference_table : list[dict]  — age, qx_ref
    reference_name  : str
════════════════════════════════════════════════════════════════

Interface : run(data, params) -> dict
"""
from __future__ import annotations

import pandas as pd
from report_agent.tools.builder._nb_loader import load_nb


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    fn = params.get("function_name", "abatement_factors")
    nb = load_nb("07_benchmarking")
    reference_name = params.get("reference_name", "TH0002")
    sexe = params.get("sexe", "H")

    try:
        if fn == "load_reference_table":
            ref_df = nb.load_reference_table(name=reference_name, sexe=sexe)
            records = ref_df.where(pd.notnull(ref_df), None).to_dict(orient="records")
            return {"reference_table": records, "reference_name": reference_name}

        elif fn == "abatement_factors":
            exposure_records = data.get("exposure_table")
            if not exposure_records:
                return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
            exposure_table = pd.DataFrame(exposure_records)

            # Fusionner taux lissés si disponibles
            smoothed_records = data.get("smoothed_table")
            if smoothed_records:
                smoothed_df = pd.DataFrame(smoothed_records)
                for col in ("q_x_lisse", "qx"):
                    if col in smoothed_df.columns and col not in exposure_table.columns:
                        exposure_table = exposure_table.merge(smoothed_df[["age", col]], on="age", how="left")
                        break

            result, smr = nb.abatement_factors(
                exposure_table,
                qx_exp_col=params.get("qx_exp_col", None),
                reference_name=reference_name,
                sexe=sexe,
            )
            records = result.where(pd.notnull(result), None).to_dict(orient="records")
            return {
                "abatement_table": records,
                "smr_global": float(smr) if smr is not None else None,
                "reference_name": reference_name,
            }

        else:
            return {"erreur": f"function_name inconnu : '{fn}'. Valeurs : abatement_factors, load_reference_table"}

    except Exception as exc:
        return {"erreur": f"Erreur benchmarking.{fn} : {exc}"}
