"""
report_agent/tools/graphs/builder_plots.py
Graphiques de construction de table de mortalité.
S'appuie sur notebooks/08_visualization.py.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
  Requises (dans data store) :
    data["exposure_table"] : list[dict]  — sortie de builder.exposure
    data["smoothed_table"] : list[dict]  — sortie de builder.smoothing (pour certains charts)

  Paramètres (params dict) :
    chart        : str — graphique à produire :
                         "exposure"       — exposition par âge (E_x, D_x)
                         "crude_smoothed" — taux bruts vs lissés (log scale)
                         "smr"            — SMR par décennie
    sexe         : str — "H" | "F" (défaut : "H") pour référence TH/TF
    title_suffix : str — texte ajouté au titre

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
    chart     : str — type de graphique produit
    image_b64 : str — image PNG encodée en base64
    erreur    : str (si données manquantes ou chart inconnu)
════════════════════════════════════════════════════════════════

Interface : run(data, params) -> dict
"""
from __future__ import annotations

import base64
import pandas as pd
from report_agent.tools.builder._nb_loader import load_nb


def _to_b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}
    chart = params.get("chart", "exposure")
    title_suffix = params.get("title_suffix", "")
    sexe = params.get("sexe", "H")

    nb = load_nb("08_visualization")

    if chart == "exposure":
        exposure_records = data.get("exposure_table")
        if not exposure_records:
            return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
        exposure_table = pd.DataFrame(exposure_records)
        png = nb.plot_exposure_by_age(exposure_table, title_suffix=title_suffix)
        return {"chart": "exposure", "image_b64": _to_b64(png)}

    elif chart == "crude_smoothed":
        exposure_records = data.get("exposure_table")
        if not exposure_records:
            return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}
        exposure_table = pd.DataFrame(exposure_records)

        # Construire smoothed_dict depuis data store
        smoothed_dict = {}
        smoothed_records = data.get("smoothed_table")
        if smoothed_records:
            smoothed_df = pd.DataFrame(smoothed_records)
            qx_col = next((c for c in ("q_x_lisse", "qx") if c in smoothed_df.columns), None)
            method = data.get("smoothing", {}).get("method", "Lissé") if isinstance(data.get("smoothing"), dict) else "Lissé"
            if qx_col:
                smoothed_dict[method] = {
                    "ages": smoothed_df["age"].tolist(),
                    "qx_smoothed": smoothed_df[qx_col].tolist(),
                }

        png = nb.plot_crude_vs_smoothed(
            exposure_table,
            smoothed_dict=smoothed_dict,
            sexe=sexe,
            title_suffix=title_suffix,
        )
        return {"chart": "crude_smoothed", "image_b64": _to_b64(png)}

    elif chart == "smr":
        smr_data = data.get("smr") or data.get("diagnostics", {})
        if isinstance(smr_data, dict) and "smr_by_decade" not in smr_data:
            return {"erreur": "Données SMR manquantes. Appeler builder.diagnostics (function_name=smr) d'abord."}

        # Reconvertir smr_by_decade en DataFrame si c'est une liste
        if isinstance(smr_data.get("smr_by_decade"), list):
            smr_data = dict(smr_data)
            smr_data["smr_by_decade"] = pd.DataFrame(smr_data["smr_by_decade"])

        png = nb.plot_smr_by_age(smr_data, title_suffix=title_suffix)
        return {"chart": "smr", "image_b64": _to_b64(png)}

    else:
        return {"erreur": f"chart inconnu : '{chart}'. Valeurs : exposure, crude_smoothed, smr"}
