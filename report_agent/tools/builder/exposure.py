"""
report_agent/tools/builder/exposure.py
Calcul de l'exposition par âge (table centrale E_x, D_x).

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
Colonnes du DataFrame (détectées automatiquement via column_schema) :

  Requises :
    • date_naissance   (rôle : date_naissance) — date de naissance
    • date_entree      (rôle : date_entree)    — date d'entrée en observation
    • date_sortie      (rôle : date_sortie)    — date de sortie
    • cause_sortie     (rôle : cause_sortie)   — indicateur de décès

Paramètres (params dict) :
    age_min : int — âge minimum (défaut : 20)
    age_max : int — âge maximum (défaut : 90)

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
    exposure_table : list[dict]  — une entrée par âge :
                       • age      : int
                       • E_x      : float  — exposition centrale (personne-années)
                       • D_x      : int    — décès observés
                       • mu_x     : float  — taux central (D_x/E_x)
                       • q_x_brut : float  — probabilité brute annuelle
    age_min        : int
    age_max        : int
    total_exposure : float
    total_deaths   : int
    erreur         : str  (si colonnes manquantes)
════════════════════════════════════════════════════════════════

Interface : run(df, params) -> dict
"""
from __future__ import annotations

import pandas as pd
from report_agent.dictionary.column_schema import find_col_by_role
from report_agent.tools.builder._nb_loader import load_nb


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or {}

    dob_col   = find_col_by_role(df, "date_naissance")
    entry_col = find_col_by_role(df, "date_entree")
    exit_col  = find_col_by_role(df, "date_sortie")
    death_col = find_col_by_role(df, "cause_sortie")

    missing = [r for r, c in [
        ("date_naissance", dob_col), ("date_entree", entry_col),
        ("date_sortie", exit_col), ("cause_sortie", death_col),
    ] if c is None]
    if missing:
        return {"erreur": f"Colonnes requises absentes : {missing}"}

    nb = load_nb("02_exposure")
    age_min = int(params.get("age_min", 20))
    age_max = int(params.get("age_max", 90))

    exposure_table = nb.compute_exposure_by_age(
        df,
        age_min=age_min,
        age_max=age_max,
        dob_col=dob_col,
        entry_col=entry_col,
        exit_col=exit_col,
        death_col=death_col,
    )

    records = exposure_table.where(pd.notnull(exposure_table), None).to_dict(orient="records")

    return {
        "exposure_table": records,
        "age_min": age_min,
        "age_max": age_max,
        "total_exposure": round(float(exposure_table["E_x"].sum()), 2),
        "total_deaths": int(exposure_table["D_x"].sum()),
    }
