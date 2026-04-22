"""
TOOL CONTRACT — statistical_analysis.age_distribution
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : statistical_analysis.age_distribution
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Calcule la distribution des âges d'entrée en observation par tranches.
Fournit les statistiques clés (min, max, médiane, moyenne) et une
distribution par tranches d'âge paramétrables. Supporte la ventilation
par sexe H/F. Résultat utilisé par graphs.analysis_plots (chart=age_pyramid).

WHEN TO USE
-----------
Appeler dans toute analyse descriptive pour caractériser la structure d'âge
du portefeuille. Résultat requis avant graphs.analysis_plots (age_pyramid).

WHEN NOT TO USE
---------------
Ne pas appeler si date_naissance ou date_entree sont absentes (retourne erreur).

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []
Note: reçoit df (DataFrame) directement.

INPUTS
------
params:
  records:
    type    : table
    note    : DataFrame assaini produit par preprocessing.clean_records.
  by_sex:
    type    : bool
    values  : true | false
    default : false
    note    : Si true, génère distribution_H et distribution_F en plus de la distribution globale.
              Requiert une colonne sexe dans le DataFrame.
  band_width:
    type    : int
    values  : 1–20
    default : 5
    note    : Largeur des tranches d'âge en années. 5 est standard pour la plupart des analyses.

OUTPUTS
-------
data_store_keys_written:
  - ages : dict — résultat complet (age_min, age_max, age_median, age_moyen, distribution)
return_payload:
  age_min           : float
  age_max           : float
  age_median        : float
  age_moyen         : float
  distribution      : dict — {tranche: nb_contrats}
  distribution_list : list[dict] — [{tranche, nb_contrats}]
  ages              : dict — résultat complet (age_min, age_max, distribution_list, ...)

QUALITY GATES
-------------
BLOCKING:
  - Colonnes date_entree ou date_naissance absentes → retourne erreur.
NON-BLOCKING:
  - avertissement présent (colonne sexe absente alors que by_sex=True) → informer
    le client que la ventilation H/F n'est pas disponible.

ERROR HANDLING
--------------
error: "Colonnes d'âge introuvables. Colonnes disponibles : [...]"
  → cause  : date_entree ou date_naissance absent du DataFrame.
  → action : Vérifier le dictionnaire de données avec le client. Ne pas relancer
             sans les colonnes requises.
error: "Aucun âge valide calculable."
  → cause  : Toutes les dates sont invalides ou manquantes.
  → action : Appeler statistical_analysis.data_quality pour diagnostiquer.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Utiliser by_sex=True uniquement si le client demande une ventilation H/F
  ET si la colonne sexe a été confirmée dans le dictionnaire de données.
  Le résultat "ages" est requis dans le data_store pour graphs.analysis_plots.
exemplar_query: >
  Comment interpréter une distribution d'âges concentrée sur 40-60 ans ?

CATALOGUE METADATA
------------------
display_name      : Distribution des âges
short_description : Calcule la distribution des âges d'entrée par tranches et statistiques clés.
domain            : descriptive
capability_group  : descriptive
depends_on        : []
required_by       : [graphs.analysis_plots, build_pdf.descriptive_report]
client_visible    : true
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from agents.mortality.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS


def _compute_age_at_entry(df: pd.DataFrame) -> pd.Series | None:
    """Calcule l'âge à l'entrée depuis date_entree - date_naissance. Retourne None si colonnes absentes."""
    entry_col = _find_col(df, _CS["date_entree"]["candidates"])
    dob_col   = _find_col(df, _CS["date_naissance"]["candidates"])
    if not entry_col or not dob_col:
        return None
    ent = pd.to_datetime(df[entry_col], format="mixed", dayfirst=True, errors="coerce")
    dob = pd.to_datetime(df[dob_col],   format="mixed", dayfirst=True, errors="coerce")
    return ((ent - dob).dt.days / 365.25).clip(lower=0)


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Distribution des âges d'entrée par tranches.
    Retourne :
      - 'distribution' : dict {tranche_label: nb_contrats}
      - 'distribution_h' / 'distribution_f' si by_sex=True
      - 'age_min', 'age_max', 'age_median'
    """
    p = params or {}
    by_sex    = bool(p.get("by_sex", False))
    band_width = int(p.get("band_width", 5))

    ages = _compute_age_at_entry(df)
    if ages is None:
        return {"erreur": "Colonnes d'âge introuvables. Colonnes disponibles : " + str(list(df.columns))}

    ages = ages.dropna()
    if len(ages) == 0:
        return {"erreur": "Aucun âge valide calculable."}

    age_min = int(ages.min())
    age_max = int(ages.max()) + 1
    bins = list(range((age_min // band_width) * band_width, age_max + band_width, band_width))
    labels = [f"{b}-{b + band_width - 1}" for b in bins[:-1]]

    def _dist(series: pd.Series) -> dict:
        cut = pd.cut(series, bins=bins, labels=labels, right=False)
        return {str(k): int(v) for k, v in cut.value_counts(sort=False).items()}

    result: dict = {
        "age_min":    round(float(ages.min()), 1),
        "age_max":    round(float(ages.max()), 1),
        "age_median": round(float(ages.median()), 1),
        "age_moyen":  round(float(ages.mean()), 1),
        "distribution": _dist(ages),
    }

    result["distribution_list"] = [
        {"tranche": k, "nb_contrats": v} for k, v in result["distribution"].items()
    ]

    if by_sex:
        sexe_col = _find_col(df, _CS["sexe"]["candidates"])
        if sexe_col:
            sexe = df[sexe_col].astype(str).str.upper().str.strip()
            mask_h = sexe.isin(["H", "M", "HOMME", "MALE", "1"])
            mask_f = sexe.isin(["F", "FEMME", "FEMALE", "2"])
            if mask_h.any():
                result["distribution_h"] = _dist(ages[mask_h.values])
                result["distribution_list_h"] = [
                    {"tranche": k, "nb_contrats": v} for k, v in result["distribution_h"].items()
                ]
            if mask_f.any():
                result["distribution_f"] = _dist(ages[mask_f.values])
                result["distribution_list_f"] = [
                    {"tranche": k, "nb_contrats": v} for k, v in result["distribution_f"].items()
                ]
        else:
            result["avertissement"] = "Colonne sexe non trouvée — distribution globale uniquement."

    return result
