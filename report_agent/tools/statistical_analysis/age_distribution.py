"""
age_distribution.py
Distribution des âges d'entrée en observation par tranches.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
Colonnes du DataFrame (détectées automatiquement via column_schema) :

  Requises :
    • date_entree      (rôle : date_entree)    — date d'entrée en observation
    • date_naissance   (rôle : date_naissance) — date de naissance → âge calculé à l'entrée

  Optionnelles :
    • sexe             (rôle : sexe)           — H/F/M — requis si by_sex=True

Paramètres (params dict) :
    by_sex     : bool  — ventiler par sexe H/F (défaut : False)
    band_width : int   — largeur des tranches d'âge en années (défaut : 5)

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
Clés toujours présentes :
    age_min        : float  — âge minimum observé
    age_max        : float  — âge maximum observé
    age_median     : float  — âge médian
    age_moyen      : float  — âge moyen
    distribution   : dict   — {tranche (ex "40-44"): nb_contrats}

Clés conditionnelles :
    distribution_h : dict   — distribution hommes  (si by_sex=True et colonne sexe présente)
    distribution_f : dict   — distribution femmes  (si by_sex=True et colonne sexe présente)
    avertissement  : str    — message si colonne sexe absente alors que by_sex=True

En cas d'erreur :
    erreur         : str    — message explicatif (colonnes manquantes ou aucun âge valide)
════════════════════════════════════════════════════════════════

Interface : run(df, params) -> dict
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from report_agent.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS


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

    if by_sex:
        sexe_col = _find_col(df, _CS["sexe"]["candidates"])
        if sexe_col:
            sexe = df[sexe_col].astype(str).str.upper().str.strip()
            mask_h = sexe.isin(["H", "M", "HOMME", "MALE", "1"])
            mask_f = sexe.isin(["F", "FEMME", "FEMALE", "2"])
            if mask_h.any():
                result["distribution_h"] = _dist(ages[mask_h.values])
            if mask_f.any():
                result["distribution_f"] = _dist(ages[mask_f.values])
        else:
            result["avertissement"] = "Colonne sexe non trouvée — distribution globale uniquement."

    return result
