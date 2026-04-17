"""
TOOL CONTRACT — builder.cox_regression
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.cox_regression
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-07

DESCRIPTION
-----------
Estime le Hazard Ratio (HR) du sexe sur la mortalité via un modèle de Poisson
(équivalent Cox en temps discret). Valide que le ratio H/F est biologiquement
plausible (HR attendu : 1.5–3.0 pour portefeuilles vie/prévoyance).
Nécessite le DataFrame individuel + exposure_table.

WHEN TO USE
-----------
Appeler après builder.exposure pour la section 2 du rapport de certification
(statistiques descriptives). Sert à confirmer que la différentiation H/F est
cohérente avec la littérature actuarielle avant de segmenter les taux.

WHEN NOT TO USE
---------------
Ne pas appeler sur un portefeuille monosexe (résultat non applicable).
Ne pas utiliser pour comparer les taux par âge (utiliser builder.benchmarking).

PREREQUISITES
-------------
required_tools:
  - builder.exposure → provides exposure_table (requis)
required_data_store_keys:
  - exposure_table (requis)
required_df: true  # nécessite le DataFrame individuel (colonnes sexe + statut)

INPUTS
------
params:
  col_sex:
    type    : string
    default : auto-détecté via column_schema
    note    : Colonne identifiant le sexe (ex: SEXEREF). Valeurs H/F ou M/F.
  col_status:
    type    : string
    default : auto-détecté via column_schema
    note    : Colonne statut de sortie. Valeur décès = 'D' ou '1'.
  col_entry:
    type    : string
    default : auto-détecté
    note    : Colonne date d'entrée (pour calculer l'exposition individuelle).
  col_exit:
    type    : string
    default : auto-détecté
    note    : Colonne date de sortie.
  col_birth:
    type    : string
    default : auto-détecté
    note    : Colonne date de naissance.
  death_value:
    type    : string
    default : D
    note    : Valeur qui identifie un décès dans col_status.

OUTPUTS
-------
data_store_keys_written:
  - cox_regression.hazard_ratio  : float — Hazard Ratio H/F (attendu 1.5–3.0)
  - cox_regression.cox_pvalue    : float — p-value du test d'égalité H/F
  - cox_regression.ci_lower_95   : float — borne inférieure IC 95% du HR
  - cox_regression.ci_upper_95   : float — borne supérieure IC 95% du HR
  - cox_regression.interpretation: str   — interprétation textuelle du HR
return_payload:
  hazard_ratio          : float  # HR sexe masculin vs féminin
  ci_lower_95           : float
  ci_upper_95           : float
  cox_pvalue            : float
  deaths_male           : int
  deaths_female         : int
  exposure_male_py      : float
  exposure_female_py    : float
  crude_rate_male       : float
  crude_rate_female     : float
  interpretation        : str

QUALITY GATES
-------------
BLOCKING:
  - exposure_table absent → retourne erreur
  - col_sex absent du DataFrame → retourne erreur
NON-BLOCKING:
  - HR < 1.0 → mortalité féminine supérieure, inhabituel, signaler
  - HR > 4.0 → ratio très élevé, vérifier la qualité des données
  - p_value > 0.05 → différence non significative (petit effectif probable)

ERROR HANDLING
--------------
error: "exposure_table manquant. Appeler builder.exposure d'abord."
error: "Colonne sexe introuvable. Préciser col_sex dans params."
error: "Aucune colonne date trouvée pour calculer l'exposition."

AGENT GUIDANCE
--------------
reasoning_hint: >
  HR attendu pour prévoyance collective : 1.8–2.5 (hommes ~2x plus mortels).
  HR < 1.5 → portefeuille féminin dominant ou erreur de codage.
  HR > 3.0 → sélection forte ou données de mauvaise qualité.
  Toujours vérifier les effectifs (deaths_male, deaths_female) avant d'interpréter.

CATALOGUE METADATA
------------------
display_name      : Régression de Cox — ratio H/F
short_description : Estime le Hazard Ratio du sexe sur la mortalité (Poisson).
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.exposure]
required_by       : [build_pdf.certification_report]
client_visible    : true
"""
from __future__ import annotations

import math
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats


def run(data: dict | None = None, params: dict | None = None, df: "pd.DataFrame | None" = None) -> dict:
    data = data or {}
    params = params or {}

    if not data.get("exposure_table"):
        return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}

    if df is None or df.empty:
        return {"erreur": "DataFrame individuel requis. Uploader le CSV d'abord."}

    # ── Auto-détection des colonnes ───────────────────────────────────────────
    col_sex    = params.get("col_sex")
    col_status = params.get("col_status")
    col_entry  = params.get("col_entry")
    col_exit   = params.get("col_exit")
    col_birth  = params.get("col_birth")
    death_val  = params.get("death_value", "D")

    cols_lower = {c.lower(): c for c in df.columns}

    def _find(candidates: list[str]) -> str | None:
        for cand in candidates:
            if cand in df.columns:
                return cand
            if cand.lower() in cols_lower:
                return cols_lower[cand.lower()]
        return None

    if not col_sex:
        col_sex = _find(["SEXEREF", "SEX", "SEXE", "GENDER", "sexe"])
    if not col_status:
        col_status = _find(["STATUT", "STATUS", "CAUSE_SORTIE", "ETAT"])
    if not col_entry:
        col_entry = _find(["CTREFFET", "DATE_ENTREE", "DEB_OBS", "DEBUT"])
    if not col_exit:
        col_exit = _find(["DATE_SORTIE", "FIN_OBS", "FIN", "DATE_FIN"])
    if not col_birth:
        col_birth = _find(["CLINAISS", "DATE_NAISS", "DATE_NAISSANCE", "NAISS"])

    if not col_sex or col_sex not in df.columns:
        return {"erreur": "Colonne sexe introuvable. Préciser col_sex dans params."}
    if not col_status or col_status not in df.columns:
        return {"erreur": "Colonne statut introuvable. Préciser col_status dans params."}

    # ── Calcul de l'exposition individuelle ───────────────────────────────────
    df_work = df.copy()
    df_work["_is_death"] = df_work[col_status].astype(str).str.upper().str.startswith(death_val.upper()).astype(int)

    # Exposition en personne-années
    if col_entry and col_exit and col_entry in df.columns and col_exit in df.columns:
        try:
            df_work["_entry"] = pd.to_datetime(df_work[col_entry], dayfirst=True, errors="coerce")
            df_work["_exit"]  = pd.to_datetime(df_work[col_exit],  dayfirst=True, errors="coerce")
            df_work["_expo"]  = (df_work["_exit"] - df_work["_entry"]).dt.days / 365.25
            df_work["_expo"]  = df_work["_expo"].clip(lower=0)
        except Exception:
            df_work["_expo"] = 1.0  # fallback : 1 PA par individu
    else:
        df_work["_expo"] = 1.0  # fallback

    # ── Normalisation du sexe ─────────────────────────────────────────────────
    sex_raw = df_work[col_sex].astype(str).str.upper().str.strip()
    is_male = sex_raw.isin(["H", "M", "MALE", "HOMME", "1"])
    df_work["_sex_m"] = is_male.astype(int)

    # ── Statistiques par sexe ─────────────────────────────────────────────────
    male   = df_work[df_work["_sex_m"] == 1]
    female = df_work[df_work["_sex_m"] == 0]

    d_male   = int(male["_is_death"].sum())
    d_female = int(female["_is_death"].sum())
    e_male   = float(male["_expo"].sum())
    e_female = float(female["_expo"].sum())

    if e_male <= 0 or e_female <= 0 or d_male == 0 or d_female == 0:
        return {
            "erreur": "Exposition ou décès nuls pour un groupe sexe. Vérifier la colonne sexe.",
            "deaths_male": d_male, "deaths_female": d_female,
        }

    rate_male   = d_male   / e_male
    rate_female = d_female / e_female

    # ── Hazard Ratio (approximation Poisson) ─────────────────────────────────
    hr = rate_male / rate_female

    # IC 95% sur log(HR) — Wald interval
    se_log_hr = math.sqrt(1.0 / d_male + 1.0 / d_female)
    log_hr = math.log(hr)
    ci_lower = math.exp(log_hr - 1.96 * se_log_hr)
    ci_upper = math.exp(log_hr + 1.96 * se_log_hr)

    # p-value : test chi² sur tableau 2×2 (décès/survie × sexe)
    n_male   = len(male)
    n_female = len(female)
    contingency = np.array([
        [d_male,          d_female],
        [n_male - d_male, n_female - d_female],
    ])
    try:
        chi2, pval, _, _ = stats.chi2_contingency(contingency, correction=False)
    except Exception:
        chi2, pval = float("nan"), float("nan")

    # ── Interprétation ────────────────────────────────────────────────────────
    if hr < 1.0:
        interp = f"HR = {hr:.2f} : mortalité masculine inférieure à la féminine. Inhabituel — vérifier les données."
    elif hr < 1.5:
        interp = f"HR = {hr:.2f} : légère sur-mortalité masculine. Atypique pour prévoyance."
    elif hr <= 3.0:
        interp = f"HR = {hr:.2f} : sur-mortalité masculine cohérente avec la littérature actuarielle (1.5–3.0)."
    else:
        interp = f"HR = {hr:.2f} : ratio élevé. Vérifier la représentativité par sexe dans le portefeuille."

    if not math.isnan(pval) and pval > 0.05:
        interp += " La différence n'est pas statistiquement significative (p > 0.05)."

    result = {
        "hazard_ratio":       round(hr, 4),
        "ci_lower_95":        round(ci_lower, 4),
        "ci_upper_95":        round(ci_upper, 4),
        "cox_pvalue":         round(float(pval), 6) if not math.isnan(pval) else None,
        "deaths_male":        d_male,
        "deaths_female":      d_female,
        "exposure_male_py":   round(e_male, 1),
        "exposure_female_py": round(e_female, 1),
        "crude_rate_male":    round(rate_male * 1000, 4),
        "crude_rate_female":  round(rate_female * 1000, 4),
        "interpretation":     interp,
    }
    data["cox_regression"] = result
    return result
