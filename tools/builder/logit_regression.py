"""
TOOL CONTRACT — builder.logit_regression
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.logit_regression
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-07

DESCRIPTION
-----------
Régression des logits : ajuste une droite entre logit(q_x_exp) et logit(q_x_ref)
sur la plage d'âges de l'étude. Un R² > 0.99 et une pente proche de 1 confirment
que la table d'expérience est structurellement cohérente avec la table de référence
(même forme de mortalité, écart proportionnel). Étape de validation réglementaire.

WHEN TO USE
-----------
Appeler après builder.smoothing ET builder.benchmarking pour la section 4.3 du
rapport de certification (positionnement vs tables réglementaires).

WHEN NOT TO USE
---------------
Ne pas appeler si smoothed_table absent (manque de données lissées).
Ne pas appeler sans table de référence configurée (benchmarking requis).
Ne pas interpréter la pente sans regarder R² simultanément.

PREREQUISITES
-------------
required_tools:
  - builder.smoothing    → provides smoothed_table
  - builder.benchmarking → provides reference rates
required_data_store_keys:
  - smoothed_table (requis)
  - benchmarking   (requis — contient les taux de référence par âge)

INPUTS
------
params:
  reference_name:
    type    : string
    values  : TH0002 | TF0002 | TD8890 | TPRV93
    default : TH0002
    note    : Doit correspondre à la table utilisée dans builder.benchmarking.
  sexe:
    type    : string
    values  : H | F
    default : H
  age_min_fit:
    type    : int
    default : 30
    note    : Âge minimum pour la régression (exclure les jeunes âges peu crédibles).
  age_max_fit:
    type    : int
    default : 80
    note    : Âge maximum pour la régression.

OUTPUTS
-------
data_store_keys_written:
  - logit_regression.slope_alpha   : float — pente α (attendu 0.9–1.1)
  - logit_regression.intercept_beta: float — intercept β
  - logit_regression.r_squared     : float — R² (seuil réglementaire > 0.99)
  - logit_regression.formula       : str   — équation : logit(q_exp) = α × logit(q_ref) + β
  - logit_regression.interpretation: str   — interprétation de la cohérence structurelle
return_payload:
  slope_alpha    : float  # pente — attendu ~0.9–1.1
  intercept_beta : float
  r_squared      : float  # attendu > 0.99
  n_ages         : int
  age_min        : int
  age_max        : int
  formula        : str    # logit(q_exp) = α × logit(q_ref) + β
  scatter_data   : list[dict]  # [{age, logit_exp, logit_ref, logit_fitted}]
  interpretation : str

QUALITY GATES
-------------
BLOCKING:
  - smoothed_table absent → erreur
  - benchmarking absent → erreur
NON-BLOCKING:
  - R² < 0.99 → ajustement insuffisant, signaler (seuil réglementaire ACPR)
  - |pente - 1| > 0.15 → structure différente de la référence, expliquer
  - |intercept| > 0.5  → biais systématique fort, commenter

ERROR HANDLING
--------------
error: "smoothed_table manquant. Appeler builder.smoothing d'abord."
error: "benchmarking manquant. Appeler builder.benchmarking d'abord."
error: "Moins de 5 âges valides pour la régression."

AGENT GUIDANCE
--------------
reasoning_hint: >
  La régression logit est un test de forme, pas de niveau.
  Pente α ≈ 1 : la structure de mortalité est similaire à la référence.
  Pente α < 0.9 : la mortalité d'expérience croît moins vite avec l'âge.
  Pente α > 1.1 : elle croît plus vite (portefeuille plus âgé).
  Intercept β capture l'écart de niveau (relié au SMR).
  Toujours citer R² dans le rapport (seuil ACPR implicite : > 0.99).

CATALOGUE METADATA
------------------
display_name      : Régression logit — positionnement vs référence
short_description : Ajuste logit(q_exp) ~ logit(q_ref) pour valider la structure.
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.smoothing, builder.benchmarking]
required_by       : [build_pdf.certification_report]
client_visible    : true
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


def _logit(p: float) -> float | None:
    """Logit avec garde-fou sur les bornes."""
    if p is None or math.isnan(p) or p <= 0 or p >= 1:
        return None
    return math.log(p / (1.0 - p))


def run(data: dict | None = None, params: dict | None = None) -> dict:
    data   = data   or {}
    params = params or {}

    smoothed_records  = data.get("smoothed_table")
    benchmarking_data = data.get("benchmarking")

    if not smoothed_records:
        return {"erreur": "smoothed_table manquant. Appeler builder.smoothing d'abord."}
    if not benchmarking_data:
        return {"erreur": "benchmarking manquant. Appeler builder.benchmarking d'abord."}

    reference_name = params.get("reference_name", "TH0002")
    sexe           = params.get("sexe", "H")
    age_min_fit    = int(params.get("age_min_fit", 30))
    age_max_fit    = int(params.get("age_max_fit", 80))

    # ── Construire les deux séries de taux ────────────────────────────────────
    # Taux d'expérience (lissés)
    smth_df = pd.DataFrame(smoothed_records)
    qx_col  = next((c for c in ("q_x_lisse", "qx") if c in smth_df.columns), None)
    if qx_col is None:
        return {"erreur": "Colonne q_x_lisse introuvable dans smoothed_table."}

    exp_map = {int(r["age"]): float(r[qx_col])
               for _, r in smth_df.iterrows()
               if r.get(qx_col) is not None}

    # Taux de référence — depuis benchmarking.abatement_table
    abatement = benchmarking_data if isinstance(benchmarking_data, list) \
        else benchmarking_data.get("abatement_table", [])

    if not abatement:
        # Fallback : charger la table de référence directement via benchmarking notebook
        try:
            from tools.builder._nb_loader import load_nb
            nb = load_nb("07_benchmarking")
            ref_df = nb.load_reference_table(name=reference_name, sexe=sexe)
            qx_ref_col = next((c for c in ("qx", "q_x") if c in ref_df.columns), ref_df.columns[-1])
            ref_map = {int(r["age"]): float(r[qx_ref_col]) for _, r in ref_df.iterrows()}
        except Exception as exc:
            return {"erreur": f"Impossible de charger la table de référence {reference_name} : {exc}"}
    else:
        # Support multiple key names: q_x_reference, qx_ref, reference_rate
        def _ref_val(r):
            for k in ("q_x_reference", "qx_ref", "reference_rate", "qx_th"):
                if r.get(k) is not None:
                    return float(r[k])
            return None
        ref_map = {int(r["age"]): _ref_val(r)
                   for r in abatement if _ref_val(r) is not None}

    # ── Filtrer la plage d'âges ───────────────────────────────────────────────
    scatter_data = []
    logit_exp_vals = []
    logit_ref_vals = []

    for age in sorted(exp_map.keys()):
        if age < age_min_fit or age > age_max_fit:
            continue
        qx_e = exp_map.get(age)
        qx_r = ref_map.get(age)
        le = _logit(qx_e)
        lr = _logit(qx_r)
        if le is None or lr is None:
            continue
        logit_exp_vals.append(le)
        logit_ref_vals.append(lr)
        scatter_data.append({"age": age, "logit_exp": round(le, 6), "logit_ref": round(lr, 6)})

    if len(logit_ref_vals) < 5:
        return {"erreur": f"Moins de 5 âges valides pour la régression (trouvé {len(logit_ref_vals)})."}

    x = np.array(logit_ref_vals)
    y = np.array(logit_exp_vals)

    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    r_squared = float(r_value ** 2)

    # Ajout des valeurs ajustées dans scatter_data
    for entry in scatter_data:
        entry["logit_fitted"] = round(slope * entry["logit_ref"] + intercept, 6)

    # ── Interprétation ────────────────────────────────────────────────────────
    lines = []
    lines.append(f"Pente α = {slope:.4f}, Intercept β = {intercept:.4f}, R² = {r_squared:.4f}.")

    if r_squared >= 0.99:
        lines.append("Ajustement excellent (R² ≥ 0.99) : la structure de mortalité suit celle de la référence.")
    elif r_squared >= 0.97:
        lines.append("Ajustement satisfaisant (R² ≥ 0.97) mais légèrement en dessous du seuil réglementaire 0.99.")
    else:
        lines.append(f"Ajustement insuffisant (R² = {r_squared:.4f} < 0.99). Vérifier les extrêmes d'âge ou les données.")

    if abs(slope - 1.0) <= 0.10:
        lines.append("Pente proche de 1 : structure de mortalité cohérente avec la référence.")
    elif slope < 0.9:
        lines.append("Pente < 0.9 : mortalité d'expérience croît moins vite avec l'âge que la référence.")
    else:
        lines.append("Pente > 1.1 : mortalité d'expérience croît plus vite avec l'âge que la référence.")

    ages = [e["age"] for e in scatter_data]
    result = {
        "slope_alpha":    round(float(slope), 6),
        "intercept_beta": round(float(intercept), 6),
        "r_squared":      round(r_squared, 6),
        "p_value":        round(float(p_value), 8),
        "std_err":        round(float(std_err), 6),
        "n_ages":         len(scatter_data),
        "age_min":        min(ages) if ages else age_min_fit,
        "age_max":        max(ages) if ages else age_max_fit,
        "reference_name": reference_name,
        "formula":        f"logit(q_exp) = {slope:.4f} × logit(q_ref) + {intercept:.4f}   [R² = {r_squared:.4f}]",
        "scatter_data":   scatter_data,
        "interpretation": " ".join(lines),
    }
    data["logit_regression"] = result
    return result
