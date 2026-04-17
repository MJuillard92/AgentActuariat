"""
TOOL CONTRACT — builder.diagnostics
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.diagnostics
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Diagnostics actuariels sur la table d'exposition : crédibilité statistique
par âge, comparaison de méthodes de lissage, ou calcul du SMR préliminaire.
La fonction "credibility" est la plus couramment appelée avant le lissage.

WHEN TO USE
-----------
Appeler après builder.exposure pour évaluer la crédibilité des taux bruts
avant de choisir la méthode et les paramètres de lissage. Indispensable
pour décider du lambda Whittaker ou de la plage d'âges.

WHEN NOT TO USE
---------------
Ne pas appeler compare_smoothers sans smoothers_dict dans le data_store.
Ne pas appeler smr avant d'avoir les taux lissés disponibles.

PREREQUISITES
-------------
required_tools:
  - builder.exposure → provides exposure_table
required_data_store_keys:
  - exposure_table

INPUTS
------
params:
  function_name:
    type    : string
    values  : credibility | compare_smoothers | smr
    default : credibility
    note    : Utiliser "credibility" systématiquement avant le lissage.
  threshold:
    type    : int
    values  : 1–100
    default : 10
    note    : Seuil de crédibilité E_x en personne-années. 10 est standard.
  qx_col:
    type    : string
    values  : q_x_lisse | q_x_brut
    default : q_x_lisse
    note    : Colonne q_x pour le calcul SMR. Utiliser q_x_lisse après lissage.
  sexe:
    type    : string
    values  : H | F
    default : H
    note    : Sexe pour la table de référence SMR.

OUTPUTS
-------
data_store_keys_written:
  - diagnostics.regime              : str   — niveau crédibilité global : high|medium|low (credibility)
  - diagnostics.pct_low_credibility : float — % d'âges sous le seuil de crédibilité (credibility)
  - diagnostics.n_low               : int   — nombre d'âges avec E_x < threshold (credibility)
  - diagnostics.recommendation      : str   — action recommandée selon la crédibilité (credibility)
  - diagnostics.comparison          : list[dict] — AIC, BIC, MSE, n_non_monotone par méthode (compare_smoothers)
  - diagnostics.best_method         : str   — méthode avec le meilleur AIC (compare_smoothers)
  - diagnostics.smr_global          : float — SMR global préliminaire (smr)
  - diagnostics.smr_by_decade       : dict  — SMR par décennie d'âge (smr)
  - diagnostics.interpretation      : str   — interprétation du SMR (smr)
return_payload:
  credibility → regime, pct_low_credibility, n_low, recommendation
  compare_smoothers → comparison (list), best_method
  smr → smr_global, smr_by_decade, interpretation

QUALITY GATES
-------------
BLOCKING:
  - pct_low_credibility > 30% → AVANT de lancer le lissage, l'agent doit soit
    réduire la plage d'âges (relancer builder.exposure avec age_min/age_max
    plus serrés), soit planifier un lambda_wh élevé (≥ 500) pour le lissage.
    Ne pas ignorer ce signal.
NON-BLOCKING:
  - Âges avec exposition nulle (zero_exposure_ages) → signaler au client
    le nombre d'âges sans données.

ERROR HANDLING
--------------
error: "exposure_table manquant. Appeler builder.exposure d'abord."
  → cause  : exposure_table absent du data_store.
  → action : Appeler builder.exposure avant de relancer.
error: "function_name inconnu : '...'"
  → cause  : Valeur de function_name incorrecte.
  → action : Utiliser uniquement : credibility, compare_smoothers, smr.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler systématiquement avec function_name="credibility" après builder.exposure.
  Si pct_low > 30%, adapter age_min/age_max ou choisir lambda_wh ≥ 500.
  Le résultat "recommendation" donne directement la méthode de lissage conseillée.
exemplar_query: >
  Comment interpréter pct_low_credibility = 45% dans un portefeuille retraite ?

CATALOGUE METADATA
------------------
display_name      : Diagnostics de crédibilité
short_description : Évalue la crédibilité statistique des taux bruts par âge.
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.exposure]
required_by       : [builder.smoothing]
client_visible    : true
"""
from __future__ import annotations

import pandas as pd
from tools.builder._nb_loader import load_nb


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
