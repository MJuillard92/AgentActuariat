"""
TOOL CONTRACT — builder.validation
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.validation
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Validation statistique de la table lissée : intervalles de confiance Poisson
par âge (IC 95%), ou test chi-carré par rapport à une table de référence.
Quantifie la fiabilité statistique des taux lissés.

WHEN TO USE
-----------
Appeler après builder.smoothing (avec smoothed_table disponible). Étape
systématique avant la génération du rapport de certification. Utiliser
"confidence_intervals" par défaut.

WHEN NOT TO USE
---------------
Ne pas appeler sans exposure_table. Ne pas appeler si n_non_monotone > 0
(résoudre d'abord la monotonie avec builder.smoothing).

PREREQUISITES
-------------
required_tools:
  - builder.exposure  → provides exposure_table (requis)
  - builder.smoothing → provides smoothed_table (recommandé fortement)
required_data_store_keys:
  - exposure_table (requis)
  - smoothed_table (optionnel mais fortement recommandé)

INPUTS
------
params:
  function_name:
    type    : string
    values  : confidence_intervals | chi_square
    default : confidence_intervals
    note    : "confidence_intervals" est le standard pour la certification.
              "chi_square" pour tester l'adéquation vs table de référence.
  alpha:
    type    : float
    values  : 0.01 | 0.05 | 0.10
    default : 0.05
    note    : Niveau de risque. 0.05 = IC 95% (standard actuariel français).
  qx_col:
    type    : string
    values  : q_x_lisse | q_x_brut
    default : q_x_lisse
    note    : Colonne des taux lissés à valider.
  sexe:
    type    : string
    values  : H | F
    default : H
    note    : Sexe pour la table de référence (chi_square uniquement).

OUTPUTS
-------
data_store_keys_written:
  - validation.ci_table      : list[dict] — IC par âge : {age, q_x_lisse, ci_lower, ci_upper} (confidence_intervals)
  - validation.alpha         : float — niveau de risque utilisé (confidence_intervals)
  - validation.p_value       : float — p-value du test chi² (chi_square)
  - validation.chi2_stat     : float — statistique chi² observée (chi_square)
  - validation.df            : int   — degrés de liberté (chi_square)
  - validation.interpretation: str   — interprétation textuelle (chi_square)
return_payload:
  confidence_intervals → ci_table (list), alpha
  chi_square → chi2_stat, p_value, df, interpretation

QUALITY GATES
-------------
BLOCKING:
  - exposure_table absent → retourne erreur.
NON-BLOCKING:
  - Pct d'âges dans l'IC < 70% → signaler au client. La table peut manquer
    de crédibilité sur certains âges. Documenter dans le rapport.

ERROR HANDLING
--------------
error: "exposure_table manquant. Appeler builder.exposure d'abord."
  → cause  : exposure_table absent du data_store.
  → action : Appeler builder.exposure puis builder.smoothing avant de relancer.
error: "function_name inconnu : '...'"
  → cause  : Valeur de function_name incorrecte.
  → action : Utiliser uniquement : confidence_intervals, chi_square.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Appeler systématiquement avec function_name="confidence_intervals".
  Si pct dans IC < 70%, noter dans le rapport et expliquer au client
  que certains âges ont peu de données. C'est informatif, pas bloquant.
exemplar_query: >
  Comment interpréter des intervalles de confiance Poisson très larges aux grands âges ?

CATALOGUE METADATA
------------------
display_name      : Validation statistique
short_description : Calcule les intervalles de confiance Poisson et teste l'adéquation statistique.
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.exposure, builder.smoothing]
required_by       : [build_pdf.certification_report]
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
