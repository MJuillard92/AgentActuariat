"""
TOOL CONTRACT — builder.benchmarking
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.benchmarking
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Compare la table d'expérience construite avec une table de référence réglementaire
(TH0002, TF0002, TD88-90, TPRV93). Calcule les facteurs d'abattement par âge et
le SMR global (Standardized Mortality Ratio). Étape finale du pipeline builder.

WHEN TO USE
-----------
Appeler après builder.smoothing (avec smoothed_table fortement recommandée) et
builder.exposure (exposure_table requise). Utiliser "abatement_factors" pour
la certification standard. Appeler "load_reference_table" pour inspecter une
table de référence sans calcul.

WHEN NOT TO USE
---------------
Ne pas appeler sans exposure_table. Ne pas comparer avec TF0002 pour un
portefeuille hommes (utiliser sexe="H" et TH0002).

PREREQUISITES
-------------
required_tools:
  - builder.exposure  → provides exposure_table (requis)
  - builder.smoothing → provides smoothed_table (recommandé — utilisé pour q_x_exp)
required_data_store_keys:
  - exposure_table (requis)
  - smoothed_table (optionnel mais recommandé)

INPUTS
------
params:
  function_name:
    type    : string
    values  : abatement_factors | load_reference_table
    default : abatement_factors
    note    : "abatement_factors" pour la certification. "load_reference_table"
              pour inspecter la table de référence.
  reference_name:
    type    : string
    values  : TH0002 | TF0002 | TD8890 | TPRV93
    default : TH0002
    note    : TH0002 pour hommes (table réglementaire française), TF0002 pour femmes.
              TD8890 et TPRV93 disponibles pour études historiques.
  sexe:
    type    : string
    values  : H | F
    default : H
    note    : Doit correspondre au portefeuille analysé. Demander au client si mixte.
  qx_exp_col:
    type    : string
    values  : q_x_lisse | q_x_brut
    default : q_x_lisse
    note    : Colonne q_x expérience. Utiliser q_x_lisse si smoothed_table disponible.

OUTPUTS
-------
data_store_keys_written:
  - benchmarking.abatement_table: list[dict] — {age, q_x_brut, q_x_reference, abattement} (abatement_factors)
  - benchmarking.smr_global     : float — SMR global = Σ(D_x observés) / Σ(D_x attendus) (abatement_factors)
  - benchmarking.reference_name : str   — nom de la table utilisée ex: TH0002 (abatement_factors)
  - benchmarking.summary        : dict  — résumé avec smr_global, n_ages, reference_name (abatement_factors)
  - benchmarking.reference_table: list[dict] — table de référence brute {age, qx} (load_reference_table)
return_payload:
  abatement_factors → abatement_table (list), smr_global (float), reference_name, summary
  load_reference_table → reference_table (list), reference_name

QUALITY GATES
-------------
BLOCKING:
  - exposure_table absent → retourne erreur.
NON-BLOCKING:
  - SMR global < 0.5 → forte sélection (portefeuille assurés ayant passé
    des formalités médicales). Mentionner dans le rapport.
  - SMR 0.5–0.8 → sélection modérée. Normal pour prévoyance entreprise.
  - SMR 0.8–1.2 → mortalité proche de la référence. Documenter.
  - SMR > 1.2 → sur-mortalité. Alerter le client et recommander prudence.

ERROR HANDLING
--------------
error: "exposure_table manquant. Appeler builder.exposure d'abord."
  → cause  : exposure_table absent du data_store.
  → action : Appeler builder.exposure (et idéalement builder.smoothing) avant.
error: "function_name inconnu : '...'"
  → cause  : Valeur incorrecte.
  → action : Utiliser uniquement : abatement_factors, load_reference_table.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Toujours vérifier que sexe correspond au portefeuille avant d'appeler.
  Pour un portefeuille mixte, appeler deux fois (H puis F) avec les sous-groupes
  filtrés, ou signaler au client que le SMR global est une approximation.
  Interpréter le SMR en contexte : < 0.5 = fort effet sélection, 0.5-0.8 =
  sélection modérée, 0.8-1.2 = proche référence, > 1.2 = sur-mortalité.
exemplar_query: >
  SMR = 0.65 pour un portefeuille prévoyance entreprise : est-ce normal ?

CATALOGUE METADATA
------------------
display_name      : Benchmarking vs table de référence
short_description : Calcule facteurs d'abattement et SMR par rapport à une table réglementaire.
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

            result, summary = nb.abatement_factors(
                exposure_table,
                qx_exp_col=params.get("qx_exp_col", None),
                reference_name=reference_name,
                sexe=sexe,
            )
            records = result.where(pd.notnull(result), None).to_dict(orient="records")
            # summary peut être un dict ou un float selon la version du notebook
            if isinstance(summary, dict):
                smr_global = (summary.get("SMR_global")
                              or summary.get("smr_global")
                              or summary.get("global_factor"))
            else:
                try:
                    smr_global = float(summary)
                except (TypeError, ValueError):
                    smr_global = None
            return {
                "abatement_table": records,
                "smr_global": float(smr_global) if smr_global is not None else None,
                "reference_name": reference_name,
                "summary": summary if isinstance(summary, dict) else {},
            }

        else:
            return {"erreur": f"function_name inconnu : '{fn}'. Valeurs : abatement_factors, load_reference_table"}

    except Exception as exc:
        return {"erreur": f"Erreur benchmarking.{fn} : {exc}"}
