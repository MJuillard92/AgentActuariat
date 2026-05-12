"""
TOOL CONTRACT — builder.crude_rates
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.crude_rates
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Estime les taux bruts de mortalité (q_x) par âge à partir de la table
d'exposition. Deux méthodes disponibles : taux central (μ̂_x = D_x / E_x)
ou binomiale. Produit qx_table consommée par builder.smoothing.

WHEN TO USE
-----------
Appeler immédiatement après builder.exposure dans le pipeline de construction
de table. Prérequis systématique avant tout lissage.

WHEN NOT TO USE
---------------
Ne pas appeler sans exposure_table disponible dans le data_store.
Ne pas utiliser pour un simple résumé descriptif (pas de construction de table).

PREREQUISITES
-------------
required_tools:
  - builder.exposure → provides exposure_table
required_data_store_keys:
  - exposure_table

INPUTS
------
params:
  exposure_table:
    type    : list[dict]
    note    : Table d'exposition par âge produite par builder.exposure
              (lue depuis le data_store, pas passée explicitement).
  method:
    type    : string
    values  : central | binomial | kaplan_meier
    default : central
    note    : >
      "central"      = μ̂_x = D_x/E_x (standard actuariel — central exposure).
      "binomial"     = D_x / (E_x + D_x/2) (correction binomiale pour
                       populations plus petites avec faible exposition).
      "kaplan_meier" = estimateur non-paramétrique de la fonction de survie
                       q_x = 1 - S(x+1)/S(x), calculé depuis le DataFrame
                       individuel cleaned_records. Nécessite que les
                       individual records soient persistés (dataset_ref).

OUTPUTS
-------
data_store_keys_written:
  - qx_table : list[dict] — age, E_x, D_x, qx, method_name par âge
  - method   : str — méthode utilisée
return_payload:
  qx_table : list[dict] — table des taux bruts
  method   : str

QUALITY GATES
-------------
BLOCKING:
  - exposure_table absent → retourne erreur — appeler builder.exposure d'abord.
NON-BLOCKING:
  - Âges avec D_x = 0 ou E_x = 0 → qx = 0 ou indéfini, à noter. Le lissage
    gérera ces points. Documenter si nombreux.

ERROR HANDLING
--------------
error: "exposure_table manquant. Appeler builder.exposure d'abord."
  → cause  : exposure_table absent du data_store.
  → action : Appeler builder.exposure avec les paramètres appropriés, puis relancer.

AGENT GUIDANCE
--------------
reasoning_hint: >
  La méthode "central" est recommandée par défaut. Utiliser "binomial" uniquement
  si le portefeuille est très petit (< 500 contrats) sur recommandation du client.
  Les âges sans décès auront qx = 0 — c'est normal et sera géré par le lissage.
exemplar_query: >
  Quelle méthode d'estimation des taux bruts choisir pour un portefeuille assurance-vie ?

CATALOGUE METADATA
------------------
display_name      : Estimation des taux bruts de mortalité
short_description : Calcule q_x bruts par âge (méthode centrale ou binomiale).
domain            : mortality_experience
capability_group  : table_construction
depends_on        : [builder.exposure]
required_by       : [builder.smoothing]
client_visible    : false
"""
from __future__ import annotations

import pandas as pd
from tools.builder._nb_loader import load_nb


def run(data: dict | None, params: dict | None = None) -> dict:
    data = data or {}
    params = params or {}

    exposure_records = data.get("exposure_table") or data.get("builder.exposure", {}).get("exposure_table")
    if not exposure_records:
        return {"erreur": "exposure_table manquant. Appeler builder.exposure d'abord."}

    exposure_table = pd.DataFrame(exposure_records)
    method = params.get("method", "central")

    nb = load_nb("03_crude_rates")

    if method == "binomial":
        qx_table = nb.crude_rates_binomial(exposure_table)
    elif method == "kaplan_meier":
        # KM nécessite le DataFrame individuel (records nettoyés). Source :
        # cleaned_records (list[dict]) ou df chargé via dataset_ref.
        df_records = data.get("cleaned_records")
        if df_records and isinstance(df_records, list):
            df_indiv = pd.DataFrame(df_records)
        else:
            # Fallback : charger le Parquet normalisé en priorité (colonnes
            # canoniques + dates parsées + sentinelles clippées). Sinon
            # l'original via MemoryManager.
            ref = data.get("_dataset_ref")
            if not ref:
                return {"erreur": "kaplan_meier nécessite cleaned_records "
                                  "(appeler preprocessing.clean_records d'abord) "
                                  "ou un dataset_ref."}
            try:
                from session.dataset_store import DatasetStore
                df_indiv = DatasetStore.load_preferring_normalized(data, ref)
                if df_indiv is None:
                    raise FileNotFoundError(f"aucun dataset chargeable pour session {ref}")
            except Exception as exc:
                return {"erreur": f"impossible de charger le DataFrame pour KM : {exc}"}
        if df_indiv is None or len(df_indiv) == 0:
            return {"erreur": "DataFrame individuel vide ou indisponible pour KM."}
        age_min = int(exposure_table["age"].min()) if "age" in exposure_table else 20
        age_max = int(exposure_table["age"].max()) if "age" in exposure_table else 90
        qx_table = nb.crude_rates_kaplan_meier(
            df_indiv, age_min=age_min, age_max=age_max,
        )
    else:
        qx_table = nb.crude_rates_central(exposure_table)

    records = qx_table.where(pd.notnull(qx_table), None).to_dict(orient="records")

    return {
        "qx_table": records,
        "method": method,
    }
