"""
TOOL CONTRACT — builder.exposure
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : builder.exposure
domain        : mortality_experience
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Calcule la table d'exposition centrale par âge (E_x, D_x) à partir des
données individuelles du portefeuille. Chaque ligne du DataFrame représente
un contrat avec des dates d'entrée, de sortie et de naissance. Produit la
table fondamentale utilisée par tous les outils du pipeline de construction.

WHEN TO USE
-----------
Appeler en premier dès que le client demande une construction de table de
mortalité d'expérience, un calcul de SMR, ou tout pipeline builder.
Obligatoire avant tout autre outil builder.

WHEN NOT TO USE
---------------
Ne pas appeler pour une analyse descriptive pure (utiliser statistical_analysis).
Ne pas relancer sans modifier age_min/age_max si une erreur de colonnes est retournée.

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []
Note: reçoit df (DataFrame) directement, pas de données du data_store.

INPUTS
------
params:
  age_min:
    type    : int
    values  : 0–120
    default : 20
    note    : Âge minimum du domaine d'analyse. Ajuster selon la population du portefeuille.
  age_max:
    type    : int
    values  : 0–120
    default : 90
    note    : Âge maximum. Réduire si crédibilité faible aux grands âges (voir builder.diagnostics).
  observation_end:
    type    : string
    values  : date au format DD/MM/YYYY
    default : 31/12/2023
    note    : Date de fin d'observation. Les dates de sortie futures sont tronquées à cette date.

OUTPUTS
-------
data_store_keys_written:
  - exposure_table : list[dict] — une entrée par âge avec age, E_x, D_x, mu_x, q_x_brut
  - age_min        : int — âge minimum effectif
  - age_max        : int — âge maximum effectif
  - total_exposure : float — exposition totale en personne-années
  - total_deaths   : int — nombre total de décès
return_payload:
  exposure_table : list[dict] — table principale
  age_min        : int
  age_max        : int
  total_exposure : float
  total_deaths   : int
  lignes_exclues : int — nombre de lignes avec dates non parsables (optionnel)

QUALITY GATES
-------------
BLOCKING:
  - Colonnes requises absentes (date_naissance, date_entree, date_sortie, cause_sortie)
    → retourne {"erreur": "Colonnes requises absentes : [...]"} — demander au client
    le mapping des colonnes avant de relancer.
NON-BLOCKING:
  - lignes_exclues > 0 → documenter le nombre de lignes exclues dans l'analyse et
    appeler statistical_analysis.data_quality pour montrer les exemples au client.

ERROR HANDLING
--------------
error: "Colonnes requises absentes : [...]"
  → cause  : Les colonnes de dates ou cause de sortie ne sont pas détectées dans le CSV.
  → action : Consulter le dictionnaire de données avec le client. Ne jamais relancer
             à l'identique. Demander la correspondance exacte des colonnes.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Avant d'appeler exposure, vérifier que le dictionnaire de données a été
  validé avec le client (Étape 0). L'age_min et age_max doivent être
  déterminés après avoir vu la distribution des âges (statistical_analysis.age_distribution).
  Un age_max trop élevé produira des âges avec exposition nulle.
exemplar_query: >
  Comment choisir age_min et age_max pour un portefeuille prévoyance entreprise ?

CATALOGUE METADATA
------------------
display_name      : Calcul d'exposition (E_x, D_x)
short_description : Calcule la table d'exposition centrale par âge à partir des données individuelles.
domain            : mortality_experience
capability_group  : table_construction
depends_on        : []
required_by       : [builder.crude_rates, builder.diagnostics, builder.validation, builder.benchmarking, build_pdf.certification_report]
client_visible    : true
"""
from __future__ import annotations

import pandas as pd
from agents.mortality.dictionary.column_schema import find_col_by_role
from tools.builder._nb_loader import load_nb


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

    # ── Normalisation de la colonne décès ────────────────────────────────────
    # Le notebook attend exactement "deces". On normalise les valeurs reconnues.
    _DEATH_VALUES = {
        "deces", "décès", "dcd", "d", "dead", "mort", "1", "true", "oui", "yes",
        "decede", "décédé", "deceased", "death",
    }
    df = df.copy()
    raw = df[death_col].astype(str).str.strip().str.lower()
    df[death_col] = raw.where(~raw.isin(_DEATH_VALUES), "deces")

    # ── Nettoyage préventif ───────────────────────────────────────────────────
    # Les dates sentinelles (31/12/2999) provoquent un OverflowError dans pandas.
    # On les remplace par la date de fin d'observation (observation_end).
    # Les dates réellement invalides (0/0/0) → lignes exclues.
    import re as _re
    from datetime import date as _date

    obs_end_str = str(params.get("observation_end", "31/12/2023"))
    obs_end = pd.to_datetime(obs_end_str, dayfirst=True)

    df_clean = df.copy()
    n_before = len(df_clean)

    _SENTINEL_RE = _re.compile(r"2999|9999|3000|0/0/0|00/00/0000|01/01/1900|01/01/1800",
                                _re.IGNORECASE)

    # exit_col : remplacer les sentinelles par obs_end (contrats actifs, censurés à droite)
    mask_sentinel_exit = df_clean[exit_col].astype(str).str.contains(_SENTINEL_RE, na=False)
    df_clean.loc[mask_sentinel_exit, exit_col] = obs_end.strftime("%d/%m/%Y")

    # Autres colonnes de date : exclure les lignes avec valeur non parsable
    for col in (dob_col, entry_col):
        mask_sentinel = df_clean[col].astype(str).str.contains(_SENTINEL_RE, na=False)
        df_clean = df_clean[~mask_sentinel].copy()
        parsed = pd.to_datetime(df_clean[col], dayfirst=True, errors="coerce")
        df_clean = df_clean[parsed.notna()].copy()

    n_dropped = n_before - len(df_clean)

    exposure_table = nb.compute_exposure_by_age(
        df_clean,
        age_min=age_min,
        age_max=age_max,
        dob_col=dob_col,
        entry_col=entry_col,
        exit_col=exit_col,
        death_col=death_col,
    )

    records = exposure_table.where(pd.notnull(exposure_table), None).to_dict(orient="records")

    result = {
        "exposure_table": records,
        "age_min": age_min,
        "age_max": age_max,
        "total_exposure": round(float(exposure_table["E_x"].sum()), 2),
        "total_deaths": int(exposure_table["D_x"].sum()),
    }
    if n_dropped > 0:
        result["lignes_exclues"] = n_dropped
        result["note"] = f"{n_dropped} ligne(s) avec dates non parsables exclues du calcul."
    return result
