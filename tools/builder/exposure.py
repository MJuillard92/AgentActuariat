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
  records:
    type    : table
    note    : DataFrame assaini produit par preprocessing.clean_records.
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


def _compute_exposure_for_subset(df: pd.DataFrame, params: dict) -> dict:
    """Calcule la table d'exposition pour un DataFrame donné (sans
    découpage par sexe). Réutilisé pour le calcul unisex (df complet)
    et — si `by_sex=True` — pour chaque sous-groupe H et F séparément.
    """
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

    # ── Nettoyage préventif des dates ─────────────────────────────────────────
    # Chantier dates 2026-05-21 — détection/parsing délégués à
    # tools._shared.date_parsing (règle « année > seuil futur », pas une regex
    # de sentinelles énumérées qui ratait 2040/2050/2060…).
    from datetime import date as _date
    from tools._shared.date_parsing import parse_dates_fr, is_sentinel

    df_clean = df.copy()
    n_before = len(df_clean)

    # Sentinelles de date_sortie (contrats actifs) AVANT parsing : on capture
    # le masque pour les clipper à obs_end (censure à droite). parse_dates_fr
    # les transforme en NaT.
    exit_sentinel = is_sentinel(df_clean[exit_col])

    # Parsing centralisé : dates illisibles, hors-plage pandas et sentinelles
    # → NaT, sans exception.
    df_clean = parse_dates_fr(df_clean, columns=[dob_col, entry_col, exit_col])

    # observation_end : dernière date de sortie parmi les décès observés.
    # Sans ça les contrats actifs seraient comptés en exposition au-delà de
    # la période d'observation réelle de l'assureur.
    obs_end_param = params.get("observation_end")
    if obs_end_param:
        obs_end = pd.to_datetime(str(obs_end_param), dayfirst=True)
    else:
        _is_dead = df_clean[death_col].astype(str).str.lower().str.strip().eq("deces")
        _real_exits = df_clean.loc[_is_dead, exit_col].dropna()
        obs_end = (_real_exits.max() if len(_real_exits) > 0
                   else pd.Timestamp(f"{_date.today().year}-12-31"))

    # exit_col : sentinelles (contrats actifs) → obs_end, censurées à droite.
    if exit_sentinel.any():
        df_clean.loc[exit_sentinel, exit_col] = obs_end

    # dob/entry : exclure les lignes dont la date est non parsable (NaT).
    # (Une sentinelle d'entrée/naissance est aberrante → ligne exclue.)
    for col in (dob_col, entry_col):
        df_clean = df_clean[df_clean[col].notna()].copy()

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

    # cohort_min_age / cohort_max_age : âges effectivement observés (E_x > 0),
    # pas les paramètres demandés. Sans cette correction le rapport annonçait
    # « cohorte 20 à 90 ans » alors que les taux ne sont calculés qu'à partir
    # de l'âge où l'exposition est non nulle (typiquement 40+ pour un
    # portefeuille adulte). Plan qualité-rapport phase 2 (2026-05-24).
    _credible = exposure_table[exposure_table["E_x"] > 0]
    if len(_credible) > 0:
        observed_min = int(_credible["age"].min())
        observed_max = int(_credible["age"].max())
    else:
        observed_min, observed_max = age_min, age_max

    result = {
        "exposure_table": records,
        "age_min":           observed_min,   # = cohort_min_age (mapping YAML)
        "age_max":           observed_max,   # = cohort_max_age (mapping YAML)
        "age_min_requested": age_min,        # paramètre d'entrée (traçabilité)
        "age_max_requested": age_max,        # paramètre d'entrée (traçabilité)
        "total_exposure": round(float(exposure_table["E_x"].sum()), 2),
        "total_deaths": int(exposure_table["D_x"].sum()),
    }
    if n_dropped > 0:
        result["lignes_exclues"] = n_dropped
        result["note"] = f"{n_dropped} ligne(s) avec dates non parsables exclues du calcul."
    return result


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    """Calcule la table d'exposition unisex (toujours) et — si
    `by_sex=True` dans params — produit également les tables ventilées
    `exposure_table_h` et `exposure_table_f`.

    La table unisex reste la sortie canonique (rétro-compat). Les tables
    ventilées sont consommées par builder.crude_rates (qx_table_h/f) puis
    par builder.smoothing pour produire des tables de mortalité par sexe.
    Plan qualité-rapport phase 2 (2026-05-24).
    """
    params = params or {}
    by_sex = bool(params.get("by_sex", False))

    # Toujours calculer l'unisex (sortie canonique requise par toutes les
    # sections existantes : table_construction, smoothing, validation).
    result = _compute_exposure_for_subset(df, params)
    if "erreur" in result or not by_sex:
        return result

    sex_col = find_col_by_role(df, "sexe")
    if not sex_col:
        # by_sex demandé mais pas de colonne sexe → on retourne l'unisex
        # avec un avertissement plutôt que d'échouer.
        result["avertissement_by_sex"] = (
            "by_sex=True ignoré : aucune colonne sexe identifiée."
        )
        return result

    # Sous-paramètres pour les calculs par sexe : désactiver by_sex pour
    # éviter la récursion infinie.
    sub_params = {k: v for k, v in params.items() if k != "by_sex"}
    sex_norm = df[sex_col].astype(str).str.strip().str.upper()
    df_h = df[sex_norm.isin(["H", "M"])].copy()
    df_f = df[sex_norm.isin(["F", "W"])].copy()

    if len(df_h) > 0:
        res_h = _compute_exposure_for_subset(df_h, sub_params)
        if "exposure_table" in res_h:
            result["exposure_table_h"] = res_h["exposure_table"]
            result["total_exposure_h"] = res_h.get("total_exposure")
            result["total_deaths_h"]   = res_h.get("total_deaths")
    if len(df_f) > 0:
        res_f = _compute_exposure_for_subset(df_f, sub_params)
        if "exposure_table" in res_f:
            result["exposure_table_f"] = res_f["exposure_table"]
            result["total_exposure_f"] = res_f.get("total_exposure")
            result["total_deaths_f"]   = res_f.get("total_deaths")
    return result
