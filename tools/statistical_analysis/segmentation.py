"""
TOOL CONTRACT — statistical_analysis.segmentation
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : statistical_analysis.segmentation
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Calcule la répartition des contrats et décès par variables catégorielles
(sexe, produit, statut de sortie, etc.). S'adapte automatiquement aux
colonnes disponibles. Résultat utilisé par graphs.analysis_plots
(chart=segmentation) et build_pdf.descriptive_report.

WHEN TO USE
-----------
Appeler pour produire des tableaux de répartition par variable catégorielle.
Utile pour identifier les sous-populations à mortalité différente.
Si le client a spécifié des colonnes après validation du dictionnaire de
données, les passer dans params.columns.

WHEN NOT TO USE
---------------
Ne pas appeler si aucune variable catégorielle n'est disponible dans le CSV.

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
  columns:
    type    : list[string]
    values  : noms exacts des colonnes CSV
    default : [sexe, produit, cause_sortie] (détection automatique)
    note    : Si le client a précisé des colonnes à analyser après validation du
              dictionnaire (Étape 0), les spécifier ici. Sinon, la détection auto
              utilise sexe, produit, et cause_sortie.

OUTPUTS
-------
data_store_keys_written:
  - segmentation : dict — résultat complet (total_contrats, total_deces, segmentations)
return_payload:
  total_contrats : int
  total_deces    : int
  segmentations  : dict — {nom_variable: list[{valeur, nb_contrats, nb_deces, pct_contrats, pct_deces}]}

QUALITY GATES
-------------
BLOCKING: []
NON-BLOCKING:
  - avertissement présent → aucune colonne catégorielle trouvée. Informer le client.
    Proposer de spécifier les colonnes manuellement via params.columns.

ERROR HANDLING
--------------
error: [aucun retour erreur structuré — retourne avertissement si aucune colonne trouvée]
  → cause  : Aucune colonne catégorielle détectée.
  → action : Demander au client quelles colonnes segmenter. Relancer avec params.columns.

AGENT GUIDANCE
--------------
reasoning_hint: >
  Si le client a validé des colonnes spécifiques dans le dictionnaire de données
  (sexe = "genre", produit = "cat_produit"), passer ces noms exacts dans
  params.columns. La détection automatique peut ne pas les reconnaître.
  Le résultat "segmentation" est requis pour graphs.analysis_plots.
exemplar_query: >
  Quelles variables de segmentation sont pertinentes pour une analyse de mortalité ?

CATALOGUE METADATA
------------------
display_name      : Segmentation du portefeuille
short_description : Calcule la répartition des contrats et décès par variables catégorielles.
domain            : descriptive
capability_group  : descriptive
depends_on        : []
required_by       : [graphs.analysis_plots, build_pdf.descriptive_report]
client_visible    : true
"""
from __future__ import annotations
import pandas as pd
from agents.mortality.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS


def _is_death(df: pd.DataFrame) -> pd.Series:
    death_col = _find_col(df, _CS["cause_sortie"]["candidates"])
    if death_col:
        col = df[death_col].astype(str).str.lower().str.strip()
        return col.isin(["deces", "décès", "decede", "décédé", "d", "1", "true", "mort", "dead", "dcd"])
    return pd.Series([False] * len(df), index=df.index)


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    """
    Tableau de répartition par variable catégorielle.
    Pour chaque valeur : nb contrats, nb décès, % sur total.
    """
    p = params or {}
    requested = p.get("columns", [])
    # Le LLM peut passer columns sous forme de string JSON (ex: '["sexe","cause_sortie"]')
    # au lieu d'une vraie liste. On parse pour rester tolérant.
    if isinstance(requested, str):
        try:
            import json as _json
            requested = _json.loads(requested)
        except Exception:
            requested = [requested]   # une seule colonne sous forme de string
    if not isinstance(requested, list):
        requested = []

    # Colonnes prioritaires si rien de spécifié
    default_candidates = [
        ("sexe",    _CS["sexe"]["candidates"]),
        ("produit", _CS["produit"]["candidates"]),
        ("statut",  _CS["cause_sortie"]["candidates"]),
    ]

    cols_to_analyze: list[tuple[str, str]] = []
    if requested:
        for c in requested:
            if not isinstance(c, str):
                continue
            found = _find_col(df, [c])
            if found:
                cols_to_analyze.append((c, found))

    # Fallback systématique sur les defaults si rien n'a été trouvé via `requested`
    # (le LLM peut avoir passé un nom de colonne qui n'existe pas dans le CSV).
    if not cols_to_analyze:
        if requested:
            import sys as _sys
            print(
                f"[segmentation] Colonnes demandées {requested} introuvables dans "
                f"le CSV (colonnes disponibles : {list(df.columns)}). "
                f"Fallback sur les defaults (sexe, produit, cause_sortie).",
                file=_sys.stderr,
            )
        for label, candidates in default_candidates:
            found = _find_col(df, candidates)
            if found:
                cols_to_analyze.append((label, found))

    if not cols_to_analyze:
        # Fallback : colonnes avec peu de valeurs distinctes
        for col in df.columns:
            if df[col].nunique() <= 20 and df[col].dtype == object:
                cols_to_analyze.append((col, col))
                if len(cols_to_analyze) >= 4:
                    break

    is_dead = _is_death(df)
    total = len(df)
    total_deces = int(is_dead.sum())

    result: dict = {
        "total_contrats": total,
        "total_deces": total_deces,
        "segmentations": {},
    }

    for label, col_name in cols_to_analyze:
        tab = (
            df.groupby(df[col_name].astype(str).str.strip())
            .agg(
                nb_contrats=(col_name, "count"),
                nb_deces=(col_name, lambda x: int(is_dead.loc[x.index].sum())),
            )
            .reset_index()
            .rename(columns={col_name: "valeur"})
        )
        tab["pct_contrats"] = (tab["nb_contrats"] / total * 100).round(1)
        tab["pct_deces"]    = (tab["nb_deces"] / total_deces * 100).round(1) if total_deces > 0 else 0.0
        result["segmentations"][label] = tab.to_dict(orient="records")

    if not result["segmentations"]:
        result["avertissement"] = "Aucune colonne catégorielle exploitable trouvée."

    return result
