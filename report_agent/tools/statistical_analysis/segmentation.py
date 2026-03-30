"""
segmentation.py
Répartition des contrats et décès par variables catégorielles.

════════════════════════════════════════════════════════════════
INPUTS
════════════════════════════════════════════════════════════════
Colonnes du DataFrame (détectées automatiquement via column_schema) :

  Requises : aucune — la fonction s'adapte aux colonnes disponibles

  Par défaut (si params.columns absent) :
    • sexe          (rôle : sexe)         — répartition H/F
    • produit       (rôle : produit)      — répartition par produit
    • cause_sortie  (rôle : cause_sortie) — répartition par statut de sortie

  Fallback : si aucune colonne reconnue, utilise toutes les colonnes
             catégorielles avec ≤ 20 valeurs distinctes (max 4)

Paramètres (params dict) :
    columns : list[str]  — noms exacts des colonnes CSV à analyser
                           (prioritaire sur la détection automatique)

════════════════════════════════════════════════════════════════
OUTPUT  (dict)
════════════════════════════════════════════════════════════════
Clés toujours présentes :
    total_contrats  : int   — nombre total de lignes
    total_deces     : int   — nombre total de décès détectés
    segmentations   : dict  — {nom_variable: list[dict]} où chaque dict contient :
                                • valeur        : str   — modalité de la variable
                                • nb_contrats   : int
                                • nb_deces      : int
                                • pct_contrats  : float — % sur total_contrats
                                • pct_deces     : float — % sur total_deces

Clés conditionnelles :
    avertissement   : str   — si aucune colonne catégorielle exploitable trouvée
════════════════════════════════════════════════════════════════

Interface : run(df, params) -> dict
"""
from __future__ import annotations
import pandas as pd
from report_agent.dictionary.column_schema import find_col as _find_col, COLUMN_SCHEMA as _CS


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

    # Colonnes prioritaires si rien de spécifié
    default_candidates = [
        ("sexe",    _CS["sexe"]["candidates"]),
        ("produit", _CS["produit"]["candidates"]),
        ("statut",  _CS["cause_sortie"]["candidates"]),
    ]

    if requested:
        cols_to_analyze = []
        for c in requested:
            found = _find_col(df, [c])
            if found:
                cols_to_analyze.append((c, found))
    else:
        cols_to_analyze = []
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
