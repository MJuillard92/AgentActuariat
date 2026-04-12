"""
TOOL CONTRACT — statistical_analysis.data_quality
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : statistical_analysis.data_quality
domain        : descriptive
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-03-31

DESCRIPTION
-----------
Inspecte la qualité des données du portefeuille : détecte et montre les
lignes problématiques (dates invalides, dates sentinelles, valeurs manquantes).
Retourne un tableau de lignes en erreur avec leurs valeurs brutes pour
affichage direct dans le chat client.

WHEN TO USE
-----------
Appeler IMMÉDIATEMENT si un tool retourne une erreur liée aux données
(dates invalides, colonnes manquantes, format incorrect). Règle absolue :
ne jamais dire "je ne peux pas afficher les lignes" sans avoir appelé
data_quality d'abord. Appeler aussi proactivement après portfolio_summary
si qualite_donnees contient des anomalies.

WHEN NOT TO USE
---------------
Ne pas appeler de manière répétée si aucune erreur de données n'est détectée.
Ne pas appeler si l'erreur est liée à un paramètre (ex: function_name incorrect).

PREREQUISITES
-------------
required_tools: []
required_data_store_keys: []
Note: reçoit df (DataFrame) directement.

INPUTS
------
params:
  focus:
    type    : string
    values  : dates | all
    default : dates
    note    : "dates" inspecte les colonnes de dates uniquement. "all" inclut
              aussi cause_sortie et sexe.
  max_rows:
    type    : int
    values  : 1–50
    default : 8
    note    : Nombre maximum de lignes problématiques à retourner dans "table".
  column:
    type    : string
    values  : nom de colonne CSV
    default : null
    note    : Si spécifié, inspecter uniquement cette colonne.

OUTPUTS
-------
data_store_keys_written: []
return_payload:
  issues         : list[dict] — {colonne, role, type, count, pct, exemples_valeurs}
  total_rows     : int
  total_issues   : int — nb lignes affectées
  table          : list[dict] — lignes problématiques avec valeurs brutes
  columns_header : list[str] — en-têtes du tableau
  summary        : str — résumé en une phrase ("X lignes avec dates invalides sur Y (Z%)")

QUALITY GATES
-------------
BLOCKING: []
NON-BLOCKING:
  - total_issues > 0 → afficher le tableau retourné directement dans le chat.
    Donner le compte exact et proposer des actions concrètes (supprimer, corriger).

ERROR HANDLING
--------------
error: [aucun retour erreur — retourne résultats vides si aucune anomalie]
  → cause  : N/A
  → action : N/A

AGENT GUIDANCE
--------------
reasoning_hint: >
  RÈGLE ABSOLUE : si un tool retourne une erreur liée aux données, appeler
  data_quality IMMÉDIATEMENT. Afficher le tableau "table" directement dans
  la réponse au client. Ne jamais estimer ou supposer le problème sans
  données concrètes. Exemple de réponse attendue :
  "J'ai détecté 12 lignes avec des dates invalides sur 45 231 (0.03%).
  Voici les exemples : [tableau]. Options : supprimer ces 12 lignes, ou
  les corriger manuellement."
exemplar_query: >
  Comment afficher les lignes avec des dates invalides dans un portefeuille assurance ?

CATALOGUE METADATA
------------------
display_name      : Inspection qualité des données
short_description : Détecte et affiche les lignes problématiques (dates invalides, valeurs manquantes).
domain            : descriptive
capability_group  : descriptive
depends_on        : []
required_by       : []
client_visible    : true
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from agents.mortality.dictionary.column_schema import find_col_by_role, COLUMN_SCHEMA


def _try_parse_date(series: pd.Series) -> pd.Series:
    """Tente de parser une série en dates ; retourne un masque des valeurs invalides."""
    try:
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")
        return parsed.isna() & series.notna() & (series.astype(str).str.strip() != "")
    except Exception:
        return pd.Series([False] * len(series), index=series.index)


def _is_sentinel(series: pd.Series) -> pd.Series:
    """Détecte les dates 'sentinelle' irréalistes (ex: 31/12/2999, 0/0/0)."""
    s = series.astype(str).str.strip()
    bad = (
        s.str.contains(r"2999|9999|0/0/0|00/00/0000|01/01/1900|01/01/1800", regex=True, na=False)
        | s.str.contains(r"^0+[/\-]0+[/\-]0+$", regex=True, na=False)
    )
    return bad


def run(df: pd.DataFrame, params: dict | None = None) -> dict:
    params   = params or {}
    focus    = params.get("focus", "dates")
    max_rows = int(params.get("max_rows", 8))
    target_col = params.get("column", None)  # colonne spécifique

    date_roles = ["date_entree", "date_sortie", "date_naissance"]
    issues = []
    bad_row_indices = set()

    # ── Inspection des colonnes de dates ──────────────────────────────────────
    for role in date_roles:
        col = find_col_by_role(df, role)
        if col is None:
            continue
        if target_col and col != target_col:
            continue

        serie = df[col].astype(str).str.strip()

        # Valeurs non parsables (hors sentinelles connues)
        unparsable = _try_parse_date(df[col])
        sentinel   = _is_sentinel(df[col])
        truly_invalid = unparsable & ~sentinel   # mauvais format, pas une sentinelle
        if truly_invalid.any():
            bad_idx = df.index[truly_invalid].tolist()
            bad_row_indices.update(bad_idx[:max_rows])
            sample_vals = sorted(set(serie[truly_invalid].tolist()))[:6]
            issues.append({
                "colonne":          col,
                "role":             role,
                "type":             "date_invalide",
                "count":            int(truly_invalid.sum()),
                "pct":              round(truly_invalid.sum() / len(df) * 100, 2),
                "exemples_valeurs": sample_vals,
            })

        # Valeurs sentinelles (ex: 31/12/2999)
        sentinel = _is_sentinel(df[col])
        if sentinel.any():
            sample_vals = sorted(set(serie[sentinel].tolist()))[:4]
            issues.append({
                "colonne":          col,
                "role":             role,
                "type":             "date_sentinelle",
                "count":            int(sentinel.sum()),
                "pct":              round(sentinel.sum() / len(df) * 100, 2),
                "exemples_valeurs": sample_vals,
                "note":             "Date technique (2999, 9999...) = contrat actif sans fin connue.",
            })

        # Valeurs manquantes
        nulls = df[col].isna() | (serie == "") | (serie == "nan")
        if nulls.any():
            issues.append({
                "colonne":  col,
                "role":     role,
                "type":     "valeur_manquante",
                "count":    int(nulls.sum()),
                "pct":      round(nulls.sum() / len(df) * 100, 2),
                "exemples_valeurs": [],
            })

    # ── Inspection des colonnes catégorielles si focus=="all" ─────────────────
    if focus == "all":
        for role in ("cause_sortie", "sexe"):
            col = find_col_by_role(df, role)
            if col is None:
                continue
            nulls = df[col].isna()
            if nulls.any():
                issues.append({
                    "colonne":          col,
                    "role":             role,
                    "type":             "valeur_manquante",
                    "count":            int(nulls.sum()),
                    "pct":              round(nulls.sum() / len(df) * 100, 2),
                    "exemples_valeurs": [],
                })

    # ── Construction du tableau d'exemples ───────────────────────────────────
    # Colonnes à afficher : date columns + cause_sortie
    display_cols = []
    for role in date_roles + ["cause_sortie", "sexe"]:
        c = find_col_by_role(df, role)
        if c and c in df.columns:
            display_cols.append(c)

    # Ajouter une colonne d'index lisible
    if not display_cols:
        display_cols = list(df.columns[:5])

    # Sélectionner les lignes problématiques (non-sentinelles)
    unparsable_issues = [i for i in issues if i["type"] == "date_invalide"]
    table_rows = []
    if unparsable_issues:
        for iss in unparsable_issues[:2]:  # Max 2 colonnes en erreur
            col = iss["colonne"]
            bad_mask = _try_parse_date(df[col])
            sample_df = df[bad_mask].head(max_rows)
            show_cols = [c for c in display_cols if c in sample_df.columns]
            for row_n, (idx, row) in enumerate(sample_df[show_cols].iterrows(), 1):
                entry = {"ligne": int(idx) + 2}  # +2 : entête + base-0
                for c in show_cols:
                    entry[c] = str(row[c]) if pd.notna(row[c]) else "(vide)"
                entry["_problème"] = f"'{col}' invalide"
                table_rows.append(entry)

    # Déduplication (même ligne peut apparaître pour 2 colonnes)
    seen = set()
    unique_rows = []
    for r in table_rows:
        key = r["ligne"]
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    columns_header = (
        ["ligne"] + [c for c in display_cols if c in df.columns] + ["_problème"]
        if unique_rows else []
    )

    # ── Résumé ────────────────────────────────────────────────────────────────
    real_issues = [i for i in issues if i["type"] == "date_invalide"]
    total_bad   = sum(i["count"] for i in real_issues)
    if total_bad == 0:
        summary = "Aucune date invalide détectée dans le portefeuille."
    else:
        cols_str = ", ".join(f"'{i['colonne']}'" for i in real_issues)
        summary = (
            f"{total_bad} ligne(s) avec des dates invalides ({cols_str}). "
            f"Voir le tableau ci-dessous pour les exemples."
        )

    return {
        "issues":         issues,
        "total_rows":     len(df),
        "total_issues":   total_bad,
        "table":          unique_rows,
        "columns_header": columns_header,
        "summary":        summary,
    }
