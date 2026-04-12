"""
column_schema.py
Source unique des mappings colonnes pour tous les tools statistical_analysis.

Utilisé par :
  - les tools (_find_col) pour détecter les colonnes du CSV
  - writer_agent._build_system_prompt pour informer l'agent des colonnes attendues
"""
from __future__ import annotations
import pandas as pd

# Chaque entrée : role → {label, question, candidates}
# 'candidates' est la liste ordonnée des noms de colonnes acceptés (insensible à la casse)
# 'question'   est la phrase en français pour demander à l'utilisateur si le rôle est absent
COLUMN_SCHEMA: dict[str, dict] = {
    "date_entree": {
        "label": "Date d'entrée en observation",
        "question": "Quelle colonne correspond à la date de début de couverture ou d'entrée en observation ?",
        "candidates": ["date_entree", "ctreffet", "entry_date", "date_d_entree"],
    },
    "date_sortie": {
        "label": "Date de sortie",
        "question": "Quelle colonne correspond à la date de fin de couverture ou de sortie du portefeuille ?",
        "candidates": ["date_sortie", "exit_date", "date_de_sortie"],
    },
    "date_naissance": {
        "label": "Date de naissance",
        "question": "Quelle colonne contient la date de naissance de l'assuré ?",
        "candidates": ["date_naissance", "clinaiss", "dob", "birth_date"],
    },
    "cause_sortie": {
        "label": "Cause de sortie (décès / vivant…)",
        "question": "Quelle colonne indique la cause de sortie ? Je dois y repérer les décès (valeurs attendues : D, deces, 1…).",
        "candidates": ["cause_sortie", "statut", "status", "cause"],
    },
    "sexe": {
        "label": "Sexe de l'assuré",
        "question": "Quelle colonne contient le sexe de l'assuré (H/F, M/F…) ?",
        "candidates": ["sexe", "sexeref", "gender", "sex"],
    },
    "produit": {
        "label": "Produit / type de contrat",
        "question": "Quelle colonne identifie le produit ou le type de contrat ?",
        "candidates": ["cdprod", "produit", "product", "type_contrat"],
    },
    "duree_obs_ans": {
        "label": "Durée d'observation (personne-années)",
        "question": "Quelle colonne contient la durée d'observation déjà calculée en années ? (Si absent, je la calcule depuis les dates d'entrée et de sortie.)",
        "candidates": ["duree_obs_ans", "duree_obs", "exposition", "exposure"],
    },
}


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Retourne la première colonne trouvée parmi les candidats (insensible à la casse)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in cols_lower:
            return cols_lower[c.lower()]
    return None


def find_col_by_role(df: pd.DataFrame, role: str) -> str | None:
    """Retourne la colonne correspondant à un rôle défini dans COLUMN_SCHEMA."""
    entry = COLUMN_SCHEMA.get(role)
    if not entry:
        return None
    return find_col(df, entry["candidates"])


def build_mapping_report(df: pd.DataFrame, capabilities: dict | None = None) -> dict:
    """
    Analyse les colonnes du CSV et retourne un rapport de mapping structuré.

    Retourne :
      matched        : {role: col_name}       — rôles détectés automatiquement
      unmatched      : {role: {label, question}} — rôles absents du CSV
      unknown_cols   : [col_name]              — colonnes CSV non reconnues
      fn_readiness   : {fn_name: {ready, missing_required, missing_optional}}
    """
    csv_cols_lower = {c.lower(): c for c in df.columns}

    matched: dict[str, str] = {}
    unmatched: dict[str, dict] = {}

    for role, info in COLUMN_SCHEMA.items():
        found = next(
            (csv_cols_lower[c.lower()] for c in info["candidates"] if c.lower() in csv_cols_lower),
            None,
        )
        if found:
            matched[role] = found
        else:
            unmatched[role] = {"label": info["label"], "question": info["question"]}

    # Colonnes non reconnues par aucun rôle
    all_recognized = {c.lower() for info in COLUMN_SCHEMA.values() for c in info["candidates"]}
    unknown_cols = [c for c in df.columns if c.lower() not in all_recognized]

    # Disponibilité par fonction (depuis builder_capabilities.json)
    fn_readiness: dict[str, dict] = {}
    if capabilities:
        for tool_name, tool_info in capabilities.get("tools", {}).items():
            for fn_name, fn_info in tool_info.get("functions", {}).items():
                if fn_info.get("disponible", True) is False:
                    continue
                req = fn_info.get("required_columns", [])
                opt = fn_info.get("optional_columns", [])
                missing_req = [r for r in req if r not in matched]
                missing_opt = [r for r in opt if r not in matched]
                fn_readiness[fn_name] = {
                    "ready": len(missing_req) == 0,
                    "missing_required": missing_req,
                    "missing_optional": missing_opt,
                }

    return {
        "matched": matched,
        "unmatched": unmatched,
        "unknown_cols": unknown_cols,
        "fn_readiness": fn_readiness,
    }
