"""
TOOL CONTRACT — conversation.apply_normalization
════════════════════════════════════════════════════════════════

IDENTITY
--------
name          : conversation.apply_normalization
domain        : conversation
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-05-13

DESCRIPTION
-----------
Déclenche la normalisation complète du fichier de données depuis le chat
(sans passer par le clic UI). Auto-détecte les mappings colonnes/valeurs
si non encore définis, applique le parsing des dates et le clipping des
sentinelles, écrit le Parquet propre sur disque, met à jour data_store.

WHEN TO USE
-----------
L'utilisateur dit "normalise le fichier", "applique le mapping", "passe
sur les noms canoniques", etc. Une fois fait, tous les tools en aval
travaillent automatiquement sur le fichier propre.

WHEN NOT TO USE
---------------
Si records_normalized est déjà True (vérifié en pré-condition).
L'utilisateur peut demander à re-normaliser : on relance proprement.

PREREQUISITES
-------------
required_data_store_keys:
  - dataset_ref (session_id) — pour localiser le Parquet original
Note: si column_mapping / value_mapping absents du data_store, ils sont
auto-détectés (COLUMN_SCHEMA + suggest_value_mapping).

INPUTS
------
params:
  force:
    type    : bool
    default : false
    note    : Si True, relance même si records_normalized=True.

OUTPUTS
-------
return_payload:
  records_normalized      : bool        — True après l'exécution
  dataset_ref_normalized  : str         — chemin du Parquet propre
  observation_end         : str | None  — date ISO de fin d'observation détectée
  column_mapping          : dict        — mapping appliqué {canonical: csv_col}
  value_mapping           : dict        — mapping enum appliqué
  rows_in                 : int         — lignes en entrée
  rows_out                : int         — lignes en sortie

CATALOGUE METADATA
------------------
display_name      : Normalisation du fichier (chat-driven)
short_description : Renomme colonnes, mappe valeurs, parse dates, clip sentinelles.
domain            : conversation
capability_group  : data_exploration
client_visible    : true
"""
from __future__ import annotations

import pandas as pd


def _autodetect_column_mapping(df: pd.DataFrame) -> dict:
    """Détecte le mapping {canonical: csv_col} depuis COLUMN_SCHEMA."""
    try:
        from agents.mortality.dictionary.column_schema import COLUMN_SCHEMA, find_col
    except Exception:
        return {}
    out = {}
    for role, info in COLUMN_SCHEMA.items():
        col = find_col(df, info["candidates"])
        if col:
            out[role] = col
    return out


def _autodetect_value_mapping(df: pd.DataFrame, column_mapping: dict) -> dict:
    """Détecte le mapping enum {col_canonique: {observed: canonical}}.
    Travaille sur le df DÉJÀ renommé (colonnes canoniques)."""
    try:
        from tools.master.suggest_value_mapping import run as _suggest
    except Exception:
        return {}
    enum_specs = {
        "sexe":         ["H", "F"],
        "cause_sortie": ["deces", "autre"],
    }
    cols_present = {k: v for k, v in enum_specs.items() if k in df.columns}
    if not cols_present:
        return {}
    res = _suggest({"records": df, "enum_specs": cols_present}, {})
    return {k: v for k, v in (res.get("value_mapping") or {}).items() if v}


def run(df: pd.DataFrame, params: dict | None = None, data: dict | None = None) -> dict:
    """Déclenche la normalisation complète. `data` est le data_store
    LangGraph — on le mute en place (path + flags) pour persistance."""
    params = params or {}
    if data is None:
        return {"erreur": "data_store non fourni — appel uniquement via Master conversation."}

    if df is None or len(df) == 0:
        return {"erreur": "DataFrame indisponible ou vide."}

    force = bool(params.get("force", False))
    if data.get("records_normalized") and not force:
        return {
            "info": "Fichier déjà normalisé.",
            "records_normalized":     True,
            "dataset_ref_normalized": data.get("dataset_ref_normalized"),
            "observation_end":        data.get("observation_end"),
        }

    # 1) Auto-détecter column_mapping si absent. Format attendu : {canonical: csv_col}
    column_mapping = data.get("column_mapping") or {}
    if not column_mapping:
        column_mapping = _autodetect_column_mapping(df)
    if not column_mapping:
        return {"erreur": "Impossible de détecter le mapping colonnes — "
                          "le fichier ne contient aucune colonne reconnue."}

    # 2) Préparer le df renommé pour auto-détecter value_mapping
    rename_inv = {v: k for k, v in column_mapping.items() if v in df.columns}
    df_renamed = df.rename(columns=rename_inv)

    value_mapping = data.get("value_mapping") or {}
    if not value_mapping:
        value_mapping = _autodetect_value_mapping(df_renamed, column_mapping)

    # 3) Marquer les flags + déléguer à maybe_normalize_records
    data["column_mapping"]           = column_mapping
    data["column_mapping_confirmed"] = True
    data["value_mapping"]            = value_mapping
    data["value_mapping_confirmed"]  = True
    data["_disambiguation_done"]     = True

    dataset_ref = data.get("_dataset_ref") or data.get("dataset_ref")
    try:
        from agents.master.disambiguation import maybe_normalize_records
        df_json = df.to_json(orient="split")
        updates = maybe_normalize_records(data, df_json, dataset_ref=dataset_ref)
    except Exception as exc:
        return {"erreur": f"normalisation échouée : {exc}"}

    if not updates:
        return {"erreur": "Normalisation ignorée (état déjà à jour ou df_json manquant)."}

    # Persister dans data_store (le caller MemoryManager.after_turn() le relira)
    data.update(updates)

    return {
        "records_normalized":     True,
        "dataset_ref_normalized": updates.get("dataset_ref_normalized"),
        "observation_end":        updates.get("observation_end"),
        "column_mapping":         column_mapping,
        "value_mapping":          value_mapping,
        "rows_in":  (updates.get("_audit", {}).get("normalization", {}) or {}).get("rows_in"),
        "rows_out": (updates.get("_audit", {}).get("normalization", {}) or {}).get("rows_out"),
    }
