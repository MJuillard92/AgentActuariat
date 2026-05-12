"""
TOOL CONTRACT — master.suggest_value_mapping
════════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : master.suggest_value_mapping
domain        : master
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Pour chaque colonne enum déclarée dans enum_specs, compare les valeurs
observées dans le DataFrame aux valeurs autorisées. Propose un mapping
via heuristique (normalisation + synonymes). Les valeurs sans
correspondance évidente sont remontées dans 'unmapped' pour arbitrage
utilisateur.

INPUTS
------
params:
  records:
    type    : table
  enum_specs:
    type    : dict
    note    : {column_name: [allowed_values]}

OUTPUTS
-------
return_payload:
  value_mapping : dict
  unmapped      : dict
"""
from __future__ import annotations

import unicodedata

import pandas as pd


_SYNONYMS: dict[str, dict[str, str]] = {
    "deces": {"deces", "décès", "decede", "décédé", "decedee", "décédée",
              "mort", "dead", "d", "1", "true"},
    "autre": {"autre", "vivant", "vivante", "alive", "sortie", "en cours",
              "encours", "actif", "active", "0", "false"},
    # Convention actuarielle française INSEE : 1=Homme, 2=Femme.
    # Si la convention diffère pour ce portefeuille, l'utilisateur doit
    # corriger via le mapping UI.
    "H": {"h", "m", "homme", "male", "masculin", "1"},
    "F": {"f", "w", "femme", "female", "féminin", "feminin", "2"},
}


def _normalize(value) -> str:
    s = str(value).strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s


def _match_canonical(observed: str, allowed: list[str]) -> str | None:
    norm = _normalize(observed)
    for canonical in allowed:
        if _normalize(canonical) == norm:
            return canonical
        synonyms = _SYNONYMS.get(canonical, set())
        if norm in synonyms:
            return canonical
    return None


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    enum_specs = data["enum_specs"]
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    value_mapping: dict[str, dict[str, str]] = {}
    unmapped: dict[str, list] = {}

    for column, allowed in enum_specs.items():
        if column not in records.columns:
            continue
        col_map: dict[str, str] = {}
        col_unmapped: list = []
        observed_unique = records[column].dropna().unique()
        for observed in observed_unique:
            if observed in allowed:
                continue
            canonical = _match_canonical(observed, allowed)
            if canonical is not None:
                col_map[observed] = canonical
            else:
                col_unmapped.append(observed)
        value_mapping[column] = col_map
        unmapped[column] = col_unmapped

    return {"value_mapping": value_mapping, "unmapped": unmapped}
