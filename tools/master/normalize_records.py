"""
TOOL CONTRACT — master.normalize_records
════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : master.normalize_records
domain        : master
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Applique column_mapping (renommage de colonnes) et value_mapping
(substitution de valeurs enum) sur un DataFrame. Retourne une copie
conforme au shape attendu par le YAML.

INPUTS
------
params:
  records:
    type    : table
  column_mapping:
    type    : dict
    note    : {old_name: new_name}.
  value_mapping:
    type    : dict
    note    : {column: {observed: canonical}}.

OUTPUTS
-------
return_payload:
  normalized_records : table
"""
from __future__ import annotations

import pandas as pd


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    column_mapping: dict = data.get("column_mapping") or {}
    value_mapping: dict = data.get("value_mapping") or {}

    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    out = records.copy()

    if column_mapping:
        out = out.rename(columns=column_mapping)

    for column, mapping in value_mapping.items():
        if not mapping or column not in out.columns:
            continue
        out[column] = out[column].map(lambda v, m=mapping: m.get(v, v))

    return {"normalized_records": out}
