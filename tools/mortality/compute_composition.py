"""
TOOL CONTRACT — mortality.compute_composition
═════════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : mortality.compute_composition
domain        : mortality
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Composition agrégée du portefeuille par dimension(s) de group_by.
Colonnes retournées : group_by + n_lives, exposure, deaths.

INPUTS
------
params:
  records:
    type    : table
  group_by:
    type    : list[string]

OUTPUTS
-------
return_payload:
  composition_table : table
"""
from __future__ import annotations

import pandas as pd

from tools.mortality.compute_exposure import run as run_exposure


def _exposure_for_group(sub_df: pd.DataFrame) -> float:
    if sub_df.empty:
        return 0.0
    entree = pd.to_datetime(sub_df["date_entree"], errors="coerce")
    sortie = pd.to_datetime(sub_df["date_sortie"], errors="coerce")
    days = (sortie - entree).dt.days.clip(lower=0).fillna(0)
    return float(days.sum() / 365.25)


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    group_by = data.get("group_by") or []
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    if not group_by or records.empty:
        return {"composition_table": pd.DataFrame(columns=[*group_by, "n_lives", "exposure", "deaths"])}

    rows = []
    for keys, sub in records.groupby(group_by, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_by, keys))
        row["n_lives"] = int(len(sub))
        row["exposure"] = _exposure_for_group(sub)
        row["deaths"] = int((sub["cause_sortie"] == "deces").sum())
        rows.append(row)

    return {"composition_table": pd.DataFrame(rows)}
