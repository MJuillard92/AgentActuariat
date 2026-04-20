"""
TOOL CONTRACT — mortality.compute_deaths
════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : mortality.compute_deaths
domain        : mortality
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Compte les décès (cause_sortie == 'deces') dont la date_sortie tombe
dans la période.

INPUTS
------
params:
  records:
    type    : table
  period:
    type    : list[int]

OUTPUTS
-------
return_payload:
  death_count : int
"""
from __future__ import annotations

import pandas as pd


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    period = data["period"]
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)
    if records.empty or not period:
        return {"death_count": 0}

    period_set = set(period)
    years = pd.to_datetime(records["date_sortie"], errors="coerce").dt.year
    mask = (records["cause_sortie"] == "deces") & years.isin(period_set)
    return {"death_count": int(mask.sum())}
