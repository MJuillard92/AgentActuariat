"""
TOOL CONTRACT — mortality.compute_exposure
══════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : mortality.compute_exposure
domain        : mortality
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Exposition centrale cumulée en années-personne sur la période
d'observation. Pour chaque record : années comptées = chevauchement
entre [date_entree, date_sortie] et [min(period), max(period)+1].

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
  cumulative_exposure : number
"""
from __future__ import annotations

import pandas as pd


def _to_ts(v):
    try:
        return pd.to_datetime(v)
    except (ValueError, TypeError):
        return pd.NaT


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    period = data["period"]
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)
    if records.empty or not period:
        return {"cumulative_exposure": 0.0}

    period_start = pd.Timestamp(f"{min(period)}-01-01")
    period_end = pd.Timestamp(f"{max(period) + 1}-01-01")

    total_days = 0.0
    for _, row in records.iterrows():
        entree = _to_ts(row.get("date_entree"))
        sortie = _to_ts(row.get("date_sortie"))
        if pd.isna(entree) or pd.isna(sortie):
            continue
        lo = max(entree, period_start)
        hi = min(sortie, period_end)
        if hi > lo:
            total_days += (hi - lo).days

    return {"cumulative_exposure": float(total_days / 365.25)}
