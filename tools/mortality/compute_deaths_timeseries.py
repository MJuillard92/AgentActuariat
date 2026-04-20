"""
TOOL CONTRACT — mortality.compute_deaths_timeseries
═══════════════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : mortality.compute_deaths_timeseries
domain        : mortality
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Série temporelle des décès par année calendaire sur la période.
Les années de period sans décès sont présentes avec deaths=0.

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
  series : list[dict]
"""
from __future__ import annotations

import pandas as pd


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    period = data["period"]
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    counts: dict[int, int] = {y: 0 for y in period}
    if not records.empty:
        years = pd.to_datetime(records["date_sortie"], errors="coerce").dt.year
        deces_mask = records["cause_sortie"] == "deces"
        for year, is_deces in zip(years, deces_mask):
            if is_deces and year in counts:
                counts[year] += 1

    series = [{"year": int(y), "deaths": int(counts[y])} for y in sorted(period)]
    return {"series": series}
