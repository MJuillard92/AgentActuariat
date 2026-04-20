"""
TOOL CONTRACT — master.analyze_data_and_request
═══════════════════════════════════════════════

CATALOGUE METADATA
------------------
name          : master.analyze_data_and_request
domain        : master
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Inférences factuelles sur la période d'observation à partir d'un
DataFrame de records normalisés. Un seul appel retourne les 4
grandeurs consommées par le preamble via output_mapping.

WHEN TO USE
-----------
- Phase master_from_data, après normalize_records.

INPUTS
------
params:
  records:
    type    : table
    note    : DataFrame normalisé avec colonnes date_sortie, cause_sortie.

OUTPUTS
-------
return_payload:
  period_years      : list[int]
  first_death_year  : int
  last_death_year   : int
  n_years           : int
"""
from __future__ import annotations

import pandas as pd


def _extract_year(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "year"):
        return int(value.year)
    try:
        return int(pd.to_datetime(value).year)
    except (ValueError, TypeError):
        return None


def run(data: dict, params: dict) -> dict:
    records = data["records"]
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    deces = records[records["cause_sortie"] == "deces"]
    if deces.empty:
        return {
            "period_years": [],
            "first_death_year": None,
            "last_death_year": None,
            "n_years": None,
        }

    years = [y for y in (_extract_year(v) for v in deces["date_sortie"]) if y is not None]
    if not years:
        return {
            "period_years": [],
            "first_death_year": None,
            "last_death_year": None,
            "n_years": None,
        }

    first, last = min(years), max(years)
    return {
        "period_years": list(range(first, last + 1)),
        "first_death_year": first,
        "last_death_year": last,
        "n_years": last - first + 1,
    }
