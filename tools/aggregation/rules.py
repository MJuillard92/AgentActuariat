"""
TOOL CONTRACT — aggregation.rules
═════════════════════════════════

CATALOGUE METADATA
------------------
name          : aggregation.rules
domain        : generic
version       : 1.0.0
author        : Marc Juillard
last_updated  : 2026-04-20

DESCRIPTION
-----------
Registre générique de règles d'agrégation. Appelable par le Builder
pour pré-agréger une table avant BUILD_DONE, selon la règle déclarée
dans visual_specs.aggregation du YAML.

Règles supportées (V1) :
  - none                 : retourne source inchangée
  - fixed_width          : buckets de largeur params.width sur bucket_col
  - equal_count          : params.n_buckets buckets équi-effectif
  - exposure_share_min   : buckets minimaux tels que weight_col cumulé
                           atteigne params.min_share du total

INPUTS
------
params:
  source:
    type    : table
  rule:
    type    : string
  params:
    type    : dict
  weight:
    type    : table
    note    : optionnel, utilisé par certaines règles.

OUTPUTS
-------
return_payload:
  aggregated : table
"""
from __future__ import annotations

from typing import Any

import pandas as pd


def _rule_none(source: pd.DataFrame, params: dict) -> pd.DataFrame:
    return source.copy()


def _numeric_cols(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]


def _fixed_width(source: pd.DataFrame, params: dict) -> pd.DataFrame:
    width = int(params.get("width", 5))
    col = params.get("bucket_col", "age")
    s = source.copy()
    lo = int(s[col].min())
    hi = int(s[col].max())
    edges = list(range(lo, hi + width + 1, width))
    labels = [f"{edges[i]}-{edges[i + 1] - 1}" for i in range(len(edges) - 1)]
    s["_bucket"] = pd.cut(s[col], bins=edges, labels=labels, right=False, include_lowest=True)
    num_cols = _numeric_cols(s, exclude={col, "_bucket"})
    agg = s.groupby("_bucket", observed=True)[num_cols].sum().reset_index().rename(columns={"_bucket": "bucket"})
    return agg


def _equal_count(source: pd.DataFrame, params: dict) -> pd.DataFrame:
    n = int(params.get("n_buckets", 5))
    col = params.get("bucket_col", "age")
    s = source.sort_values(col).copy().reset_index(drop=True)
    s["_bucket"] = pd.qcut(s[col], q=n, duplicates="drop")
    num_cols = _numeric_cols(s, exclude={col, "_bucket"})
    agg = s.groupby("_bucket", observed=True)[num_cols].sum().reset_index().rename(columns={"_bucket": "bucket"})
    return agg


def _exposure_share_min(source: pd.DataFrame, params: dict) -> pd.DataFrame:
    min_share = float(params.get("min_share", 0.05))
    col = params.get("bucket_col", "age")
    weight_col = params.get("weight_col", "exposure")
    s = source.sort_values(col).copy().reset_index(drop=True)
    total = s[weight_col].sum()
    if total <= 0:
        return s

    threshold = min_share * total
    groups: list[list[int]] = []
    current: list[int] = []
    current_weight = 0.0
    for idx, w in enumerate(s[weight_col]):
        current.append(idx)
        current_weight += w
        if current_weight >= threshold:
            groups.append(current)
            current = []
            current_weight = 0.0
    if current:
        if groups:
            groups[-1].extend(current)
        else:
            groups.append(current)

    rows = []
    num_cols = _numeric_cols(s, exclude={col})
    for g in groups:
        sub = s.iloc[g]
        lo = int(sub[col].min())
        hi = int(sub[col].max())
        row: dict[str, Any] = {"bucket": f"{lo}-{hi}" if lo != hi else str(lo)}
        for c in num_cols:
            row[c] = sub[c].sum()
        rows.append(row)
    return pd.DataFrame(rows)


_RULES = {
    "none":               _rule_none,
    "fixed_width":        _fixed_width,
    "equal_count":        _equal_count,
    "exposure_share_min": _exposure_share_min,
}


def run(data: dict, params: dict) -> dict:
    source = data["source"]
    rule = params.get("rule", "none")
    rule_params = params.get("params") or {}
    if rule not in _RULES:
        raise ValueError(f"règle d'agrégation inconnue : {rule!r}. Valides : {list(_RULES)}")
    if not isinstance(source, pd.DataFrame):
        source = pd.DataFrame(source)
    return {"aggregated": _RULES[rule](source, rule_params)}
