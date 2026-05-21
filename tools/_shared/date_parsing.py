"""
tools/_shared/date_parsing.py
Utilitaire centralisé de parsing des dates françaises.

HOTFIX-pre-refacto-2026-05 (Bug 14) — créé pour résoudre le crash récurrent
de builder.crude_rates :
    time data "28/11/2007" doesn't match format "%m/%d/%Y"

Cause : `pd.to_datetime(serie)` sans `dayfirst=True` infère le format sur les
premières lignes. Les dates FR `JJ/MM/AAAA` dont le jour <= 12 sont ambiguës
(pandas choisit le format US `%m/%d/%Y`) ; la 1ʳᵉ ligne avec jour > 12 crashe.

Ce module factorise la logique de parsing jusqu'ici dupliquée entre
agents/master/disambiguation.py:_parse_and_clip_dates et
tools/builder/exposure.py.

API :
    parse_dates_fr(obj, columns=None) -> Series | DataFrame
"""
from __future__ import annotations

import re

import pandas as pd

# Dates sentinelles à coercer en NaT :
#  - 2099/2199/.../2999/3000/3999/9999 : contrats actifs (fin inconnue)
#  - 0/0/0, 00/00/0000 : dates invalides
#  - 01/01/1900, 01/01/1800 : placeholders "date inconnue"
_SENTINEL_RE = re.compile(
    r"\b(?:209\d|2[1-9]\d{2}|[3-9]\d{3})\b"      # années >= 2090
    r"|0/0/0|00/00/0000"                          # dates nulles
    r"|01/01/1900|01/01/1800",                    # placeholders anciens
    re.IGNORECASE,
)


def _parse_one_series(s: pd.Series) -> pd.Series:
    """Parse une Series de dates FR en datetime64.

    - Si déjà datetime64 : retournée telle quelle (idempotent).
    - Sentinelles → NaT (retirées AVANT parsing : 2999 dépasse la plage
      pandas Timestamp [1677, 2262] et lèverait OutOfBoundsDatetime).
    - Parsing : format="mixed", dayfirst=True, errors="coerce".
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    as_str = s.astype(str)
    sentinel_mask = as_str.str.contains(_SENTINEL_RE, na=False)

    cleaned = s.copy()
    if sentinel_mask.any():
        cleaned = cleaned.astype(object)
        cleaned[sentinel_mask] = pd.NaT

    return pd.to_datetime(
        cleaned, format="mixed", dayfirst=True, errors="coerce",
    )


def parse_dates_fr(obj, columns: list[str] | None = None):
    """Parse des dates françaises JJ/MM/AAAA en datetime64.

    Args:
        obj     : pd.Series OU pd.DataFrame.
        columns : si obj est un DataFrame, liste des colonnes à parser.
                  None → toutes les colonnes dont le nom contient "date".

    Returns:
        - Series datetime64 si obj est une Series.
        - DataFrame (copie) avec les colonnes ciblées converties si obj est
          un DataFrame.

    Idempotent : une colonne déjà datetime64 passe à travers sans changement.
    Robuste : les valeurs non parsables et les sentinelles deviennent NaT,
    jamais d'exception.
    """
    if isinstance(obj, pd.Series):
        return _parse_one_series(obj)

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if columns is None:
            columns = [c for c in df.columns if "date" in str(c).lower()]
        for col in columns:
            if col in df.columns:
                df[col] = _parse_one_series(df[col])
        return df

    raise TypeError(
        f"parse_dates_fr attend une Series ou un DataFrame, reçu {type(obj)}"
    )
