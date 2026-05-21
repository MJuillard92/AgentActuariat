"""
tools/_shared/date_parsing.py
Utilitaire centralisé de parsing des dates françaises.

HOTFIX-pre-refacto-2026-05 (Bug 14) — créé pour résoudre le crash récurrent
de builder.crude_rates :
    time data "28/11/2007" doesn't match format "%m/%d/%Y"

Chantier dates 2026-05-21 — détection des sentinelles par RÈGLE, pas par
énumération de regex. L'ancienne regex `209\\d|2[1-9]\\d{2}|...` ratait les
années 2000-2089 (2040, 2050, 2060, 2080). La détection est désormais :
  - parsing tolérant : `errors="coerce"` neutralise les dates illisibles ET
    hors-plage pandas (> 2262 : 2999, 9999) → NaT, sans exception ;
  - règle numérique : toute date dont l'année dépasse `année_courante + 1`
    est une sentinelle / date implausible (futur lointain) — capture
    2040/2050/2060/2099/2200… uniformément, aucun trou.

API :
    parse_dates_fr(obj, columns=None) -> Series | DataFrame
    is_sentinel(series) -> Series[bool]   — masque des dates futur-lointain
"""
from __future__ import annotations

import datetime as _dt
import re

import pandas as pd

# Placeholders « date inconnue » — chaînes exactes connues (2 constantes
# documentées, PAS une énumération d'années sentinelles).
_PLACEHOLDER_RE = re.compile(r"01/01/1900|01/01/1800", re.IGNORECASE)


def _sentinel_year_threshold() -> int:
    """Année au-delà de laquelle une date est implausible (= sentinelle).

    Un contrat ne se termine pas, une personne ne naît pas, dans le futur
    lointain. Seuil = année courante + 1 (cohérent avec le catch-all
    historique de disambiguation._parse_and_clip_dates).
    """
    return _dt.date.today().year + 1


def is_sentinel(s: pd.Series) -> pd.Series:
    """Masque booléen des dates sentinelles (futur lointain) d'une Series.

    Règle : année > `année_courante + 1`. Fonctionne sur une Series string
    (extraction textuelle du 1ᵉʳ groupe de 4 chiffres = année) OU datetime64.
    Détecte 2040/2050/2999/9999 uniformément — aucune énumération d'années.

    NB : `0/0/0` (aucune année 4 chiffres) → False ici (c'est du garbage,
    neutralisé par errors="coerce" au parsing, pas une sentinelle de censure).
    """
    thr = _sentinel_year_threshold()
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.notna() & (s.dt.year > thr)
    years = s.astype(str).str.extract(r"(\d{4})", expand=False)
    years_num = pd.to_numeric(years, errors="coerce")
    return years_num.notna() & (years_num > thr)


def _parse_one_series(s: pd.Series) -> pd.Series:
    """Parse une Series de dates FR en datetime64.

    - Déjà datetime64 : seules les sentinelles futur-lointain sont neutralisées.
    - String : placeholders connus → NaT, puis parsing tolérant
      (format="mixed", dayfirst=True, errors="coerce" — les dates illisibles
      et hors-plage pandas deviennent NaT sans exception), puis règle
      sentinelle (année > seuil → NaT).
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        parsed = s
    else:
        cleaned = s
        as_str = s.astype(str)
        placeholder = as_str.str.contains(_PLACEHOLDER_RE, na=False)
        if placeholder.any():
            cleaned = s.astype(object)
            cleaned[placeholder] = pd.NaT
        parsed = pd.to_datetime(
            cleaned, format="mixed", dayfirst=True, errors="coerce",
        )

    # Règle sentinelle : toute année future lointaine → NaT.
    far = is_sentinel(s)
    if far.any():
        parsed = parsed.copy()
        parsed[far] = pd.NaT
    return parsed


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

    Idempotent : une colonne déjà datetime64 passe à travers (les sentinelles
    futur-lointain y sont quand même neutralisées).
    Robuste : valeurs non parsables et sentinelles → NaT, jamais d'exception.
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
