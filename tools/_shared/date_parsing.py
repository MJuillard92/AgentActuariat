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
import logging
import re

import pandas as pd

log = logging.getLogger(__name__)

# Au-delà de ce taux de dates coercées en NaT (hors sentinelles), on émet un
# WARNING : signe probable d'un format de date inattendu, pas d'un cas isolé.
_COERCE_WARN_THRESHOLD = 0.005  # 0,5 %

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


def _coerce_report(raw: pd.Series, parsed: pd.Series, col: str = "") -> dict:
    """Compte les dates devenues NaT alors qu'elles étaient présentes en
    entrée et ne sont PAS des sentinelles — i.e. les dates RÉELLEMENT
    illisibles (format inattendu, garbage). Logge un WARNING si le taux
    dépasse le seuil.

    `errors="coerce"` est silencieux par conception : sans ce compteur,
    une colonne au mauvais format perdrait des milliers de lignes en NaT
    sans aucun signal — exposition sous-estimée → table fausse.
    """
    if pd.api.types.is_datetime64_any_dtype(raw):
        input_present = raw.notna()
    else:
        as_str = raw.astype(str).str.strip()
        input_present = raw.notna() & (as_str != "") & (as_str.str.lower() != "nan")
    sentinel = is_sentinel(raw)
    coerced = input_present & parsed.isna() & ~sentinel
    n = int(coerced.sum())
    total = int(input_present.sum())
    pct = (n / total) if total else 0.0
    if pct > _COERCE_WARN_THRESHOLD:
        log.warning(
            "[date_parsing] colonne %r : %d date(s) illisible(s) coercée(s) "
            "en NaT (%.1f%%) — format de date inattendu probable.",
            col or "?", n, pct * 100,
        )
    return {"n_coerced": n, "pct_coerced": round(pct, 4)}


def parse_dates_fr(obj, columns: list[str] | None = None,
                   return_report: bool = False):
    """Parse des dates françaises JJ/MM/AAAA en datetime64.

    Args:
        obj           : pd.Series OU pd.DataFrame.
        columns       : si obj est un DataFrame, colonnes à parser.
                        None → toutes les colonnes dont le nom contient "date".
        return_report : si True, retourne aussi un rapport de coercition.

    Returns:
        - `return_report=False` (défaut) : Series datetime64 ou DataFrame.
        - `return_report=True` : tuple `(obj_parsé, report)` où `report` =
          `{"n_coerced": int, "pct_coerced": float, "by_column": {...}}`
          (`by_column` absent pour une Series).

    Idempotent : une colonne déjà datetime64 passe à travers (les sentinelles
    futur-lointain y sont quand même neutralisées).
    Robuste : valeurs non parsables et sentinelles → NaT, jamais d'exception.
    """
    if isinstance(obj, pd.Series):
        parsed = _parse_one_series(obj)
        if return_report:
            return parsed, _coerce_report(obj, parsed)
        return parsed

    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
        if columns is None:
            columns = [c for c in df.columns if "date" in str(c).lower()]
        by_col: dict[str, int] = {}
        total_coerced = 0
        for col in columns:
            if col in df.columns:
                raw_col = df[col]
                parsed_col = _parse_one_series(raw_col)
                if return_report:
                    rep = _coerce_report(raw_col, parsed_col, col)
                    by_col[col] = rep["n_coerced"]
                    total_coerced += rep["n_coerced"]
                df[col] = parsed_col
        if return_report:
            return df, {"n_coerced": total_coerced, "by_column": by_col}
        return df

    raise TypeError(
        f"parse_dates_fr attend une Series ou un DataFrame, reçu {type(obj)}"
    )
