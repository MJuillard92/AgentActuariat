"""
HOTFIX-pre-refacto-2026-05 — Bug 14 : utilitaire centralisé parse_dates_fr().

Vérifie le parsing robuste des dates FR JJ/MM/AAAA, en particulier le cas
qui crashait builder.crude_rates : un mix de dates jour<=12 (ambiguës) et
jour>12 dans la même colonne.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools._shared.date_parsing import parse_dates_fr


def test_ambiguous_then_unambiguous_dates() -> None:
    """Le cas du crash : 11/04/2009 (ambiguë, jour 11) PUIS 28/11/2007
    (jour 28 > 12). Sans dayfirst, pandas infère US et crashe sur la 2e."""
    s = pd.Series(["11/04/2009", "28/11/2007", "07/08/2003"])
    out = parse_dates_fr(s)
    assert out.iloc[0] == pd.Timestamp("2009-04-11")  # 11 avril, pas 4 nov
    assert out.iloc[1] == pd.Timestamp("2007-11-28")  # 28 nov
    assert out.iloc[2] == pd.Timestamp("2003-08-07")  # 7 août


def test_day_le_12_parsed_as_dayfirst() -> None:
    """Une date jour<=12 doit être lue jour-d'abord (FR), pas mois-d'abord."""
    out = parse_dates_fr(pd.Series(["04/03/2010"]))
    assert out.iloc[0] == pd.Timestamp("2010-03-04")  # 4 mars, PAS 3 avril


def test_sentinel_2999_becomes_nat() -> None:
    """31/12/2999 (contrat actif) → NaT, sans OutOfBoundsDatetime."""
    out = parse_dates_fr(pd.Series(["31/12/2010", "31/12/2999"]))
    assert out.iloc[0] == pd.Timestamp("2010-12-31")
    assert pd.isna(out.iloc[1])


@pytest.mark.parametrize("sentinel", [
    "31/12/2099", "31/12/2199", "31/12/2999", "31/12/9999",
])
def test_sentinel_years_become_nat(sentinel: str) -> None:
    out = parse_dates_fr(pd.Series([sentinel]))
    assert pd.isna(out.iloc[0]), f"{sentinel} devrait être NaT"


def test_invalid_zero_date_becomes_nat() -> None:
    """0/0/0 (date invalide) → NaT."""
    out = parse_dates_fr(pd.Series(["01/01/2010", "0/0/0"]))
    assert out.iloc[0] == pd.Timestamp("2010-01-01")
    assert pd.isna(out.iloc[1])


def test_idempotent_on_datetime64() -> None:
    """Une Series déjà datetime64 passe à travers inchangée."""
    s = pd.to_datetime(pd.Series(["2010-01-01", "2015-06-15"]))
    out = parse_dates_fr(s)
    assert (out == s).all()
    assert pd.api.types.is_datetime64_any_dtype(out)


def test_dataframe_auto_detects_date_columns() -> None:
    """Sur un DataFrame, les colonnes contenant 'date' sont parsées."""
    df = pd.DataFrame({
        "date_naissance": ["01/01/1950", "15/06/1960"],
        "date_sortie":    ["28/11/2007", "31/12/2999"],
        "sexe":           ["H", "F"],
    })
    out = parse_dates_fr(df)
    assert pd.api.types.is_datetime64_any_dtype(out["date_naissance"])
    assert out["date_naissance"].iloc[0] == pd.Timestamp("1950-01-01")
    assert out["date_sortie"].iloc[0] == pd.Timestamp("2007-11-28")
    assert pd.isna(out["date_sortie"].iloc[1])
    # Colonne non-date intacte
    assert list(out["sexe"]) == ["H", "F"]


def test_dataframe_explicit_columns() -> None:
    """columns= explicite : seules ces colonnes sont parsées."""
    df = pd.DataFrame({
        "CTREFFET":  ["01/01/2000"],
        "autre":     ["12/12/2012"],
    })
    out = parse_dates_fr(df, columns=["CTREFFET"])
    assert pd.api.types.is_datetime64_any_dtype(out["CTREFFET"])
    assert list(out["autre"]) == ["12/12/2012"]  # inchangée (string)


def test_dataframe_is_copy() -> None:
    """parse_dates_fr ne mute pas le DataFrame d'entrée."""
    df = pd.DataFrame({"date_x": ["01/01/2000"]})
    _ = parse_dates_fr(df)
    assert df["date_x"].iloc[0] == "01/01/2000"  # original intact (string)


def test_unparsable_string_becomes_nat() -> None:
    """Une string non-date → NaT (errors='coerce'), pas d'exception."""
    out = parse_dates_fr(pd.Series(["01/01/2010", "pas une date"]))
    assert out.iloc[0] == pd.Timestamp("2010-01-01")
    assert pd.isna(out.iloc[1])


def test_invalid_type_raises() -> None:
    with pytest.raises(TypeError):
        parse_dates_fr("01/01/2010")
    with pytest.raises(TypeError):
        parse_dates_fr([1, 2, 3])
