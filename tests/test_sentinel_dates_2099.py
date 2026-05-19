"""
HOTFIX-pre-refacto-2026-05 — Bug 10 : _parse_and_clip_dates doit aussi
clipper les sentinelles 2099/2199/2299/... (toute année > obs_end + N),
pas seulement 2999/9999.

Symptôme prod : un dataset avec date_sortie=31/12/2099 passe la normalisation
sans clipping (la regex actuelle ne matche que 2999/9999/3000/3999) → les
tools downstream comparent à observation_end et crashent ou produisent
des résultats aberrants.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _build_df_with_sentinel(sentinel_year: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date_entree": ["01/01/2000", "01/01/2005", "01/01/2010"],
        "date_sortie": ["31/12/2015", f"31/12/{sentinel_year}", f"31/12/{sentinel_year}"],
        "cause_sortie": ["deces", "actif", "actif"],
        "date_naissance": ["01/01/1950", "01/01/1960", "01/01/1970"],
    })


@pytest.mark.parametrize("sentinel_year", [2099, 2199, 2299, 2399, 2999, 9999])
def test_sentinel_year_is_clipped(sentinel_year: int) -> None:
    """Toute année strictement supérieure à observation_end + horizon
    doit être clippée à observation_end (ou NaT si pas d'obs_end)."""
    from agents.master.disambiguation import _parse_and_clip_dates

    df = _build_df_with_sentinel(sentinel_year)
    df_out, obs_end_iso = _parse_and_clip_dates(df.copy(), dataset_ref="test")

    assert obs_end_iso is not None, "obs_end devrait être détecté depuis le décès 2015"
    obs_end = pd.Timestamp(obs_end_iso)

    sortie = pd.to_datetime(df_out["date_sortie"], errors="coerce")
    max_sortie = sortie.max()
    assert max_sortie is pd.NaT or max_sortie <= obs_end, (
        f"date_sortie max ({max_sortie}) > obs_end ({obs_end}) après clip "
        f"pour sentinelle {sentinel_year}"
    )


def test_legitimate_future_within_horizon_preserved() -> None:
    """Une date future raisonnable (obs_end + 1 an, dans l'horizon) ne doit
    PAS être clippée. Test de non-régression."""
    from agents.master.disambiguation import _parse_and_clip_dates

    df = pd.DataFrame({
        "date_entree":  ["01/01/2010", "01/01/2015"],
        "date_sortie":  ["31/12/2020", "31/12/2024"],
        "cause_sortie": ["deces",      "actif"],
        "date_naissance": ["01/01/1950", "01/01/1960"],
    })
    df_out, _ = _parse_and_clip_dates(df.copy(), dataset_ref="test")

    sortie = pd.to_datetime(df_out["date_sortie"], errors="coerce")
    # La date 2024 doit être préservée (≈ obs_end)
    assert sortie.iloc[1].year == 2024


def test_no_sentinel_no_change() -> None:
    """Aucune sentinelle : tout passe inchangé."""
    from agents.master.disambiguation import _parse_and_clip_dates

    df = pd.DataFrame({
        "date_entree":  ["01/01/2010"],
        "date_sortie":  ["31/12/2018"],
        "cause_sortie": ["deces"],
        "date_naissance": ["01/01/1950"],
    })
    df_out, obs_end_iso = _parse_and_clip_dates(df.copy(), dataset_ref="test")

    assert obs_end_iso is not None
    sortie = pd.to_datetime(df_out["date_sortie"], errors="coerce")
    assert sortie.iloc[0].year == 2018
