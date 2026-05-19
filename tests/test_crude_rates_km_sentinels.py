"""
HOTFIX-pre-refacto-2026-05 — Bug 11 : crude_rates.kaplan_meier defense
in depth contre dates sentinelles résiduelles.

Scénario : le user upload un dataset, normalisation auto OK → parquet
normalisé clippé. Mais si quelque chose casse la normalisation (config
manquante, format inattendu), le fallback charge le dataset original
brut. Sans defense in depth, KM crashe sur pd.to_datetime("31/12/2999").

Le test simule un DataFrame brut (non passé par normalisation) avec
sentinelles, et vérifie que crude_rates.run ne crashe pas + retourne
une qx_table cohérente.
"""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest


def _build_raw_df_with_sentinels() -> pd.DataFrame:
    """DataFrame individuel brut (non normalisé) avec sentinelles 2999/2099."""
    return pd.DataFrame({
        "date_naissance": ["01/01/1950", "01/01/1955", "01/01/1960", "01/01/1965"] * 25,
        "date_entree":    ["01/01/2000", "01/01/2005", "01/01/2010", "01/01/2012"] * 25,
        "date_sortie":    ["31/12/2020", "31/12/2999", "31/12/2099", "31/12/2018"] * 25,
        "cause_sortie":   ["deces",      "actif",      "actif",      "deces"] * 25,
    })


def test_crude_rates_km_handles_sentinel_dates() -> None:
    """crude_rates.run en méthode kaplan_meier ne doit PAS crasher
    quand cleaned_records contient des sentinelles non clippées."""
    from tools.builder.crude_rates import run

    df = _build_raw_df_with_sentinels()
    data = {
        "exposure_table":   [{"age": a, "E_x": 50.0, "D_x": 1} for a in range(60, 80)],
        "cleaned_records":  df.to_dict(orient="records"),  # données BRUTES avec sentinelles
    }
    result = run(data, params={"method": "kaplan_meier"})

    assert "erreur" not in result, f"crude_rates KM crashé : {result.get('erreur')}"
    assert "qx_table" in result
    assert result["method"] == "kaplan_meier"
    assert len(result["qx_table"]) > 0


def test_crude_rates_central_unaffected() -> None:
    """Régression : la méthode 'central' (qui n'utilise pas df_indiv) reste
    fonctionnelle. Pas de sentinelles à clipper côté individual data."""
    from tools.builder.crude_rates import run

    data = {
        "exposure_table": [{"age": a, "E_x": 100.0, "D_x": 2} for a in range(60, 80)],
    }
    result = run(data, params={"method": "central"})

    assert "erreur" not in result
    assert result["method"] == "central"
    assert len(result["qx_table"]) > 0
