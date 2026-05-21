"""
HOTFIX-pre-refacto-2026-05 — Bug 14 : crude_rates kaplan_meier ne doit plus
crasher sur des dates FR string ambiguës.

Reproduit le scénario prod exact : une colonne date_sortie qui commence par
des dates jour<=12 (ambiguës, pandas infère US %m/%d/%Y) puis contient
28/11/2007 (jour 28 > 12) → l'ancien code crashait.
"""
from __future__ import annotations

import pandas as pd
import pytest


def _df_with_ambiguous_then_unambiguous_dates() -> pd.DataFrame:
    """DataFrame individuel brut : les 3 premières lignes ont jour<=12
    (ambiguës), puis 28/11/2007 (jour 28) qui faisait crasher pandas."""
    naiss = ["01/01/1950", "02/01/1955", "03/01/1960", "04/01/1965"]
    entree = ["11/04/2009", "05/03/2008", "07/06/2010", "28/11/2007"]
    sortie = ["11/04/2015", "05/03/2018", "31/12/2999", "28/11/2017"]
    cause = ["deces", "deces", "actif", "deces"]
    n = 30
    return pd.DataFrame({
        "date_naissance": naiss * n,
        "date_entree":    entree * n,
        "date_sortie":    sortie * n,
        "cause_sortie":   cause * n,
    })


def test_km_does_not_crash_on_ambiguous_fr_dates() -> None:
    """crude_rates KM sur cleaned_records avec dates FR string ambiguës :
    ne crashe plus, produit qx_table."""
    from tools.builder.crude_rates import run

    df = _df_with_ambiguous_then_unambiguous_dates()
    data = {
        "exposure_table":  [{"age": a, "E_x": 40.0, "D_x": 1} for a in range(55, 75)],
        "cleaned_records": df.to_dict(orient="records"),
    }
    result = run(data, params={"method": "kaplan_meier"})

    assert "erreur" not in result, f"crude_rates KM a crashé : {result.get('erreur')}"
    assert "qx_table" in result
    assert result["method"] == "kaplan_meier"
    assert len(result["qx_table"]) > 0


def test_km_dates_parsed_dayfirst_not_usfirst() -> None:
    """Vérifie que _prepare_dates_for_km lit bien JJ/MM (28/11 = 28 nov)."""
    from tools.builder.crude_rates import _prepare_dates_for_km

    df = pd.DataFrame({
        "date_naissance": ["01/01/1950"],
        "date_entree":    ["28/11/2007"],
        "date_sortie":    ["11/04/2015"],
    })
    out = _prepare_dates_for_km(df)
    assert out["date_entree"].iloc[0] == pd.Timestamp("2007-11-28")  # 28 nov
    assert out["date_sortie"].iloc[0] == pd.Timestamp("2015-04-11")  # 11 avril


def test_km_central_method_still_works() -> None:
    """Régression : la méthode 'central' (sans dates individuelles) reste OK."""
    from tools.builder.crude_rates import run

    data = {"exposure_table": [{"age": a, "E_x": 100.0, "D_x": 2}
                               for a in range(55, 75)]}
    result = run(data, params={"method": "central"})
    assert "erreur" not in result
    assert result["method"] == "central"
