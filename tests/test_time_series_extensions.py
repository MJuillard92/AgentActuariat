import pandas as pd
from tools.statistical_analysis.time_series import run


def _fixture_df():
    return pd.DataFrame({
        "date_naissance": ["1970-01-01", "1980-01-01", "1975-01-01"],
        "date_entree":    ["2010-06-01", "2010-03-01", "2011-01-01"],
        "date_sortie":    ["2015-04-01", "2013-09-01", "2014-12-01"],
        "cause_sortie":   ["deces",      "deces",      "autre"],
        "sexe":           ["H",          "F",          "H"],
    })


def test_series_includes_age_moyen_entres():
    result = run(_fixture_df())
    row_2010 = next(r for r in result["serie"] if r["annee"] == 2010)
    assert "age_moyen_entres" in row_2010
    assert 30 <= row_2010["age_moyen_entres"] <= 50


def test_series_includes_age_moyen_deces():
    result = run(_fixture_df())
    row_2013 = next(r for r in result["serie"] if r["annee"] == 2013)
    assert 30 <= row_2013["age_moyen_deces"] <= 40


def test_series_includes_taux_deces():
    result = run(_fixture_df())
    for row in result["serie"]:
        if row["exposition_pa"] > 0:
            expected = row["nb_deces"] / row["exposition_pa"] * 1000
            assert abs(row["taux_deces"] - expected) < 1e-6


def test_by_sex_produces_serie_h_and_serie_f():
    result = run(_fixture_df(), params={"by_sex": True})
    assert "serie_h" in result
    assert "serie_f" in result
    # H : 2 contrats (sexe=H), F : 1 contrat (sexe=F)
    total_entres_h = sum(r["nb_entres"] for r in result["serie_h"])
    total_entres_f = sum(r["nb_entres"] for r in result["serie_f"])
    assert total_entres_h == 2
    assert total_entres_f == 1


def test_by_sex_false_omits_sex_keys():
    result = run(_fixture_df(), params={"by_sex": False})
    assert "serie_h" not in result
    assert "serie_f" not in result
