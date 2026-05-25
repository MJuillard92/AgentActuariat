"""Tests — pipeline par sexe (exposure → crude_rates → smoothing).

Vérifie que le paramètre by_sex=True propage correctement à travers les
trois tools du pipeline mortalité, produisant les variantes _h et _f en
plus des sorties unisex (rétro-compat préservée).

Plan qualité-rapport phase 2 (2026-05-24).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _make_records_with_sex():
    """Mini-portefeuille mixte avec H et F clairement identifiés."""
    return pd.DataFrame({
        "date_naissance": ["01/01/1950", "01/01/1955", "01/01/1960",
                           "01/01/1948", "01/01/1958", "01/01/1962"],
        "date_entree":    ["01/01/2000", "01/01/2002", "01/01/2003",
                           "01/01/2001", "01/01/2002", "01/01/2003"],
        "date_sortie":    ["31/12/2010", "31/12/2012", "31/12/2015",
                           "31/12/2011", "31/12/2014", "31/12/2018"],
        "cause_sortie":   ["deces", "deces", "autre",
                           "autre", "deces", "deces"],
        "sexe":           ["H", "H", "H", "F", "F", "F"],
    })


# ── builder.exposure(by_sex=True) ────────────────────────────────────────────

def test_exposure_by_sex_produces_h_f_tables():
    from tools.builder.exposure import run
    df = _make_records_with_sex()
    res = run(df, {"by_sex": True})

    assert "exposure_table" in res, "unisex toujours présent (rétro-compat)"
    assert "exposure_table_h" in res, "table hommes attendue"
    assert "exposure_table_f" in res, "table femmes attendue"
    assert "total_exposure_h" in res
    assert "total_exposure_f" in res
    assert "total_deaths_h" in res
    assert "total_deaths_f" in res

    # Cohérence : somme H + F ≈ unisex
    assert abs(res["total_exposure_h"] + res["total_exposure_f"]
               - res["total_exposure"]) < 0.5
    assert res["total_deaths_h"] + res["total_deaths_f"] == res["total_deaths"]


def test_exposure_without_by_sex_unchanged():
    """Sans by_sex, seul l'unisex est produit (rétro-compat stricte)."""
    from tools.builder.exposure import run
    df = _make_records_with_sex()
    res = run(df, {})
    assert "exposure_table" in res
    assert "exposure_table_h" not in res
    assert "exposure_table_f" not in res


def test_exposure_by_sex_no_sex_column_warns():
    """by_sex=True sans colonne sexe → unisex + avertissement, pas d'échec."""
    from tools.builder.exposure import run
    df = _make_records_with_sex().drop(columns=["sexe"])
    res = run(df, {"by_sex": True})
    assert "exposure_table" in res
    assert "exposure_table_h" not in res
    assert "avertissement_by_sex" in res


# ── builder.crude_rates(by_sex=True) ─────────────────────────────────────────

def test_crude_rates_by_sex_propagates():
    from tools.builder.exposure import run as run_expo
    from tools.builder.crude_rates import run as run_crude

    df = _make_records_with_sex()
    expo = run_expo(df, {"by_sex": True})
    res = run_crude(data=expo, params={"by_sex": True})

    assert "qx_table" in res, "unisex (rétro-compat)"
    assert "qx_table_h" in res
    assert "qx_table_f" in res
    # Les tables H/F doivent contenir les mêmes colonnes que unisex
    for k in ("qx_table_h", "qx_table_f"):
        assert all("age" in r for r in res[k])
        assert all("qx" in r for r in res[k])


def test_crude_rates_by_sex_without_expo_h_warns():
    """by_sex=True mais exposure_table_h absent → unisex + avertissement."""
    from tools.builder.crude_rates import run as run_crude
    data = {"exposure_table": [{"age": 30, "E_x": 100.0, "D_x": 1}]}
    res = run_crude(data=data, params={"by_sex": True})
    assert "qx_table" in res
    assert "qx_table_h" not in res
    assert "avertissement_by_sex" in res


# ── builder.smoothing(by_sex=True) ───────────────────────────────────────────

def test_smoothing_by_sex_propagates_stub_mode():
    """En mode STUB (var d'env), le lissage = identité → on vérifie juste
    la propagation des clés _h / _f, sans dépendre du notebook 04_smoothing."""
    import os
    os.environ["AGENT_SMOOTHING_STUB"] = "1"
    try:
        from tools.builder.smoothing import run as run_smooth
        qx_h = [{"age": a, "E_x": 100.0, "D_x": 1, "qx": 0.01 + a * 0.001}
                for a in range(30, 60)]
        qx_f = [{"age": a, "E_x": 100.0, "D_x": 1, "qx": 0.008 + a * 0.0008}
                for a in range(30, 60)]
        data = {"qx_table": qx_h, "qx_table_h": qx_h, "qx_table_f": qx_f}
        res = run_smooth(data=data, params={"by_sex": True})
        assert "smoothed_table" in res
        assert "smoothed_table_h" in res
        assert "smoothed_table_f" in res
        assert all("q_x_lisse" in r for r in res["smoothed_table_h"])
        assert all("q_x_lisse" in r for r in res["smoothed_table_f"])
    finally:
        os.environ.pop("AGENT_SMOOTHING_STUB", None)


def test_smoothing_by_sex_without_qx_h_warns():
    from tools.builder.smoothing import run as run_smooth
    import os
    os.environ["AGENT_SMOOTHING_STUB"] = "1"
    try:
        data = {"qx_table": [{"age": 30, "E_x": 100.0, "D_x": 1, "qx": 0.01}]}
        res = run_smooth(data=data, params={"by_sex": True})
        assert "smoothed_table" in res
        assert "smoothed_table_h" not in res
        assert "avertissement_by_sex" in res
    finally:
        os.environ.pop("AGENT_SMOOTHING_STUB", None)
