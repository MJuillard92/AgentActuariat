"""Tests pour les 4 tools mortality.compute_* consommés par le preamble."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.mortality.compute_exposure import run as run_exposure        # noqa: E402
from tools.mortality.compute_deaths import run as run_deaths            # noqa: E402
from tools.mortality.compute_composition import run as run_composition  # noqa: E402
from tools.mortality.compute_deaths_timeseries import run as run_ts     # noqa: E402


def _records():
    return pd.DataFrame([
        # entree         sortie            cause    sexe  naissance
        ["2010-01-01", "2013-06-30", "deces", "H", "1950-05-01"],
        ["2011-03-15", "2014-12-31", "autre", "F", "1960-08-20"],
        ["2012-01-01", "2012-12-31", "deces", "F", "1945-03-10"],
        ["2010-06-01", "2015-03-15", "deces", "H", "1955-11-05"],
    ], columns=["date_entree", "date_sortie", "cause_sortie", "sexe", "date_naissance"])


# ───────── compute_exposure ─────────

def test_exposure_sums_years_over_period():
    out = run_exposure({"records": _records(), "period": [2010, 2011, 2012, 2013, 2014, 2015]}, {})
    assert isinstance(out["cumulative_exposure"], float)
    assert out["cumulative_exposure"] > 0


def test_exposure_empty_records():
    df = pd.DataFrame(columns=["date_entree", "date_sortie"])
    out = run_exposure({"records": df, "period": [2010]}, {})
    assert out["cumulative_exposure"] == 0.0


# ───────── compute_deaths ─────────

def test_deaths_counts_only_deces_in_period():
    out = run_deaths({"records": _records(), "period": [2010, 2011, 2012, 2013, 2014, 2015]}, {})
    assert out["death_count"] == 3  # 3 cause_sortie == deces


def test_deaths_excludes_out_of_period():
    out = run_deaths({"records": _records(), "period": [2020]}, {})
    assert out["death_count"] == 0


# ───────── compute_composition ─────────

def test_composition_by_sex():
    out = run_composition({"records": _records(), "group_by": ["sexe"]}, {})
    tbl = out["composition_table"]
    assert isinstance(tbl, pd.DataFrame)
    assert set(tbl.columns) >= {"sexe", "n_lives", "exposure", "deaths"}
    row_h = tbl[tbl["sexe"] == "H"].iloc[0]
    assert row_h["n_lives"] == 2
    assert row_h["deaths"] == 2


# ───────── compute_deaths_timeseries ─────────

def test_timeseries_returns_one_entry_per_year():
    out = run_ts({"records": _records(), "period": [2012, 2013, 2014, 2015]}, {})
    series = out["series"]
    assert len(series) == 4
    assert {s["year"] for s in series} == {2012, 2013, 2014, 2015}
    total = sum(s["deaths"] for s in series)
    assert total == 3


def test_timeseries_missing_year_has_zero():
    df = pd.DataFrame([
        ["2010-01-01", "2012-06-01", "deces", "H", "1950-01-01"],
    ], columns=["date_entree", "date_sortie", "cause_sortie", "sexe", "date_naissance"])
    out = run_ts({"records": df, "period": [2010, 2011, 2012]}, {})
    years_to_deaths = {s["year"]: s["deaths"] for s in out["series"]}
    assert years_to_deaths[2010] == 0
    assert years_to_deaths[2011] == 0
    assert years_to_deaths[2012] == 1


# ───────── Registry discovery ─────────

def test_all_four_tools_in_registry():
    from knowledge_base.report_template.tool_registry import build_registry
    registry = build_registry(_PROJECT_ROOT / "tools")
    expected = {
        "mortality.compute_exposure":          ("records", "period", "cumulative_exposure"),
        "mortality.compute_deaths":            ("records", "period", "death_count"),
        "mortality.compute_composition":       ("records", "group_by", "composition_table"),
        "mortality.compute_deaths_timeseries": ("records", "period", "series"),
    }
    for name, (in1, in2, out_key) in expected.items():
        assert name in registry, f"manquant: {name}"
        spec = registry[name]
        assert in1 in spec["inputs"], f"{name}: input {in1} manquant"
        assert in2 in spec["inputs"], f"{name}: input {in2} manquant"
        assert out_key in spec["outputs"], f"{name}: output {out_key} manquant"
