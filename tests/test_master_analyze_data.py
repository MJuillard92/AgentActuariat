"""Tests pour tools/master/analyze_data_and_request.py (US-10)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.master.analyze_data_and_request import run  # noqa: E402


def _mk_records(rows):
    return pd.DataFrame(rows)


def test_basic_period_2010_2015():
    df = _mk_records([
        {"date_sortie": "2010-03-15", "cause_sortie": "deces"},
        {"date_sortie": "2012-06-01", "cause_sortie": "deces"},
        {"date_sortie": "2015-11-22", "cause_sortie": "deces"},
        {"date_sortie": "2020-01-01", "cause_sortie": "autre"},  # non pris en compte
    ])
    out = run({"records": df}, {})
    assert out["first_death_year"] == 2010
    assert out["last_death_year"] == 2015
    assert out["n_years"] == 6
    assert out["period_years"] == [2010, 2011, 2012, 2013, 2014, 2015]


def test_single_year():
    df = _mk_records([
        {"date_sortie": "2018-01-01", "cause_sortie": "deces"},
        {"date_sortie": "2018-12-31", "cause_sortie": "deces"},
    ])
    out = run({"records": df}, {})
    assert out["first_death_year"] == 2018
    assert out["last_death_year"] == 2018
    assert out["n_years"] == 1
    assert out["period_years"] == [2018]


def test_no_deaths_returns_none():
    df = _mk_records([
        {"date_sortie": "2020-01-01", "cause_sortie": "autre"},
    ])
    out = run({"records": df}, {})
    assert out["first_death_year"] is None
    assert out["last_death_year"] is None
    assert out["n_years"] is None
    assert out["period_years"] == []


def test_accepts_date_objects():
    import datetime as dt
    df = _mk_records([
        {"date_sortie": dt.date(2011, 5, 1), "cause_sortie": "deces"},
        {"date_sortie": dt.date(2013, 5, 1), "cause_sortie": "deces"},
    ])
    out = run({"records": df}, {})
    assert out["first_death_year"] == 2011
    assert out["last_death_year"] == 2013


def test_contract_discoverable_by_registry():
    from knowledge_base.report_template.tool_registry import build_registry
    registry = build_registry(_PROJECT_ROOT / "tools")
    assert "master.analyze_data_and_request" in registry
    spec = registry["master.analyze_data_and_request"]
    assert "records" in spec["inputs"]
    for out_key in ("period_years", "first_death_year", "last_death_year", "n_years"):
        assert out_key in spec["outputs"], f"manque {out_key}"
