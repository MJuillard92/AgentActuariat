"""Tests US-20 : builder exécute le DAG du manifest preamble."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _make_records():
    return pd.DataFrame({
        "date_naissance": ["1960-01-01", "1965-06-15", "1970-03-20"],
        "date_entree":    ["2019-01-01", "2019-01-01", "2019-01-01"],
        "date_sortie":    ["2020-06-01", "2021-12-31", "2021-07-15"],
        "cause_sortie":   ["deces",      "autre",      "deces"],
        "sexe":           ["H",          "F",          "H"],
    })


def test_execute_manifest_dag_fills_all_builder_outputs():
    from agents.mortality.agents.builder_node import _execute_manifest_dag

    data_store = {
        "input_records":       _make_records(),
        "raw_user_request":    "construis-moi une table de mortalité",
    }

    result = _execute_manifest_dag(data_store)

    assert result is not None
    updates = result["updates"]
    for key in (
        "total_exposure_years",
        "total_deaths",
        "portfolio_composition_by_sex",
        "deaths_by_year_series",
        "observation_period_years",
        "start_year",
        "end_year",
        "num_observation_years",
        "study_objective",
    ):
        assert key in updates, f"clé manquante: {key}"
    assert "needed_keys" in result
    assert set(result["needed_keys"]).issubset(set(updates.keys()))


def test_execute_manifest_dag_noop_without_records():
    from agents.mortality.agents.builder_node import _execute_manifest_dag

    result = _execute_manifest_dag({"input_records": None})

    assert result is None


def test_builder_node_short_circuits_llm_when_dag_complete():
    """builder_node ne doit PAS appeler le LLM si le DAG remplit tous les builder_outputs."""
    from agents.mortality.agents.builder_node import builder_node

    state = {
        "data_store": {
            "input_records":    _make_records(),
            "raw_user_request": "construis-moi une table de mortalité",
        },
        "messages": [],
    }

    result = builder_node(state)

    assert result["plan_established"] is True
    assert result["active_agent"] == "master"
    # <BUILD_DONE> doit être dans le dernier message
    msgs = result["messages"]
    assert any("<BUILD_DONE>" in getattr(m, "content", "") for m in msgs)
    # Les 4 clés builder_outputs doivent être dans le data_store
    ds = result["data_store"]
    for k in ("total_exposure_years", "total_deaths",
              "portfolio_composition_by_sex", "deaths_by_year_series"):
        assert ds.get(k) is not None, f"{k} manquant"
