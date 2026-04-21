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

    updates = _execute_manifest_dag(data_store)

    assert updates is not None
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


def test_execute_manifest_dag_noop_without_records():
    from agents.mortality.agents.builder_node import _execute_manifest_dag

    updates = _execute_manifest_dag({"input_records": None})

    assert updates is None
