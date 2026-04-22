"""Intégration : load_plan → _04_redaction sur preamble (Design 3)."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _data_store():
    return {
        "study_objective":        "construction_table_mortalite",
        "start_year":             2019,
        "end_year":               2021,
        "num_observation_years":  3,
        "total_exposure":         1234.5,
        "total_deaths":           42,
        "segmentations":          {
            "sexe": [
                {"valeur": "H", "nb_contrats": 500, "nb_deces": 25,
                 "pct_contrats": 50.0, "pct_deces": 59.5},
                {"valeur": "F", "nb_contrats": 500, "nb_deces": 17,
                 "pct_contrats": 50.0, "pct_deces": 40.5},
            ],
        },
        "serie":                  [
            {"annee": 2019, "nb_deces": 10},
            {"annee": 2020, "nb_deces": 15},
            {"annee": 2021, "nb_deces": 17},
        ],
    }


def test_load_plan_produces_ready_preamble():
    from agents.report.pipeline._01_load_plan import load_plan
    plan = load_plan(_data_store())
    assert plan.n_ready == 1
    assert plan.sections[0].ready


def test_redaction_hydrates_both_visuals():
    from agents.report.pipeline._01_load_plan import load_plan
    from agents.report.pipeline._04_redaction import _run_tables

    plan = load_plan(_data_store())
    tables = _run_tables(plan.sections[0], _data_store())
    assert len(tables) == 1
    # shape de compat : rows[0] = headers, rows[1:] = data
    assert tables[0]["rows"][0][0] == "Sexe"
    assert len(tables[0]["rows"]) == 3  # 1 header + 2 data rows
