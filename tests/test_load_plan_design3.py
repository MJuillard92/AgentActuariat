"""Tests US-22 : load_plan v2 lit Design 3 via template_loader."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.report.pipeline._01_load_plan import load_plan, SectionPlan, ReportPlan  # noqa: E402


def _preamble_data_store():
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


def test_load_plan_returns_one_section_for_preamble_yaml():
    plan = load_plan(_preamble_data_store())
    assert isinstance(plan, ReportPlan)
    assert len(plan.sections) == 1
    assert plan.sections[0].section_id == "preamble"


def test_section_plan_has_resolved_narrative():
    plan = load_plan(_preamble_data_store())
    preamble = plan.sections[0]
    assert "{{ study_objective }}" not in preamble.prompt
    assert "construction_table_mortalite" in preamble.prompt
    assert "2019" in preamble.prompt
    assert "2021" in preamble.prompt


def test_section_plan_visual_specs_pass_through():
    plan = load_plan(_preamble_data_store())
    preamble = plan.sections[0]
    ids = [v["id"] for v in preamble.visual_specs]
    assert "portfolio_composition" in ids
    assert "deaths_per_year" in ids


def test_section_plan_ready_when_all_placeholders_resolvable():
    plan = load_plan(_preamble_data_store())
    assert plan.sections[0].ready is True
    assert plan.missing_fields == []


def test_section_plan_not_ready_on_missing_placeholder():
    ds = _preamble_data_store()
    del ds["total_deaths"]
    plan = load_plan(ds)
    assert plan.sections[0].ready is False
    assert "total_deaths" in plan.missing_fields


def test_completion_plan_reads_rag_query_from_yaml():
    from agents.report.pipeline._03_completion_plan import _query_for_section
    q = _query_for_section("preamble", "Préambule")
    assert q == "formulation préambule table mortalité portefeuille"
