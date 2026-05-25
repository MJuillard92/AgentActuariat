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
        "total_records":          950,
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
        # US-39 : série H/F requis par la section data_analysis_by_sex
        "serie_h":                [
            {"annee": 2019, "nb_deces": 7},
            {"annee": 2020, "nb_deces": 10},
            {"annee": 2021, "nb_deces": 11},
        ],
        "serie_f":                [
            {"annee": 2019, "nb_deces": 3},
            {"annee": 2020, "nb_deces": 5},
            {"annee": 2021, "nb_deces": 6},
        ],
        # US-39 : ages requis par les sections data_analysis_*
        "ages":                   {
            "distribution_list":   [{"tranche": "20-30", "nb_contrats": 100}],
            "distribution_list_h": [{"tranche": "20-30", "nb_contrats": 60}],
            "distribution_list_f": [{"tranche": "20-30", "nb_contrats": 40}],
        },
        # US-38 : exclusion_report requis par la section data_preprocessing
        "exclusion_report":       {
            "initial_count": 1000,
            "final_count":   950,
            "rules": [
                {"rule_label": "Âge à la sortie < âge à l'entrée", "count": 30},
                {"rule_label": "Données manquantes",                "count": 20},
            ],
        },
        # Section table_construction (raw_rates / full_report) requiert :
        "cohort_min_age": 25,
        "cohort_max_age": 80,
        "qx_table":      [
            {"age": 30, "E_x": 100.0, "D_x": 1, "qx": 0.01, "method_name": "central"},
        ],
        # Section smoothing (full_report) requiert :
        "smoothed_table": [
            {"age": 30, "q_x_brut": 0.01, "q_x_lisse": 0.011},
        ],
        # Nouvelle section table_construction nécessite qx_deciles_table
        "qx_deciles_table": [
            {"age_range": "20-30", "E_x_sum": 100.0, "proportion": 10.0,
             "D_x_observed": 1, "D_x_predicted": 1.0,
             "ecart": 0.0, "ecart_pct": 0.0,
             "ci_lower": 0.0, "ci_upper": 2.0},
        ],
        # Section smoothing (refonte 2026-05-24) nécessite smoothed_deciles_table.
        # Section validation (nouvelle) nécessite validation_tests_table.
        # Produites par builder.statistical_validation (full_report).
        "smoothed_deciles_table": [
            {"age_range": "20-30", "E_x_sum": 100.0, "proportion": 10.0,
             "D_x_observed": 1, "D_x_predicted": 1.1,
             "ecart_pct": 9.0, "ci_lower": 0.0, "ci_upper": 3.0},
        ],
        "validation_tests_table": [
            {"test": "chi_square", "statistic": 1.0, "p_value": 0.5,
             "df": 60, "decision": "accepted",
             "interpretation": "Le lissage est compatible avec les observations."},
        ],
        "smoothness_metrics": {"sum_squared_d2": 0.001,
                               "sum_squared_d3": 0.0001,
                               "mean_abs_d2": 0.005},
        "validation_summary": {"n_tests_accepted": 4, "n_tests_rejected": 1,
                               "alpha": 0.05, "global_assessment": "acceptable"},
        "ci_table": [{"age": 30, "q_x_lisse": 0.011,
                      "ci_lower": 0.005, "ci_upper": 0.017}],
        # Section benchmarking (phase 2) — produites par builder.benchmarking.
        "abatement_table": [{"age": 30, "qx_exp": 0.011, "qx_ref": 0.015,
                             "abatement_factor": 0.73}],
        "smr_global":      0.78,
        "reference_name":  "TH0002",
        "benchmarking_summary": {"global_factor": 0.78, "min_factor": 0.50,
                                 "max_factor": 1.10, "age_min_factor": 25,
                                 "age_max_factor": 80},
    }


def test_load_plan_returns_one_section_for_preamble_yaml():
    plan = load_plan(_preamble_data_store())
    assert isinstance(plan, ReportPlan)
    # preamble + data_preprocessing + data_analysis_{unisex,by_sex}
    # + table_construction + smoothing + smoothing_by_sex + validation
    # + benchmarking + conclusion + annex = 11
    # (smoothing_by_sex activé par défaut quand gender_segmentation
    # n'est pas filtré — la fixture ne fixe pas ce contexte.)
    assert len(plan.sections) == 11
    assert plan.sections[0].section_id == "preamble"
    assert plan.sections[1].section_id == "data_preprocessing"


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
