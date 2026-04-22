"""E2E rendering : modes unisex (US-40) et by_sex (US-41)."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

TEMPLATE = Path("knowledge_base/report_template/mortality_template.yaml")


def _data_store_unisex() -> dict:
    """data_store minimal couvrant preamble + data_preprocessing + data_analysis_unisex."""
    return {
        "study_objective":        "construction_table_mortalite",
        "gender_segmentation":    "unisex",
        "start_year":             2019,
        "end_year":               2021,
        "num_observation_years":  3,
        "total_records":          900,
        "total_exposure":         2700.0,
        "total_deaths":           42,
        "exclusion_report": {
            "initial_count": 1000,
            "final_count":   900,
            "rules": [
                {"rule_id": f"R{i}", "rule_label": f"Règle {i}", "count": 0, "detail": {}}
                for i in range(1, 7)
            ],
        },
        "segmentations": {"sexe": [
            {"valeur": "H", "nb_contrats": 500, "nb_deces": 25, "pct_contrats": 55.6, "pct_deces": 59.5},
            {"valeur": "F", "nb_contrats": 400, "nb_deces": 17, "pct_contrats": 44.4, "pct_deces": 40.5},
        ]},
        "serie": [
            {"annee": 2019, "nb_entres": 300, "nb_deces": 10, "exposition_pa": 900.0,
             "age_moyen_entres": 45.1, "age_moyen_deces": 62.3, "taux_deces": 11.11},
        ],
        "ages": {
            "age_min": 30, "age_max": 85, "age_moyen": 47.5,
            "distribution_list": [{"tranche": "30-34", "nb_contrats": 50},
                                  {"tranche": "35-39", "nb_contrats": 120}],
        },
    }


def _data_store_by_sex() -> dict:
    base = _data_store_unisex()
    base["gender_segmentation"] = "by_sex"
    base["serie_h"] = [{"annee": 2019, "nb_entres": 180, "nb_deces": 6, "exposition_pa": 540.0,
                        "age_moyen_entres": 45.1, "age_moyen_deces": 62.3, "taux_deces": 11.11}]
    base["serie_f"] = [{"annee": 2019, "nb_entres": 120, "nb_deces": 4, "exposition_pa": 360.0,
                        "age_moyen_entres": 43.2, "age_moyen_deces": 64.1, "taux_deces": 11.11}]
    base["ages"]["distribution_list_h"] = [{"tranche": "30-34", "nb_contrats": 30}]
    base["ages"]["distribution_list_f"] = [{"tranche": "30-34", "nb_contrats": 20}]
    return base


def test_unisex_manifest_activates_correct_sections():
    from knowledge_base.report_template.template_loader import build_manifest
    context = {"gender_segmentation": "unisex"}
    manifest = build_manifest(TEMPLATE, context=context)
    # manifest.sections est une list[dict] (dicts YAML bruts)
    ids = [s["id"] for s in manifest.sections]
    assert "preamble" in ids
    assert "data_preprocessing" in ids
    assert "data_analysis_unisex" in ids
    assert "data_analysis_by_sex" not in ids


def test_unisex_preprocessing_renders_exclusion_table():
    from agents.report.pipeline._01_load_plan import load_plan
    from agents.report.pipeline._04_redaction import _run_tables
    context = {"gender_segmentation": "unisex"}
    plan = load_plan(_data_store_unisex(), context=context)
    preprocessing = next(s for s in plan.sections if s.section_id == "data_preprocessing")
    tables = _run_tables(preprocessing, _data_store_unisex())
    assert len(tables) == 1
    # rows[0] = header, rows[1:] = data (1 par règle)
    assert len(tables[0]["rows"]) == 7  # 1 header + 6 rules


def test_by_sex_manifest_activates_correct_sections():
    from knowledge_base.report_template.template_loader import build_manifest
    context = {"gender_segmentation": "by_sex"}
    manifest = build_manifest(TEMPLATE, context=context)
    ids = [s["id"] for s in manifest.sections]
    assert "data_analysis_by_sex" in ids
    assert "data_analysis_unisex" not in ids


def test_by_sex_section_has_four_visuals():
    from agents.report.pipeline._01_load_plan import load_plan
    context = {"gender_segmentation": "by_sex"}
    plan = load_plan(_data_store_by_sex(), context=context)
    by_sex_section = next(s for s in plan.sections if s.section_id == "data_analysis_by_sex")
    vs = by_sex_section.visual_specs
    # visual_specs sont des dicts YAML bruts (pas des dataclasses)
    vs_ids = [v["id"] for v in vs]
    assert vs_ids == ["annual_statistics_male", "annual_statistics_female",
                      "exposure_distribution_male", "exposure_distribution_female"]
