"""Tests US-15 : master lit les clés Builder depuis build_manifest()."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_get_builder_keys_from_manifest_returns_preamble_keys():
    """_get_builder_keys doit être une projection fidèle de build_manifest().builder_outputs."""
    from agents.mortality.agents.master_node import _get_builder_keys
    from knowledge_base.report_template.template_loader import build_manifest

    keys = _get_builder_keys()
    manifest = build_manifest()

    assert keys == [entry.key for entry in manifest.builder_outputs]
    assert len(keys) >= 1  # preamble a 4 clés aujourd'hui ; garde-fou anti-vide
    # Les 4 clés preamble (noms natifs des tools existants) :
    assert {
        "total_exposure",
        "total_deaths",
        "segmentations",
        "serie",
    }.issubset(set(keys))


def test_preflight_writer_ready_when_all_manifest_keys_present():
    from agents.mortality.agents.master_node import _preflight_writer

    data_store = {
        # preprocessing outputs
        "cleaned_records":  [{"id": 1}],   # non-empty list (falsy [] would fail)
        "exclusion_report": {"initial_count": 50, "final_count": 48, "rules": []},
        "total_records":    48,
        # builder outputs (exposure + crude_rates)
        "total_exposure":   1234.5,
        "exposure_table":   [{"age": 30, "E_x": 100.0, "D_x": 1}],
        "total_deaths":     42,
        "cohort_min_age":   25,
        "cohort_max_age":   80,
        "segmentations":    {"sexe": [{"valeur": "H", "nb_contrats": 500}]},
        "serie":            [{"annee": 2020, "nb_deces": 10}],
        "serie_h":          [{"annee": 2020, "nb_deces": 6}],
        "serie_f":          [{"annee": 2020, "nb_deces": 4}],
        "ages":             {"age_min": 25.0, "age_max": 80.0},
        "qx_table":         [{"age": 30, "E_x": 100.0, "D_x": 1, "qx": 0.01, "method_name": "central"}],
        "smoothed_table":   [{"age": 30, "q_x_brut": 0.01, "q_x_lisse": 0.011}],
        "qx_deciles_table": [{"age_range": "20-30", "E_x_sum": 100.0, "proportion": 10.0,
                              "D_x_observed": 1, "D_x_predicted": 1.0,
                              "ecart": 0.0, "ecart_pct": 0.0,
                              "ci_lower": 0.0, "ci_upper": 2.0}],
        "ci_table":         [{"age": 30, "q_x_lisse": 0.011, "ci_lower": 0.005, "ci_upper": 0.017}],
    }

    ready, missing = _preflight_writer(data_store)

    assert ready is True, f"missing : {missing}"
    assert missing == []


def test_preflight_writer_missing_keys():
    from agents.mortality.agents.master_node import _preflight_writer

    data_store = {"total_exposure": 100}
    ready, missing = _preflight_writer(data_store)

    assert ready is False
    # 17 builder_outputs keys minus total_exposure = 16 missing
    assert len(missing) == 16
