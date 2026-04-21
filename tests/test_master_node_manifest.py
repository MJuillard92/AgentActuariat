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
    # Les 4 clés preamble doivent rester présentes (sous-ensemble) :
    assert {
        "total_exposure_years",
        "total_deaths",
        "portfolio_composition_by_sex",
        "deaths_by_year_series",
    }.issubset(set(keys))


def test_preflight_writer_ready_when_all_manifest_keys_present():
    from agents.mortality.agents.master_node import _preflight_writer

    data_store = {
        "total_exposure_years":          1234.5,
        "total_deaths":                  42,
        "portfolio_composition_by_sex":  [{"sexe": "H"}],
        "deaths_by_year_series":         [{"year": 2020, "deaths": 10}],
    }

    ready, missing = _preflight_writer(data_store)

    assert ready is True
    assert missing == []


def test_preflight_writer_missing_keys():
    from agents.mortality.agents.master_node import _preflight_writer

    data_store = {"total_exposure_years": 100}
    ready, missing = _preflight_writer(data_store)

    assert ready is False
    assert len(missing) == 3
