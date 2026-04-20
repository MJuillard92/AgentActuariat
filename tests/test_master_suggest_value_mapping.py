"""Tests pour tools/master/suggest_value_mapping.py (US-11)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.master.suggest_value_mapping import run  # noqa: E402


def test_maps_french_death_synonyms():
    df = pd.DataFrame({"cause_sortie": ["decede", "Vivant", "Décès", "autre"]})
    out = run(
        {"records": df, "enum_specs": {"cause_sortie": ["deces", "autre"]}},
        {},
    )
    m = out["value_mapping"]["cause_sortie"]
    assert m["decede"] == "deces"
    assert m["Décès"] == "deces"
    assert m["Vivant"] == "autre"


def test_maps_sexe():
    df = pd.DataFrame({"sexe": ["Homme", "Femme", "M", "F"]})
    out = run(
        {"records": df, "enum_specs": {"sexe": ["H", "F"]}},
        {},
    )
    m = out["value_mapping"]["sexe"]
    assert m["Homme"] == "H"
    assert m["M"] == "H"
    assert m["Femme"] == "F"


def test_conformant_values_passthrough():
    df = pd.DataFrame({"sexe": ["H", "F"]})
    out = run(
        {"records": df, "enum_specs": {"sexe": ["H", "F"]}},
        {},
    )
    assert out["value_mapping"]["sexe"] == {}
    assert out["unmapped"]["sexe"] == []


def test_unknown_values_reported_as_unmapped():
    df = pd.DataFrame({"sexe": ["alien", "H"]})
    out = run(
        {"records": df, "enum_specs": {"sexe": ["H", "F"]}},
        {},
    )
    assert "alien" in out["unmapped"]["sexe"]


def test_does_not_mutate_input():
    df = pd.DataFrame({"sexe": ["Homme"]})
    before = df.copy()
    run({"records": df, "enum_specs": {"sexe": ["H", "F"]}}, {})
    pd.testing.assert_frame_equal(df, before)


def test_contract_discoverable_by_registry():
    from knowledge_base.report_template.tool_registry import build_registry
    registry = build_registry(_PROJECT_ROOT / "tools")
    assert "master.suggest_value_mapping" in registry
    spec = registry["master.suggest_value_mapping"]
    assert "records" in spec["inputs"]
    assert "enum_specs" in spec["inputs"]
    assert "value_mapping" in spec["outputs"]
    assert "unmapped" in spec["outputs"]
