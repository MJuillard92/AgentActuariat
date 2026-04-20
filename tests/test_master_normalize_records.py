"""Tests pour tools/master/normalize_records.py (US-12)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.master.normalize_records import run  # noqa: E402


def test_renames_columns():
    df = pd.DataFrame({"dt_sortie": ["2020-01-01"], "cause": ["deces"]})
    out = run(
        {
            "records": df,
            "column_mapping": {"dt_sortie": "date_sortie", "cause": "cause_sortie"},
            "value_mapping": {},
        },
        {},
    )
    cols = set(out["normalized_records"].columns)
    assert "date_sortie" in cols
    assert "cause_sortie" in cols
    assert "dt_sortie" not in cols


def test_substitutes_enum_values():
    df = pd.DataFrame({"sexe": ["Homme", "Femme", "Homme"]})
    out = run(
        {
            "records": df,
            "column_mapping": {},
            "value_mapping": {"sexe": {"Homme": "H", "Femme": "F"}},
        },
        {},
    )
    assert list(out["normalized_records"]["sexe"]) == ["H", "F", "H"]


def test_unmapped_columns_preserved():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = run(
        {"records": df, "column_mapping": {"a": "alpha"}, "value_mapping": {}},
        {},
    )
    cols = set(out["normalized_records"].columns)
    assert cols == {"alpha", "b"}


def test_original_not_mutated():
    df = pd.DataFrame({"sexe": ["Homme"]})
    before = df.copy()
    run(
        {
            "records": df,
            "column_mapping": {},
            "value_mapping": {"sexe": {"Homme": "H"}},
        },
        {},
    )
    pd.testing.assert_frame_equal(df, before)


def test_combined_rename_and_substitute():
    df = pd.DataFrame({"cause": ["decede", "vivant"]})
    out = run(
        {
            "records": df,
            "column_mapping": {"cause": "cause_sortie"},
            "value_mapping": {"cause_sortie": {"decede": "deces", "vivant": "autre"}},
        },
        {},
    )
    assert list(out["normalized_records"]["cause_sortie"]) == ["deces", "autre"]


def test_contract_discoverable_by_registry():
    from knowledge_base.report_template.tool_registry import build_registry
    registry = build_registry(_PROJECT_ROOT / "tools")
    assert "master.normalize_records" in registry
    spec = registry["master.normalize_records"]
    for k in ("records", "column_mapping", "value_mapping"):
        assert k in spec["inputs"]
    assert "normalized_records" in spec["outputs"]
