"""Tests pour tools/aggregation/rules.py (US-21)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.aggregation.rules import run  # noqa: E402


@pytest.fixture
def age_table():
    return pd.DataFrame({
        "age":      list(range(20, 30)),
        "exposure": [100, 200, 300, 400, 500, 400, 300, 200, 100, 50],
        "deaths":   [1,   2,   3,   4,   5,   4,   3,   2,   1,  1],
    })


def test_none_rule_returns_source_unchanged(age_table):
    out = run({"source": age_table}, {"rule": "none"})
    pd.testing.assert_frame_equal(out["aggregated"], age_table)


def test_fixed_width_aggregation(age_table):
    out = run(
        {"source": age_table},
        {"rule": "fixed_width", "params": {"width": 5, "bucket_col": "age"}},
    )
    df = out["aggregated"]
    assert len(df) == 2
    assert df["exposure"].sum() == age_table["exposure"].sum()
    assert df["deaths"].sum() == age_table["deaths"].sum()


def test_equal_count_rule(age_table):
    out = run(
        {"source": age_table},
        {"rule": "equal_count", "params": {"n_buckets": 5, "bucket_col": "age"}},
    )
    df = out["aggregated"]
    assert len(df) == 5


def test_exposure_share_min_rule(age_table):
    out = run(
        {"source": age_table},
        {"rule": "exposure_share_min", "params": {"min_share": 0.10, "bucket_col": "age", "weight_col": "exposure"}},
    )
    df = out["aggregated"]
    total = age_table["exposure"].sum()
    assert all(df["exposure"] / total >= 0.10 - 1e-9)


def test_unknown_rule_raises(age_table):
    with pytest.raises(ValueError):
        run({"source": age_table}, {"rule": "bogus"})


def test_contract_discoverable_by_registry():
    from knowledge_base.report_template.tool_registry import build_registry
    registry = build_registry(_PROJECT_ROOT / "tools")
    assert "aggregation.rules" in registry
    spec = registry["aggregation.rules"]
    assert "source" in spec["inputs"]
    assert "aggregated" in spec["outputs"]
