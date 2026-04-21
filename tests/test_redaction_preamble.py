"""Tests US-24 : _04_redaction sur visual_specs Design 3."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.report.pipeline._04_redaction import _hydrate_visual_spec  # noqa: E402


def test_hydrate_table_reads_direct_from_data_store():
    spec = {
        "id": "portfolio_composition",
        "type": "table",
        "source": "portfolio_composition_by_sex",
        "columns": [
            {"key": "sexe",     "label": "Sexe"},
            {"key": "n_lives",  "label": "Vies"},
            {"key": "exposure", "label": "Exposition"},
            {"key": "deaths",   "label": "Décès"},
        ],
    }
    data_store = {
        "portfolio_composition_by_sex": [
            {"sexe": "H", "n_lives": 500, "exposure": 700.0, "deaths": 25},
            {"sexe": "F", "n_lives": 500, "exposure": 534.5, "deaths": 17},
        ],
    }

    out = _hydrate_visual_spec(spec, data_store)

    assert out["type"] == "table"
    assert out["headers"] == ["Sexe", "Vies", "Exposition", "Décès"]
    assert out["rows"] == [
        ["H", 500, 700.0, 25],
        ["F", 500, 534.5, 17],
    ]


def test_hydrate_chart_reads_direct_from_data_store():
    spec = {
        "id": "deaths_per_year",
        "type": "chart",
        "chart_type": "bar",
        "source": "deaths_by_year_series",
        "x_axis": {"key": "year",   "label": "Année"},
        "y_axis": {"key": "deaths", "label": "Décès"},
    }
    data_store = {
        "deaths_by_year_series": [
            {"year": 2019, "deaths": 10},
            {"year": 2020, "deaths": 15},
        ],
    }

    out = _hydrate_visual_spec(spec, data_store)

    assert out["type"] == "chart"
    assert out["chart_type"] == "bar"
    assert out["x_values"] == [2019, 2020]
    assert out["y_values"] == [10, 15]
    assert out["x_label"] == "Année"
    assert out["y_label"] == "Décès"


def test_hydrate_missing_source_returns_error_marker():
    spec = {"id": "foo", "type": "table", "source": "absent", "columns": []}
    out = _hydrate_visual_spec(spec, {})
    assert out["error"] is not None
