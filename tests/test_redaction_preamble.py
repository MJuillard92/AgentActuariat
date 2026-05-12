"""Tests US-24 : _04_redaction sur visual_specs Design 3."""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.report.pipeline._04_redaction import _hydrate_visual_spec  # noqa: E402


def test_hydrate_table_reads_sub_path_from_data_store():
    """Le sub-path `segmentations.sexe` doit être résolu dans le data_store."""
    spec = {
        "id": "portfolio_composition",
        "type": "table",
        "source": "segmentations.sexe",
        "columns": [
            {"key": "valeur",      "label": "Sexe"},
            {"key": "nb_contrats", "label": "Vies"},
            {"key": "nb_deces",    "label": "Décès"},
        ],
    }
    data_store = {
        "segmentations": {
            "sexe": [
                {"valeur": "H", "nb_contrats": 500, "nb_deces": 25},
                {"valeur": "F", "nb_contrats": 500, "nb_deces": 17},
            ],
            "produit": [{"valeur": "A", "nb_contrats": 1000, "nb_deces": 42}],
        },
    }

    out = _hydrate_visual_spec(spec, data_store)

    assert out["type"] == "table"
    assert out["headers"] == ["Sexe", "Vies", "Décès"]
    # Les cellules sont désormais formattées en strings via _format_cell.
    # Les colonnes sans format explicite reçoivent le format depuis
    # `formats.defaults` du YAML — `nb_contrats` et `nb_deces` sont `int`.
    assert out["rows"] == [
        ["H", "500", "25"],
        ["F", "500", "17"],
    ]


def test_hydrate_chart_reads_direct_from_data_store():
    spec = {
        "id": "deaths_per_year",
        "type": "chart",
        "chart_type": "bar",
        "source": "serie",
        "x_axis": {"key": "annee",    "label": "Année"},
        "y_axis": {"key": "nb_deces", "label": "Décès"},
    }
    data_store = {
        "serie": [
            {"annee": 2019, "nb_deces": 10},
            {"annee": 2020, "nb_deces": 15},
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
