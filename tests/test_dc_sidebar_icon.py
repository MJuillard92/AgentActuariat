"""Tests — Dérivation des 3 états de l'icône sidebar « Data catalogue ».

Plan refonte garde-fou 2026-06-03 (Partie A.2).

La fonction `_derive_dc_state` vit dans `canvas_app.py`. On l'importe
directement et on teste les trois branches :
  1. bulle inline ouverte → « à compléter » (warning)
  2. datacatalogue complet → « prêt » (success)
  3. sinon → « pas requis » (muted)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(scope="module")
def _derive_dc_state():
    """Import différé : canvas_app charge Dash, on isole le helper sans
    instancier le serveur."""
    import importlib
    mod = importlib.import_module("canvas_app")
    return mod._derive_dc_state


def test_dormant_when_no_history_and_no_mapping(_derive_dc_state):
    """Cas A — data_store vide, history vide → « pas requis »."""
    cls, label = _derive_dc_state({}, [])
    assert "text-muted" in cls
    assert "pas requis" in label


def test_to_complete_when_open_bubble(_derive_dc_state):
    """Cas B — bulle inline ouverte → « à compléter » (priorité 1)."""
    history = [{"role": "_datacatalogue_form",
                "form_id": "dc-1", "submitted": False}]
    cls, label = _derive_dc_state({}, history)
    assert "text-warning" in cls
    assert "à compléter" in label


def test_ready_when_datacatalogue_complete(_derive_dc_state):
    """Cas C — tous les prérequis remplis → « prêt »."""
    data_store = {
        "mapping_validated": True,
        "report_mode":       "full_report",
        "study_plan": {
            "gender_segmentation":      "unisex",
            "observation_period_years": [2020, 2024],
            "start_year":               2020,
            "end_year":                 2024,
            "num_observation_years":    5,
            "methods_auto":             True,
        },
    }
    cls, label = _derive_dc_state(data_store, [])
    assert "text-success" in cls
    assert "prêt" in label


def test_submitted_bubble_does_not_count_as_open(_derive_dc_state):
    """Bulle déjà soumise → on retombe sur le helper datacatalogue (et
    donc sur « pas requis » si data_store vide)."""
    history = [{"role": "_datacatalogue_form",
                "form_id": "dc-1", "submitted": True}]
    cls, label = _derive_dc_state({}, history)
    assert "text-muted" in cls
    assert "pas requis" in label
