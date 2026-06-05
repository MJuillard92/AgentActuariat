"""Tests — helper compute_datacatalogue_state.

Vérifie que le helper agrège correctement les 3 sources de vérité (YAML,
catalogue tools, flag UI) et signale précisément les manquants. Plan
datacatalogue-gate 2026-05-25.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from knowledge_base.report_template.datacatalogue import (  # noqa: E402
    compute_datacatalogue_state,
    DataCatalogueState,
    _is_empty,
)


# ── _is_empty ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value,expected", [
    (None,    True),
    ("",      True),
    ("   ",   True),
    ([],      True),
    ({},      True),
    (set(),   True),
    # Valeurs valides — surtout False qui est une réponse explicite
    (False,   False),
    (0,       False),
    ("ok",    False),
    ([1, 2],  False),
    ({"a": 1}, False),
])
def test_is_empty(value, expected):
    assert _is_empty(value) is expected


# ── Cas extrêmes : data_store vide ───────────────────────────────────────────

def test_empty_data_store_reports_mapping_and_yaml_fields_missing():
    """Sans rien dans data_store, on doit voir au minimum mapping_validated
    + les champs YAML avec confirm_with_user: true (gender_segmentation
    selon le YAML mortality actuel)."""
    res = compute_datacatalogue_state({})
    assert isinstance(res, DataCatalogueState)
    assert res.complete is False
    assert "mapping_validated" in res.missing
    # YAML mortality_template.yaml : gender_segmentation porte
    # confirm_with_user: true (vérifié dans le YAML).
    assert "gender_segmentation" in res.missing


def test_none_data_store_does_not_crash():
    """Le helper doit accepter None et le traiter comme dict vide."""
    res = compute_datacatalogue_state(None)
    assert res.complete is False
    assert "mapping_validated" in res.missing


# ── Mapping validé sans rien d'autre ─────────────────────────────────────────

def test_mapping_validated_alone_not_enough():
    """mapping_validated=True ne suffit pas — il manque encore gender et
    potentiellement les méthodes."""
    res = compute_datacatalogue_state({"mapping_validated": True})
    assert res.complete is False
    assert "mapping_validated" not in res.missing
    assert "gender_segmentation" in res.missing


# ── Cas complet : tout est renseigné ─────────────────────────────────────────

def test_all_filled_marks_complete():
    """Mapping + gender + (toutes les) méthodes choisies → complete=True."""
    # On choisit methods_auto pour éviter d'avoir à lister explicitement
    # chaque tool — c'est aussi l'usage typique.
    ds = {
        "mapping_validated": True,
        "report_mode":       "full_report",
        "study_plan": {
            "gender_segmentation":     "unisex",
            "observation_period_years":[2020, 2024],
            "start_year":               2020,
            "end_year":                 2024,
            "num_observation_years":    5,
            "methods_auto":             True,  # délégation explicite au LLM
        },
    }
    res = compute_datacatalogue_state(ds)
    assert res.complete is True, f"missing : {res.missing}"
    assert res.missing == []


def test_methods_auto_skips_method_choices():
    """Quand methods_auto=True, les choix de méthodes ne comptent pas
    comme manquants."""
    ds = {
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
    res = compute_datacatalogue_state(ds)
    method_missing = [m for m in res.missing if m.startswith("methods.")]
    assert method_missing == []


def test_methods_explicit_must_list_all_tools():
    """Sans methods_auto, study_plan.methods doit lister TOUS les tools
    requis par le mode. Sinon les non-listés apparaissent en manquant."""
    ds = {
        "mapping_validated": True,
        "report_mode":       "full_report",
        "study_plan": {
            "gender_segmentation":      "unisex",
            "observation_period_years": [2020, 2024],
            "start_year":               2020,
            "end_year":                 2024,
            "num_observation_years":    5,
            # methods_auto absent + methods incomplet → manquants
            "methods": {"builder.crude_rates": "central"},
        },
    }
    res = compute_datacatalogue_state(ds)
    # Au moins une méthode doit manquer (smoothing et/ou validation selon
    # le catalogue full_report).
    method_missing = [m for m in res.missing if m.startswith("methods.")]
    assert len(method_missing) >= 1, (
        f"toutes les méthodes ne devraient pas être résolues : missing={res.missing}"
    )


# ── state snapshot ───────────────────────────────────────────────────────────

def test_state_snapshot_contains_all_checked_keys():
    """Le state snapshot doit contenir chaque champ vérifié (None ou non),
    pour permettre à l'UI d'afficher l'état complet."""
    res = compute_datacatalogue_state({})
    assert "mapping_validated" in res.state
    assert "gender_segmentation" in res.state
    # mapping_validated False = bien remonté
    assert res.state["mapping_validated"] is False


# ── Robustesse : data_store partiel ──────────────────────────────────────────

def test_partial_data_store_lists_only_real_missing():
    """Avec mapping validé + gender posé, le seul manquant restant doit
    être les méthodes (si pas methods_auto)."""
    ds = {
        "mapping_validated": True,
        "report_mode":       "description",  # pas de méthodes nécessaires
        "study_plan": {
            "gender_segmentation":      "unisex",
            "observation_period_years": [2020, 2024],
            "start_year":               2020,
            "end_year":                 2024,
            "num_observation_years":    5,
        },
    }
    res = compute_datacatalogue_state(ds)
    # En mode description, pas de méthodes → complete possible
    assert "mapping_validated" not in res.missing
    assert "gender_segmentation" not in res.missing
