"""Tests US-13 : détection du stage value_mapping dans la désambiguation.

Scénarios couverts :
  - records avec valeurs non conformes → suggestion renvoyée
  - records avec valeurs inconnues → blocage (status=unclear)
  - records déjà conformes → stage skip
  - intégration dans run_disambiguation quand column_mapping_confirmed=True
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.master.disambiguation import (  # noqa: E402
    detect_value_mapping_stage,
    run_disambiguation,
)


# ───────────────── detect_value_mapping_stage (pure) ─────────────────

def test_detect_needs_mapping_when_nonconformant():
    df = pd.DataFrame({
        "cause_sortie": ["décédé", "vivant", "décédé"],
        "sexe": ["H", "F", "H"],
    })
    enum_specs = {"cause_sortie": ["deces", "autre"], "sexe": ["H", "F"]}

    out = detect_value_mapping_stage(df, enum_specs)

    assert out["stage"] == "needs_value_mapping"
    assert out["suggestion"]["cause_sortie"]["décédé"] == "deces"
    assert out["suggestion"]["cause_sortie"]["vivant"] == "autre"
    assert out["unmapped"] == {}


def test_detect_blocked_on_unmapped_values():
    df = pd.DataFrame({"cause_sortie": ["XXX", "deces"]})
    enum_specs = {"cause_sortie": ["deces", "autre"]}

    out = detect_value_mapping_stage(df, enum_specs)

    assert out["stage"] == "blocked"
    assert "XXX" in out["unmapped"]["cause_sortie"]
    assert "XXX" in out["message"]


def test_detect_skip_when_all_conformant():
    df = pd.DataFrame({"cause_sortie": ["deces", "autre"], "sexe": ["H", "F"]})
    enum_specs = {"cause_sortie": ["deces", "autre"], "sexe": ["H", "F"]}

    out = detect_value_mapping_stage(df, enum_specs)

    assert out["stage"] == "skip"
    assert out["suggestion"] == {}
    assert out["unmapped"] == {}


def test_detect_ignores_column_absent_from_records():
    df = pd.DataFrame({"cause_sortie": ["deces"]})  # pas de colonne sexe
    enum_specs = {"cause_sortie": ["deces", "autre"], "sexe": ["H", "F"]}

    out = detect_value_mapping_stage(df, enum_specs)

    assert out["stage"] == "skip"


# ───────────────── run_disambiguation branche value_mapping ─────────────────

def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split")


def test_run_disambiguation_returns_value_mapping_stage():
    """Si column_mapping_confirmed et pas value_mapping_confirmed,
    run_disambiguation doit détecter et renvoyer la suggestion value_mapping."""
    df = pd.DataFrame({
        "date_naissance": ["1980-01-01"],
        "date_entree":    ["2020-01-01"],
        "date_sortie":    ["2021-01-01"],
        "cause_sortie":   ["décédé"],
        "sexe":           ["H"],
    })
    data_store = {
        "column_mapping": {
            "date_naissance": "date_naissance",
            "date_entree":    "date_entree",
            "date_sortie":    "date_sortie",
            "cause_sortie":   "cause_sortie",
            "sexe":           "sexe",
        },
        "column_mapping_confirmed": True,
    }

    out = run_disambiguation(
        "construis une table de mortalité",
        _df_to_json(df),
        data_store,
    )

    assert out["status"] == "needs_input"
    assert out.get("needs_value_mapping") is True
    assert out["value_mapping_suggestion"]["cause_sortie"]["décédé"] == "deces"


def test_run_disambiguation_blocks_on_unmapped():
    df = pd.DataFrame({
        "date_naissance": ["1980-01-01"],
        "date_entree":    ["2020-01-01"],
        "date_sortie":    ["2021-01-01"],
        "cause_sortie":   ["XXX"],
        "sexe":           ["H"],
    })
    data_store = {
        "column_mapping": {
            "date_naissance": "date_naissance",
            "date_entree":    "date_entree",
            "date_sortie":    "date_sortie",
            "cause_sortie":   "cause_sortie",
            "sexe":           "sexe",
        },
        "column_mapping_confirmed": True,
    }

    out = run_disambiguation(
        "construis une table de mortalité",
        _df_to_json(df),
        data_store,
    )

    assert out["status"] == "unclear"
    assert "XXX" in out["message"]


def test_run_disambiguation_ready_when_value_mapping_confirmed():
    """Si les deux mappings sont confirmés, disambiguation laisse passer."""
    df = pd.DataFrame({
        "date_naissance": ["1980-01-01"],
        "date_entree":    ["2020-01-01"],
        "date_sortie":    ["2021-01-01"],
        "cause_sortie":   ["deces"],
        "sexe":           ["H"],
    })
    data_store = {
        "column_mapping": {
            "date_naissance": "date_naissance",
            "date_entree":    "date_entree",
            "date_sortie":    "date_sortie",
            "cause_sortie":   "cause_sortie",
            "sexe":           "sexe",
        },
        "column_mapping_confirmed": True,
        "value_mapping_confirmed": True,
    }

    out = run_disambiguation(
        "construis une table de mortalité",
        _df_to_json(df),
        data_store,
    )

    assert out["status"] == "ready"
