"""Tests US-14 : déclenchement de normalize_records par le Master.

L'étape de normalisation s'exécute quand :
  - column_mapping_confirmed=True
  - value_mapping_confirmed=True
  - records_normalized != True
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.master.disambiguation import maybe_normalize_records  # noqa: E402


def _df_to_json(df: pd.DataFrame) -> str:
    return df.to_json(orient="split")


def test_no_op_when_not_fully_confirmed():
    df = pd.DataFrame({"cause": ["décédé"]})
    data_store = {"column_mapping_confirmed": True}  # value_mapping manquant

    out = maybe_normalize_records(data_store, _df_to_json(df))

    assert out is None


def test_no_op_when_already_normalized():
    df = pd.DataFrame({"cause": ["décédé"]})
    data_store = {
        "column_mapping_confirmed": True,
        "value_mapping_confirmed":  True,
        "records_normalized":       True,
    }

    out = maybe_normalize_records(data_store, _df_to_json(df))

    assert out is None


def test_applies_column_and_value_mappings():
    df = pd.DataFrame({
        "cause": ["décédé", "vivant"],
        "g":     ["Homme", "Femme"],
    })
    data_store = {
        "column_mapping":           {"cause": "cause_sortie", "g": "sexe"},
        "column_mapping_confirmed": True,
        "value_mapping": {
            "cause_sortie": {"décédé": "deces", "vivant": "autre"},
            "sexe":         {"Homme": "H", "Femme": "F"},
        },
        "value_mapping_confirmed": True,
    }

    out = maybe_normalize_records(data_store, _df_to_json(df))

    assert out is not None
    normalized = out["input_records"]
    assert list(normalized.columns) == ["cause_sortie", "sexe"]
    assert list(normalized["cause_sortie"]) == ["deces", "autre"]
    assert list(normalized["sexe"]) == ["H", "F"]
    assert out["records_normalized"] is True


def test_audit_trace_written():
    df = pd.DataFrame({"cause": ["décédé"]})
    data_store = {
        "column_mapping":           {"cause": "cause_sortie"},
        "column_mapping_confirmed": True,
        "value_mapping":            {"cause_sortie": {"décédé": "deces"}},
        "value_mapping_confirmed":  True,
    }

    out = maybe_normalize_records(data_store, _df_to_json(df))

    audit = out["_audit"]["normalization"]
    assert audit["column_mapping"] == {"cause": "cause_sortie"}
    assert audit["value_mapping"]  == {"cause_sortie": {"décédé": "deces"}}
    assert audit["rows_in"]  == 1
    assert audit["rows_out"] == 1
