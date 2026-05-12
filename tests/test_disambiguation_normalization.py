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


def test_applies_column_and_value_mappings(tmp_path, monkeypatch):
    """Format column_mapping = {canonical: csv_col} (cf. build_mapping_report).
    Le DataFrame normalisé est écrit sur disque (Parquet) — on le relit
    pour vérifier le contenu (pas dans data_store, qui contiendrait un
    DataFrame non sérialisable msgpack)."""
    import pandas as _pd
    from session import dataset_store as _ds
    monkeypatch.setattr(_ds, "_ARTIFACTS_DIR", tmp_path)

    df = pd.DataFrame({
        "cause": ["décédé", "vivant"],
        "g":     ["Homme", "Femme"],
    })
    data_store = {
        "column_mapping":           {"cause_sortie": "cause", "sexe": "g"},
        "column_mapping_confirmed": True,
        "value_mapping": {
            "cause_sortie": {"décédé": "deces", "vivant": "autre"},
            "sexe":         {"Homme": "H", "Femme": "F"},
        },
        "value_mapping_confirmed": True,
    }

    out = maybe_normalize_records(
        data_store, _df_to_json(df), dataset_ref="test_session",
    )

    assert out is not None
    assert out["records_normalized"] is True
    # Le Parquet normalisé est écrit sur disque
    norm_path = out.get("dataset_ref_normalized")
    assert norm_path is not None, "dataset_ref_normalized absent du retour"
    normalized = _pd.read_parquet(norm_path)
    assert list(normalized.columns) == ["cause_sortie", "sexe"]
    assert list(normalized["cause_sortie"]) == ["deces", "autre"]
    assert list(normalized["sexe"]) == ["H", "F"]


def test_audit_trace_written():
    df = pd.DataFrame({"cause": ["décédé"]})
    data_store = {
        "column_mapping":           {"cause_sortie": "cause"},
        "column_mapping_confirmed": True,
        "value_mapping":            {"cause_sortie": {"décédé": "deces"}},
        "value_mapping_confirmed":  True,
    }

    out = maybe_normalize_records(data_store, _df_to_json(df))

    audit = out["_audit"]["normalization"]
    assert audit["column_mapping"] == {"cause_sortie": "cause"}
    assert audit["value_mapping"]  == {"cause_sortie": {"décédé": "deces"}}
    assert audit["rows_in"]  == 1
    assert audit["rows_out"] == 1
