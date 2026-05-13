"""Tests du tool conversation.apply_normalization et de la persistance
SessionState des flags de normalisation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _sample_raw_df():
    return pd.DataFrame({
        "CLINAISS":   ["01/01/1950", "01/01/1960", "01/01/1970"],
        "CTREFFET":   ["01/01/2000", "01/01/2005", "01/01/2010"],
        "DATE_SORTIE": ["31/12/2010", "31/12/2999", "31/12/2999"],
        "STATUT":     ["Decede",     "Vivant",     "Vivant"],
        "SEXEREF":    ["1",          "2",          "1"],
    })


# ──────────────────────────────────────────────────────────────────────
# apply_normalization end-to-end
# ──────────────────────────────────────────────────────────────────────

def test_apply_normalization_writes_parquet_and_mutates_data_store(tmp_path, monkeypatch):
    """E2E : appel via le tool met data_store dans le bon état."""
    from session import dataset_store as _ds
    monkeypatch.setattr(_ds, "_ARTIFACTS_DIR", tmp_path)
    from tools.conversation.apply_normalization import run

    df = _sample_raw_df()
    data_store = {"_dataset_ref": "test_e2e"}

    res = run(df, {}, data=data_store)

    assert "erreur" not in res, f"erreur : {res.get('erreur')}"
    assert res["records_normalized"] is True

    # data_store doit être muté avec les flags
    assert data_store.get("column_mapping_confirmed") is True
    assert data_store.get("value_mapping_confirmed") is True
    assert data_store.get("records_normalized") is True
    assert data_store.get("dataset_ref_normalized") is not None

    # Le Parquet propre existe sur disque
    norm_path = Path(data_store["dataset_ref_normalized"])
    assert norm_path.exists(), f"Parquet absent : {norm_path}"

    # Contenu : colonnes canoniques, valeurs mappées, dates datetime64
    norm = pd.read_parquet(norm_path)
    assert {"date_naissance", "date_entree", "date_sortie",
            "cause_sortie", "sexe"} <= set(norm.columns)
    assert pd.api.types.is_datetime64_any_dtype(norm["date_sortie"])
    # Sentinelles 2999 → clippées à 2010 (max décès)
    assert (norm["date_sortie"].dt.year <= 2100).all()
    # Valeurs enum mappées : 1→H, 2→F (convention INSEE) ;
    # Decede→deces, Vivant→autre
    assert set(norm["sexe"]) <= {"H", "F"}
    assert set(norm["cause_sortie"]) <= {"deces", "autre"}


def test_apply_normalization_idempotent_without_force(tmp_path, monkeypatch):
    """Si records_normalized=True déjà, on ne refait pas."""
    from session import dataset_store as _ds
    monkeypatch.setattr(_ds, "_ARTIFACTS_DIR", tmp_path)
    from tools.conversation.apply_normalization import run

    df = _sample_raw_df()
    data_store = {
        "_dataset_ref":           "test_idem",
        "records_normalized":     True,
        "dataset_ref_normalized": "/tmp/existing.parquet",
    }
    res = run(df, {}, data=data_store)
    assert "info" in res or "records_normalized" in res
    # data_store n'a pas été retouché (pas de re-écriture)
    assert data_store["dataset_ref_normalized"] == "/tmp/existing.parquet"


def test_apply_normalization_requires_data_store():
    """L'appel sans data_store retourne une erreur (le tool doit muter
    le state pour persister, sans data_store ça ne sert à rien)."""
    from tools.conversation.apply_normalization import run
    res = run(_sample_raw_df(), {}, data=None)
    assert "erreur" in res


# ──────────────────────────────────────────────────────────────────────
# Persistance SessionState — round-trip
# ──────────────────────────────────────────────────────────────────────

def test_session_state_persists_normalization_fields():
    """Régression : les 5 champs de normalisation doivent survivre à
    un round-trip update_from_data_store → to_data_store. Sans ça, le
    Parquet propre est silencieusement oublié entre les tours (même
    type de bug que methods auparavant)."""
    from session.session_state import SessionState

    ds_in = {
        "_disambiguation_done":     True,
        "column_mapping":           {"date_naissance": "CLINAISS"},
        "column_mapping_confirmed": True,
        "value_mapping":            {"sexe": {"1": "H", "2": "F"}},
        "value_mapping_confirmed":  True,
        "records_normalized":       True,
        "dataset_ref_normalized":   "/tmp/test_normalized.parquet",
        "observation_end":          "2010-12-31T00:00:00",
    }
    state = SessionState(session_id="test_persist")
    state.update_from_data_store(ds_in)

    # Round-trip : to_data_store doit re-exposer ces champs
    ds_out = state.to_data_store()
    for key in (
        "value_mapping", "value_mapping_confirmed",
        "records_normalized", "dataset_ref_normalized",
        "observation_end",
    ):
        assert ds_out.get(key) == ds_in[key], (
            f"Champ '{key}' perdu au round-trip : "
            f"in={ds_in[key]!r}, out={ds_out.get(key)!r}"
        )


def test_session_state_normalization_survives_model_dump():
    """Régression Pydantic : les 5 champs doivent être présents après
    model_dump() — sinon ils sont droppés silencieusement."""
    from session.session_state import SessionState

    state = SessionState(
        session_id="test_dump",
        value_mapping={"sexe": {"1": "H"}},
        value_mapping_confirmed=True,
        records_normalized=True,
        dataset_ref_normalized="/tmp/x.parquet",
        observation_end="2010-12-31",
    )
    dumped = state.model_dump()
    assert dumped["value_mapping"]            == {"sexe": {"1": "H"}}
    assert dumped["value_mapping_confirmed"]  is True
    assert dumped["records_normalized"]       is True
    assert dumped["dataset_ref_normalized"]   == "/tmp/x.parquet"
    assert dumped["observation_end"]          == "2010-12-31"
