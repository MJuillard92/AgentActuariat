"""
HOTFIX-pre-refacto-2026-05 — Bug 14 (Option B-full) : DatasetStore.store()
parse les dates dès la construction du dataset.

Garantit que le parquet stocké contient des colonnes datetime64 — donc
tout tool downstream relit du datetime64, jamais une string ambiguë.
"""
from __future__ import annotations

import pandas as pd
import pytest

from session.dataset_store import DatasetStore, _parse_dates_at_construction


def test_parse_dates_at_construction_raw_column_names() -> None:
    """Colonnes brutes CTREFFET / DATE_SORTIE / CLINAISS détectées via
    COLUMN_SCHEMA et parsées en datetime64."""
    df = pd.DataFrame({
        "CLINAISS":    ["01/01/1950", "15/06/1960"],
        "CTREFFET":    ["28/11/2007", "11/04/2009"],
        "DATE_SORTIE": ["31/12/2015", "31/12/2999"],
        "STATUT":      ["D", "V"],
    })
    out = _parse_dates_at_construction(df)
    assert pd.api.types.is_datetime64_any_dtype(out["CTREFFET"])
    assert pd.api.types.is_datetime64_any_dtype(out["DATE_SORTIE"])
    assert pd.api.types.is_datetime64_any_dtype(out["CLINAISS"])
    # 28/11/2007 lu en dayfirst (28 nov, pas crash %m/%d)
    assert out["CTREFFET"].iloc[0] == pd.Timestamp("2007-11-28")
    # sentinelle clippée
    assert pd.isna(out["DATE_SORTIE"].iloc[1])
    # colonne non-date intacte
    assert list(out["STATUT"]) == ["D", "V"]


def test_parse_dates_no_date_columns_returns_unchanged() -> None:
    """Aucune colonne date reconnue → df inchangé, pas de crash."""
    df = pd.DataFrame({"foo": [1, 2], "bar": ["x", "y"]})
    out = _parse_dates_at_construction(df)
    assert list(out["foo"]) == [1, 2]
    assert list(out["bar"]) == ["x", "y"]


def test_store_persists_datetime64(tmp_path, monkeypatch) -> None:
    """DatasetStore.store() écrit un parquet dont les colonnes date sont
    datetime64 ; rechargé, le dtype est préservé."""
    import session.dataset_store as ds_mod
    monkeypatch.setattr(ds_mod, "_ARTIFACTS_DIR", tmp_path)

    df = pd.DataFrame({
        "CLINAISS":    ["01/01/1950"] * 5,
        "CTREFFET":    ["28/11/2007"] * 5,
        "DATE_SORTIE": ["31/12/2015"] * 5,
        "SEXEREF":     ["1"] * 5,
    })
    meta = ds_mod.DatasetStore.store("sess_test_dates", df)

    reloaded = pd.read_parquet(meta.path)
    assert pd.api.types.is_datetime64_any_dtype(reloaded["CTREFFET"])
    assert reloaded["CTREFFET"].iloc[0] == pd.Timestamp("2007-11-28")


def test_store_is_idempotent(tmp_path, monkeypatch) -> None:
    """store() reste idempotent : 2e appel ne réécrit pas."""
    import session.dataset_store as ds_mod
    monkeypatch.setattr(ds_mod, "_ARTIFACTS_DIR", tmp_path)

    df = pd.DataFrame({"CTREFFET": ["28/11/2007"] * 3, "SEXEREF": ["1"] * 3})
    meta1 = ds_mod.DatasetStore.store("sess_idem", df)
    meta2 = ds_mod.DatasetStore.store("sess_idem", df)
    assert meta1.path == meta2.path
    assert meta1.sha256 == meta2.sha256
