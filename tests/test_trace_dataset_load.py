"""
HOTFIX-pre-refacto-2026-05 — Bug 15 (A1) : execute_tools émet un stage
'0.load' annonçant le chargement de la base de données.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def _make_state(tool_calls=None):
    """État LangGraph minimal avec un AIMessage portant des tool_calls."""
    msg = MagicMock()
    msg.tool_calls = tool_calls or []
    return {
        "messages":    [msg],
        "dataset_ref": "sess_load_test",
        "data_store":  {"_dataset_ref": "sess_load_test"},
    }


def test_stage_0load_emitted_when_df_loaded() -> None:
    """Quand un DataFrame est chargé, un stage 0.load doit être émis."""
    from agents.mortality.agents import tools_node as tn

    fake_df = pd.DataFrame({"date_naissance": ["01/01/1950"] * 100,
                            "sexe": ["H"] * 100})

    fake_mm = MagicMock()
    fake_mm.load.return_value = fake_mm
    fake_mm.load_dataframe.return_value = fake_df

    with patch("session.memory_manager.MemoryManager", return_value=fake_mm):
        result = tn.execute_tools(_make_state(tool_calls=[]))

    stages = [e for e in result.get("events", [])
              if e.get("type") == "master_stage" and e.get("stage") == "0.load"]
    assert len(stages) == 1, f"stage 0.load manquant : {result.get('events')}"
    label = stages[0]["label"]
    assert "Base de données chargée" in label
    assert "100 lignes" in label
    assert "source : originale" in label


def test_stage_0load_mentions_normalized_source(tmp_path) -> None:
    """Si le parquet normalisé est utilisé, le stage mentionne 'normalisée'."""
    from agents.mortality.agents import tools_node as tn

    norm_file = tmp_path / "norm.parquet"
    pd.DataFrame({"sexe": ["H"] * 50}).to_parquet(norm_file)

    state = {
        "messages":    [MagicMock(tool_calls=[])],
        "dataset_ref": "sess_x",
        "data_store":  {"dataset_ref_normalized": str(norm_file)},
    }
    result = tn.execute_tools(state)

    stages = [e for e in result.get("events", [])
              if e.get("stage") == "0.load"]
    assert len(stages) == 1
    assert "source : normalisée" in stages[0]["label"]


def test_no_stage_0load_when_no_dataset() -> None:
    """Sans dataset chargeable, pas de stage 0.load (pas de crash)."""
    from agents.mortality.agents import tools_node as tn

    fake_mm = MagicMock()
    fake_mm.load.return_value = fake_mm
    fake_mm.load_dataframe.return_value = None

    with patch("session.memory_manager.MemoryManager", return_value=fake_mm):
        result = tn.execute_tools(_make_state(tool_calls=[]))

    stages = [e for e in result.get("events", [])
              if e.get("stage") == "0.load"]
    assert len(stages) == 0
