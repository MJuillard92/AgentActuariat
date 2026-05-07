"""Tests : execute_tools doit alimenter les bonnes clés data_store.

Régression détectée par code review :
  - `analysis_plots.py:211` lit `data["segmentation"]` (singulier).
  - Le YAML du Writer attend `data_store["segmentations"]` (pluriel).
  → Les deux doivent être présentes après execute_tools.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

from langchain_core.messages import AIMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _build_state_with_segmentation_call() -> dict:
    """Construit un state où le LLM a émis un tool_call segmentation."""
    tool_call = {
        "id":   "tc-1",
        "name": "statistical_analysis",
        "args": {
            "function_name": "segmentation",
            "params":        {"columns": ["sexe"]},
        },
    }
    msg = AIMessage(content="", tool_calls=[tool_call])
    return {
        "messages":    [msg],
        "data_store":  {},
        "dataset_ref": None,
    }


def test_segmentation_populates_both_singular_and_plural_keys(monkeypatch):
    """Les consommateurs aval lisent soit `segmentation` (graphs.analysis_plots)
    soit `segmentations` (YAML Writer). Les deux clés doivent être alimentées."""
    from agents.mortality.agents import tools_node as tn

    fake_result = {
        "total_contrats": 1000,
        "total_deces":    31,
        "segmentations":  {"sexe": [{"valeur": "H", "nb_contrats": 485}]},
    }

    def _fake_call_tool(**kwargs):
        return fake_result

    monkeypatch.setattr(tn, "call_tool", _fake_call_tool)

    state = _build_state_with_segmentation_call()
    out = tn.execute_tools(state)

    ds = out.get("data_store") or {}
    # Pluriel (YAML Writer)
    assert ds.get("segmentations") == fake_result["segmentations"], (
        f"segmentations (pluriel) manquante : {ds.get('segmentations')!r}"
    )
    # Singulier (graphs.analysis_plots backward-compat)
    assert ds.get("segmentation") is not None, (
        "segmentation (singulier, lu par graphs.analysis_plots) manquante !"
    )
