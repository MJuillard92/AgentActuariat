"""Tests : Master maintient `data_store["_user_messages"]` avec uniquement
les messages user (pas les synthétiques émis par lui-même)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _fake_classify(*a, **kw):
    return {
        "kind": "task", "write": "yes", "report_mode": "full_report",
        "intent": "build_and_write", "reply": "",
    }


def test_master_stores_first_user_message():
    from agents.mortality.agents import master_node as mn
    state = {
        "messages":    [HumanMessage(content="construis-moi une table")],
        "data_store":  {
            "_disambiguation_done": True,
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    user_msgs = out["data_store"].get("_user_messages") or []
    assert "construis-moi une table" in user_msgs


def test_master_does_not_store_synthetic_messages():
    """Les HumanMessages avec source=master_synthetic ne sont PAS dans _user_messages."""
    from agents.mortality.agents import master_node as mn
    synthetic = HumanMessage(
        content="Mode de rapport : full_report\nSections actives : [...]",
        additional_kwargs={"source": "master_synthetic"},
    )
    real = HumanMessage(content="construit avec un lissage doux")
    state = {
        "messages":    [synthetic, real],
        "data_store":  {
            "_disambiguation_done": True,
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    user_msgs = out["data_store"].get("_user_messages") or []
    assert "construit avec un lissage doux" in user_msgs
    assert all("Mode de rapport" not in m for m in user_msgs)


def test_master_accumulates_multiple_user_messages():
    from agents.mortality.agents import master_node as mn
    state = {
        "messages":    [HumanMessage(content="bonjour")],
        "data_store":  {
            "_disambiguation_done": True,
            "_user_messages":       ["fais-moi un rapport"],
            "study_plan":           {"gender_segmentation": "unisex"},
        },
        "dataset_ref": None,
    }
    with patch.object(mn, "_classify_intent", _fake_classify):
        out = mn.master_node(state)
    user_msgs = out["data_store"].get("_user_messages") or []
    assert "fais-moi un rapport" in user_msgs
    assert "bonjour" in user_msgs
