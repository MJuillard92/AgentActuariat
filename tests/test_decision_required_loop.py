"""Tests — Fix boucle infinie quand un tool retourne `decision_required`.

Bug originel : le garde-fou strippait les tool_calls SANS surfacer la
décision à l'user, le Builder était relancé en boucle.

Fix : émission d'un event `decision_required` + pose d'un verrou
`data_store["_pending_decision"]` qui coupe le graphe sur END (côté
Builder et Master). Plan refonte garde-fou 2026-06-03.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, ToolMessage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ── Helpers d'extraction ─────────────────────────────────────────────────────

def _tool_message_with_decision(reason: str = "8 violations détectées",
                                 method: str = "whittaker") -> ToolMessage:
    payload = {
        "smoothed_table": [{"age": 30, "q_x_lisse": 0.001}],
        "decision_required": {
            "reason":  reason,
            "options": [
                {"id": "increase_lambda",   "label": "Augmenter lambda"},
                {"id": "change_method",     "label": f"Changer (autre que {method})"},
                {"id": "accept_with_note",  "label": "Accepter avec note"},
            ],
        },
    }
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id="tc-1",
        name="builder.smoothing",
    )


# ── Cas 1 : helpers du builder_node détectent et extraient correctement ─────

def test_extract_decision_required_returns_full_payload():
    from agents.mortality.agents.builder_node import (
        _has_pending_decision, _extract_decision_required,
    )
    msg = _tool_message_with_decision("12 violations")
    assert _has_pending_decision([msg]) is True
    dr = _extract_decision_required([msg])
    assert dr["tool"]    == "builder.smoothing"
    assert dr["reason"]  == "12 violations"
    assert len(dr["options"]) == 3
    assert {o["id"] for o in dr["options"]} == {
        "increase_lambda", "change_method", "accept_with_note",
    }


def test_extract_decision_required_returns_empty_when_no_marker():
    from agents.mortality.agents.builder_node import (
        _has_pending_decision, _extract_decision_required,
    )
    msg = ToolMessage(content=json.dumps({"smoothed_table": []}),
                       tool_call_id="tc-1", name="builder.smoothing")
    assert _has_pending_decision([msg]) is False
    dr = _extract_decision_required([msg])
    assert dr["tool"]    == ""
    assert dr["options"] == []


# ── Cas 2 : _should_continue_* coupent quand _pending_decision est posée ────

def test_should_continue_builder_returns_END_when_pending_decision():
    from langgraph.graph import END
    from agents.mortality.agents.graph import _should_continue_builder

    state = {
        "data_store": {"_pending_decision": {"tool": "builder.smoothing"}},
        "messages":   [AIMessage(content="…")],
        "active_agent": "builder",
    }
    assert _should_continue_builder(state) == END


def test_should_continue_master_returns_END_when_pending_decision():
    from langgraph.graph import END
    from agents.mortality.agents.graph import _should_continue_master

    state = {
        "data_store":   {"_pending_decision": {"tool": "builder.smoothing"}},
        "messages":     [AIMessage(content="…")],
        "active_agent": "master",
    }
    assert _should_continue_master(state) == END


# ── Cas 3 : Quand _pending_decision est levée, graphe re-route normalement ─

def test_should_continue_master_routes_after_decision_cleared():
    """Une fois la décision user enregistrée par le callback, le verrou
    est popé du data_store. Master doit pouvoir re-router normalement.
    Le test simule un active_agent='builder' → re-route 'to_builder'."""
    from agents.mortality.agents.graph import _should_continue_master

    state = {
        "data_store":   {},     # plus de verrou
        "messages":     [AIMessage(content="…")],
        "active_agent": "builder",
    }
    assert _should_continue_master(state) == "to_builder"
