"""Tests — Gate datacatalogue dans builder_node.

Vérifie que le Builder refuse net si le data catalogue n'est pas complet,
et qu'il passe normalement quand tout est renseigné. Plan
datacatalogue-gate 2026-05-25.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _state(data_store: dict, dataset_ref: str | None = None) -> dict:
    return {
        "messages":    [],
        "data_store":  data_store,
        "dataset_ref": dataset_ref,
    }


# ── Cas refus : datacatalogue incomplet ──────────────────────────────────────

def test_builder_refuses_when_mapping_not_validated():
    """data_store vide → mapping_validated absent → refus net.

    Le refus NE DOIT PAS poser active_agent='master' (qui causait une
    boucle infinie Master ↔ Builder). On laisse LangGraph terminer via
    le AIMessage sans tool_calls. Fix 2026-05-28."""
    from agents.mortality.agents.builder_node import builder_node

    result = builder_node(_state(data_store={}))

    assert result.get("active_agent") != "master", (
        "active_agent=master cause une boucle infinie"
    )
    events = result.get("events") or []
    types = [e.get("type") for e in events]
    assert "datacatalogue_incomplete" in types
    missing_event = next(e for e in events if e.get("type") == "datacatalogue_incomplete")
    assert "mapping_validated" in missing_event.get("missing", [])


def test_builder_refuses_when_gender_missing():
    """Mapping validé mais gender_segmentation manquant → refus net."""
    from agents.mortality.agents.builder_node import builder_node

    ds = {
        "mapping_validated": True,
        "report_mode":       "full_report",
    }
    result = builder_node(_state(data_store=ds))

    assert result.get("active_agent") != "master"
    missing_event = next(e for e in result["events"]
                         if e.get("type") == "datacatalogue_incomplete")
    assert "gender_segmentation" in missing_event["missing"]


def test_builder_refusal_message_is_user_friendly():
    """Le message de refus doit citer les manquants et orienter l'user
    vers le bouton sidebar."""
    from agents.mortality.agents.builder_node import builder_node

    result = builder_node(_state(data_store={}))

    msg = (result.get("messages") or [None])[0]
    assert msg is not None
    content = msg.content
    # Le message doit inviter explicitement à remplir le formulaire inline
    # (la bulle apparaît en-dessous du message dans le chat).
    assert "formulaire" in content.lower()
    assert "mapping_validated" in content  # champs manquants listés


def test_builder_refuses_returns_done_event():
    """Le refus émet un événement done — le Master ne doit pas re-router
    indéfiniment vers le Builder."""
    from agents.mortality.agents.builder_node import builder_node

    result = builder_node(_state(data_store={}))

    events = result.get("events") or []
    assert any(e.get("type") == "done" for e in events)


# ── Cas passage : datacatalogue complet ──────────────────────────────────────

def _complete_data_store() -> dict:
    return {
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


def test_builder_does_not_emit_gate_event_when_complete(monkeypatch):
    """Quand le datacatalogue est complet, builder_node ne doit PAS émettre
    datacatalogue_incomplete et doit poursuivre son flux normal."""
    from agents.mortality.agents import builder_node as bn

    # Mock OpenAI pour éviter un vrai appel API
    class _FakeClient:
        def __init__(self, *args, **kwargs):
            self.chat = self
            self.completions = self
        def create(self, *args, **kwargs):
            class _Resp:
                class _Choice:
                    finish_reason = "stop"
                    class _Msg:
                        content = "ok"
                        tool_calls = None
                    message = _Msg()
                choices = [_Choice()]
                usage = type("U", (), {"prompt_tokens": 0,
                                       "completion_tokens": 0,
                                       "total_tokens": 0})()
            return _Resp()

    monkeypatch.setattr("openai.OpenAI", _FakeClient)

    result = bn.builder_node(_state(data_store=_complete_data_store(),
                                     dataset_ref="test_session"))

    events = result.get("events") or []
    gate_events = [e for e in events if e.get("type") == "datacatalogue_incomplete"]
    assert gate_events == [], (
        f"Gate ne devrait pas émettre incomplete : {gate_events}"
    )
