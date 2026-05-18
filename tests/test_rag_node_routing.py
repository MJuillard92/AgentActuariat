"""Tests de l'adapter LangGraph rag_node.

On vérifie :
 - le signal `<RAG_DONE>` est bien émis en fin de message
 - `active_agent="master"` dans le return (retour au superviseur)
 - les stage_events sont poussés dans `data_store["_stage_buffer"]`
 - les events `agent_switch` + `message` sont émis
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage


def _state(text: str, data_store: dict | None = None) -> dict:
    return {
        "messages": [HumanMessage(content=text)],
        "data_store": data_store or {},
        "active_agent": "rag",
    }


def _mock_pipeline_result() -> dict:
    return {
        "answer":  "Le lissage Whittaker pénalise les différences [D03.02].\n\n"
                   "Sources :\n- D03.02 — Whittaker-Henderson 1D",
        "sources": [
            {"doc_id": "D03", "section_id": "D03.02",
             "section_title": "Whittaker-Henderson 1D", "text": "..."},
        ],
        "stage_events": [
            ("RAG.1", "Question extraite"),
            ("RAG.2", "Aucune typo détectée"),
            ("RAG.4", "Retrieval hybride : 1 chunks"),
            ("RAG.5", "Synthèse rédigée avec citations"),
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# Signal et routing
# ──────────────────────────────────────────────────────────────────────

def test_rag_node_emits_rag_done_signal():
    from agents.mortality.agents.rag_node import rag_node

    with patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               return_value=_mock_pipeline_result()):
        result = rag_node(_state("c'est quoi whittaker ?"))

    msgs = result["messages"]
    assert len(msgs) == 1
    assert isinstance(msgs[0], AIMessage)
    assert "<RAG_DONE>" in msgs[0].content


def test_rag_node_sets_active_agent_to_master():
    """L'adapter doit rendre la main au Master après son tour unique."""
    from agents.mortality.agents.rag_node import rag_node

    with patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               return_value=_mock_pipeline_result()):
        result = rag_node(_state("question"))

    assert result["active_agent"] == "master"


# ──────────────────────────────────────────────────────────────────────
# Events visibles côté canvas
# ──────────────────────────────────────────────────────────────────────

def test_rag_node_emits_agent_switch_event():
    from agents.mortality.agents.rag_node import rag_node

    with patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               return_value=_mock_pipeline_result()):
        result = rag_node(_state("question"))

    events = result.get("events") or []
    types = [(e.get("type"), e.get("agent")) for e in events]
    assert ("agent_switch", "RAGAgent") in types


def test_rag_node_emits_message_event_without_signal():
    """L'event 'message' (visible chat) doit contenir la réponse SANS le
    signal `<RAG_DONE>` — ce signal est strictement réservé au routage."""
    from agents.mortality.agents.rag_node import rag_node

    with patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               return_value=_mock_pipeline_result()):
        result = rag_node(_state("question"))

    msg_events = [e for e in result["events"] if e.get("type") == "message"]
    assert len(msg_events) == 1
    assert "<RAG_DONE>" not in msg_events[0]["content"]
    assert "[D03.02]" in msg_events[0]["content"]


# ──────────────────────────────────────────────────────────────────────
# Stage tracking → data_store["_stage_buffer"]
# ──────────────────────────────────────────────────────────────────────

def test_rag_node_pushes_stage_events_into_stage_buffer():
    from agents.mortality.agents.rag_node import rag_node

    state = _state("question")
    with patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               return_value=_mock_pipeline_result()):
        result = rag_node(state)

    ds = result.get("data_store") or {}
    buf = ds.get("_stage_buffer") or []
    # 4 stage events doivent être poussés
    rag_stages = [b for b in buf if b.get("type") == "master_stage"
                  and (b.get("stage") or "").startswith("RAG.")]
    assert len(rag_stages) == 4
    stage_ids = [b["stage"] for b in rag_stages]
    assert stage_ids == ["RAG.1", "RAG.2", "RAG.4", "RAG.5"]


def test_rag_node_preserves_existing_stage_buffer():
    """Les stages déjà présents dans le data_store doivent être préservés."""
    from agents.mortality.agents.rag_node import rag_node

    existing = [{"type": "master_stage", "stage": "0.d", "label": "Classification"}]
    state = _state("question", data_store={"_stage_buffer": list(existing)})
    with patch("agents.mortality.agents.rag_node._run_rag_pipeline",
               return_value=_mock_pipeline_result()):
        result = rag_node(state)

    buf = result["data_store"]["_stage_buffer"]
    # L'existant doit être conservé en tête + les 4 RAG.* ajoutés
    assert buf[0]["stage"] == "0.d"
    assert any(b.get("stage") == "RAG.5" for b in buf)
