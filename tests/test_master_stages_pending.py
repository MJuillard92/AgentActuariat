"""Tests pour Bug 5 — Instrumentation stages dans les 5 paths pending_need.

Régression : les paths P3-P7 du master_node (résolutions/forwards de
questions pendantes) court-circuitent `_classify_intent` et émettent
maintenant `0.d-pending` + `0.e-pending` pour visibilité UI.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage


def _build_state(human_text: str, data_store: dict | None = None) -> dict:
    ds = data_store or {}
    return {
        "messages":   [HumanMessage(content=human_text)],
        "data_store": ds,
        "dataset_ref": None,
    }


# ──────────────────────────────────────────────────────────────────────
# P6 — PENDING_RESOLVED (cas user "unisex" pour gender_segmentation)
# ──────────────────────────────────────────────────────────────────────

def test_pending_resolved_emits_stages():
    """Quand l'user répond à une question pending (gender_segmentation =
    'unisex'), les stages 0.d-pending et 0.e-pending sont émis."""
    from agents.mortality.agents.master_node import master_node

    pending_need = {
        "context_key": "gender_segmentation",
        "question": "unisex ou par sexe ?",
        "options": ["unisex", "by_sex"],
        "default": "unisex",
    }
    state = _build_state("unisex", data_store={"_pending_need": pending_need})

    # Mock extract_user_answer pour qu'il retourne la valeur extraite
    with patch("agents.master.question_filter.extract_user_answer",
               return_value="unisex"):
        result = master_node(state)

    events = result.get("events") or []
    stage_ids = [e.get("stage") for e in events if e.get("type") == "master_stage"]
    assert "0.d-pending" in stage_ids
    assert "0.e-pending" in stage_ids
    # Active_agent reste "builder" (path P6 nominal)
    assert result.get("active_agent") == "builder"


def test_pending_resolved_stage_label_contains_value():
    """Le label de 0.e-pending doit mentionner la valeur résolue."""
    from agents.mortality.agents.master_node import master_node

    pending_need = {
        "context_key": "gender_segmentation",
        "question": "?",
        "options": ["unisex", "by_sex"],
        "default": "unisex",
    }
    state = _build_state("unisex", data_store={"_pending_need": pending_need})

    with patch("agents.master.question_filter.extract_user_answer",
               return_value="unisex"):
        result = master_node(state)

    events = result.get("events") or []
    e_pending = next((e for e in events
                      if e.get("type") == "master_stage" and e.get("stage") == "0.e-pending"),
                     None)
    assert e_pending is not None
    assert "unisex" in e_pending["label"]


# ──────────────────────────────────────────────────────────────────────
# P7 — PENDING_REASK (extract_user_answer renvoie None → re-poser)
# ──────────────────────────────────────────────────────────────────────

def test_pending_reask_emits_stages():
    """Quand extract_user_answer renvoie None, on re-pose la question
    mais on doit émettre quand même les stages pour visibilité."""
    from agents.mortality.agents.master_node import master_node

    pending_need = {
        "context_key": "gender_segmentation",
        "question": "?",
        "options": ["unisex", "by_sex"],
    }
    state = _build_state("réponse ambiguë", data_store={"_pending_need": pending_need})

    with patch("agents.master.question_filter.extract_user_answer",
               return_value=None):
        result = master_node(state)

    events = result.get("events") or []
    stage_ids = [e.get("stage") for e in events if e.get("type") == "master_stage"]
    assert "0.d-pending" in stage_ids
    assert "0.e-pending" in stage_ids
    # Pas de routing builder — on attend une nouvelle réponse user
    assert result.get("active_agent") != "builder"


def test_pending_reask_stage_label_mentions_options():
    """Le label de 0.e-pending P7 doit mentionner les options et le motif."""
    from agents.mortality.agents.master_node import master_node

    pending_need = {
        "context_key": "gender_segmentation",
        "question": "?",
        "options": ["unisex", "by_sex"],
    }
    state = _build_state("blabla", data_store={"_pending_need": pending_need})

    with patch("agents.master.question_filter.extract_user_answer",
               return_value=None):
        result = master_node(state)

    events = result.get("events") or []
    e_pending = next((e for e in events
                      if e.get("type") == "master_stage" and e.get("stage") == "0.e-pending"),
                     None)
    assert e_pending is not None
    # Label doit indiquer extract_user_answer=None ou options
    label = e_pending["label"].lower()
    assert "extract_user_answer" in label or "options" in label or "re-poser" in label


# ──────────────────────────────────────────────────────────────────────
# Régression : comportement nominal préservé
# ──────────────────────────────────────────────────────────────────────

def test_pending_paths_preserve_messages_and_routing():
    """Le contenu messages + active_agent ne change PAS, seuls les stages
    s'ajoutent (régression positive vs comportement antérieur)."""
    from agents.mortality.agents.master_node import master_node

    pending_need = {
        "context_key": "gender_segmentation",
        "question": "?",
        "options": ["unisex", "by_sex"],
        "default": "unisex",
    }
    state = _build_state("unisex", data_store={"_pending_need": pending_need})

    with patch("agents.master.question_filter.extract_user_answer",
               return_value="unisex"):
        result = master_node(state)

    # 1 message synthetic emit (instr "[Master] L'utilisateur a répondu...")
    assert len(result.get("messages", [])) == 1
    msg = result["messages"][0]
    assert "[Master]" in msg.content
    assert "unisex" in msg.content
    # study_plan stocké
    assert result["data_store"]["study_plan"]["gender_segmentation"] == "unisex"
    # pending consommé
    assert "_pending_need" not in result["data_store"]
