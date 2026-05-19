"""Tests pour Bug 6 — Garde dataset_ref avant routing Builder.

Régression : sans CSV chargé, le Master ne doit PAS router vers
BuilderAgent (qui hallucinerait les colonnes).
"""
from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import HumanMessage


def _state(human: str, dataset_ref=None, ds=None):
    return {
        "messages":    [HumanMessage(content=human)],
        "data_store":  ds or {},
        "dataset_ref": dataset_ref,
    }


def _fake_classify(*args, **kwargs):
    """Mock classify_intent : retourne build_only sans LLM."""
    return {
        "kind":        "task",
        "write":       "no",
        "intent":      "build_only",
        "report_mode": "full_report",
        "confidence":  0.95,
        "reply":       "",
    }


def test_calcul_without_dataset_refuses_politely():
    """Sans dataset chargé, master refuse poliment au lieu de router Builder."""
    from agents.mortality.agents.master_node import master_node

    # `_disambiguation_done=True` pour bypasser le bloc disam (qui dépend
    # d'un dataset chargé). Le but du test = vérifier la GARDE Bug 6,
    # pas la disam.
    state = _state(
        "calcule la table de mortalité",
        ds={"_disambiguation_done": True},
    )

    with patch("agents.mortality.agents.master_node._classify_intent",
               side_effect=_fake_classify):
        result = master_node(state)

    # Pas de routing vers Builder
    assert result.get("active_agent") != "builder"
    # Message de refus poli
    msgs = result.get("messages", [])
    assert len(msgs) >= 1
    content = msgs[0].content
    assert "csv" in content.lower() or "fichier" in content.lower()
    assert "portefeuille" in content.lower() or "table" in content.lower()
    # Stage 0.e dédié au refus
    events = result.get("events") or []
    stage_labels = [e.get("label", "") for e in events
                    if e.get("type") == "master_stage"]
    assert any("refus" in label.lower() for label in stage_labels), (
        f"Stage refus absent dans : {stage_labels}"
    )


def test_calcul_with_dataset_routes_to_builder():
    """Régression positive : avec dataset chargé, comportement nominal préservé."""
    from agents.mortality.agents.master_node import master_node

    # Dataset présent → has_data=True → doit router (après éventuelles
    # désambiguations write/gender). Le test vérifie au minimum qu'on ne
    # tombe PAS sur le refus du Bug 6.
    state = _state(
        "calcule la table",
        dataset_ref="session_abc",
        ds={"_dataset_ref": "session_abc",
            "study_plan": {"gender_segmentation": "unisex"},
            "_write_question_asked": True},
    )

    with patch("agents.master.classify_intent.classify_intent",
               side_effect=_fake_classify):
        result = master_node(state)

    # Pas de message de refus "fichier CSV" / "portefeuille"
    msgs = result.get("messages", [])
    refusal_present = any(
        ("fichier csv" in m.content.lower() or "portefeuille" in m.content.lower())
        and "uploadez" in m.content.lower()
        for m in msgs
    )
    assert not refusal_present, (
        "Régression : refus Bug 6 déclenché malgré dataset présent"
    )
