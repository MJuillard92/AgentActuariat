"""Tests pour Bug 7 — BuilderAgent stages BUILDER.0 → BUILDER.3.

Avant ce fix, l'UI n'affichait que "BuilderAgent actif" pendant les 30s+
d'appel GPT-4o. Maintenant 4 stages encadrent l'activité.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.messages import HumanMessage


def _mock_llm_response(content="Calculs en cours...", tool_calls=None,
                      finish_reason="stop", total_tokens=1234):
    """Construit une réponse OpenAI mockée."""
    response = MagicMock()
    choice = MagicMock()
    choice.finish_reason = finish_reason
    choice.message.content = content
    choice.message.tool_calls = tool_calls or []
    response.choices = [choice]
    usage = MagicMock()
    usage.total_tokens = total_tokens
    usage.prompt_tokens = 1000
    usage.completion_tokens = total_tokens - 1000
    response.usage = usage
    return response


def _build_state(content="calcule la table", data_store=None):
    # Data catalogue complet par défaut (mapping + gender + methods_auto +
    # période confirmée) pour que la gate Builder passe et que les tests
    # valident le comportement NOMINAL du Builder, pas la gate.
    # Plan datacatalogue-gate 2026-05-25.
    ds = dict(data_store or {})
    ds.setdefault("mapping_validated", True)
    ds.setdefault("report_mode", "full_report")
    sp = ds.setdefault("study_plan", {})
    sp.setdefault("gender_segmentation", "unisex")
    sp.setdefault("observation_period_years", [2020, 2024])
    sp.setdefault("start_year", 2020)
    sp.setdefault("end_year", 2024)
    sp.setdefault("num_observation_years", 5)
    sp.setdefault("methods_auto", True)
    return {
        "messages":   [HumanMessage(content=content)],
        "data_store": ds,
        "dataset_ref": None,
    }


def _extract_stage_events(events):
    return [e for e in (events or [])
            if e.get("type") == "master_stage"
            and (e.get("stage") or "").startswith("BUILDER.")]


# ──────────────────────────────────────────────────────────────────────
# Stages présents
# ──────────────────────────────────────────────────────────────────────

def test_builder_emits_stages_on_nominal_path():
    """Sur appel nominal, BUILDER.0/1/2/3 émis dans les events."""
    from agents.mortality.agents.builder_node import builder_node

    state = _build_state()
    fake_response = _mock_llm_response(content="<BUILD_DONE>")
    with patch("openai.OpenAI"), \
         patch("agents.mortality.agents._utils.call_with_retry",
               return_value=fake_response):
        result = builder_node(state)

    stages = _extract_stage_events(result.get("events"))
    stage_ids = [s["stage"] for s in stages]
    assert "BUILDER.0" in stage_ids
    assert "BUILDER.1" in stage_ids
    assert "BUILDER.2" in stage_ids
    assert "BUILDER.3" in stage_ids


def test_builder_stage_2_includes_finish_reason():
    """Le label de BUILDER.2 doit contenir finish_reason."""
    from agents.mortality.agents.builder_node import builder_node

    state = _build_state()
    # finish_reason="stop" suffit pour valider la présence dans le label
    fake_response = _mock_llm_response(content="texte simple", finish_reason="stop")
    with patch("openai.OpenAI"), \
         patch("agents.mortality.agents._utils.call_with_retry",
               return_value=fake_response):
        result = builder_node(state)

    stages = _extract_stage_events(result.get("events"))
    b2 = next((s for s in stages if s["stage"] == "BUILDER.2"), None)
    assert b2 is not None
    # Nouveau label user-friendly (Bug 13) : "Réponse textuelle reçue" ou "Décision : N outil(s)"
    assert ("Réponse textuelle" in b2["label"]) or ("outil" in b2["label"]), b2["label"]


def test_builder_stage_3_detects_build_done_signal():
    """BUILDER.3 doit mentionner BUILD_DONE quand le signal est dans content."""
    from agents.mortality.agents.builder_node import builder_node

    state = _build_state()
    fake_response = _mock_llm_response(content="Calculs terminés <BUILD_DONE>")
    with patch("openai.OpenAI"), \
         patch("agents.mortality.agents._utils.call_with_retry",
               return_value=fake_response):
        result = builder_node(state)

    stages = _extract_stage_events(result.get("events"))
    b3 = next((s for s in stages if s["stage"] == "BUILDER.3"), None)
    assert b3 is not None
    # Nouveau label user-friendly (Bug 13) : "Calculs terminés ✓"
    assert "Calculs terminés" in b3["label"], b3["label"]
    # Et active_agent=master (signal détecté)
    assert result.get("active_agent") == "master"


def test_builder_stage_3_handles_no_signal():
    """Sans signal, BUILDER.3 mentionne 'aucun' et active_agent inchangé."""
    from agents.mortality.agents.builder_node import builder_node

    state = _build_state()
    fake_response = _mock_llm_response(content="En cours...")
    with patch("openai.OpenAI"), \
         patch("agents.mortality.agents._utils.call_with_retry",
               return_value=fake_response):
        result = builder_node(state)

    stages = _extract_stage_events(result.get("events"))
    b3 = next((s for s in stages if s["stage"] == "BUILDER.3"), None)
    assert b3 is not None
    # Nouveau label user-friendly (Bug 13) : "En attente d'instructions"
    assert "attente" in b3["label"].lower(), b3["label"]
    # Pas de transition active_agent
    assert "active_agent" not in result or result.get("active_agent") != "master"
