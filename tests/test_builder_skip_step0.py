"""
HOTFIX-pre-refacto-2026-05 — Bug 21 : l'instruction Builder du chemin pending
doit ordonner de SAUTER l'étape 0 (dictionnaire de données) et lancer les
tools. Sans ça le Builder décrit le fichier et stalle.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import HumanMessage

_ROOT = Path(__file__).resolve().parent.parent


def _state_pending_gender():
    return {
        "messages":   [HumanMessage(content="par sexe")],
        "data_store": {
            "_pending_need": {
                "context_key": "gender_segmentation",
                "question":    "unisex ou by_sex ?",
                "options":     ["unisex", "by_sex"],
            },
            "study_plan":  {},
            "report_mode": "full_report",
        },
        "dataset_ref": "test_session",
    }


def test_pending_instruction_orders_skip_step0() -> None:
    """L'instruction envoyée au Builder doit contenir la directive de saut
    de l'étape 0 + l'ordre de lancer les tools."""
    from agents.mortality.agents import master_node as mn

    with patch("openai.OpenAI"):
        result = mn.master_node(_state_pending_gender())

    assert result.get("active_agent") == "builder"
    instr = (result.get("messages") or [None])[0].content
    assert "NE refais PAS l'étape 0" in instr, f"directive saut étape 0 absente : {instr}"
    assert "LANCE DIRECTEMENT" in instr
    assert "Reste à produire" in instr


def test_step0_md_documents_skip_condition() -> None:
    """step0_data_dictionary.md doit documenter la condition de saut pour que
    le Builder coopère avec l'instruction du Master."""
    src = (_ROOT / "agents/mortality/agent_instructions/step0_data_dictionary.md").read_text(
        encoding="utf-8"
    )
    assert "SAUTER cette étape" in src
    assert "Reste à produire" in src
    assert "NE refais PAS l'étape 0" in src
