"""Tests des schémas Pydantic de la mémoire RAG."""
from __future__ import annotations

import pytest


def test_rag_turn_minimal_fields():
    from agents.rag.memory.schemas import RAGTurn
    t = RAGTurn(user_q="qu'est-ce que Whittaker ?",
                rag_answer="Le lissage Whittaker [D03.02]...",
                sources=[{"doc_id": "D03", "section_id": "D03.02"}])
    assert t.user_q.startswith("qu'est-ce")
    assert "[D03.02]" in t.rag_answer
    assert len(t.sources) == 1
    assert t.timestamp  # auto-rempli


def test_rag_turn_serialization_roundtrip():
    from agents.rag.memory.schemas import RAGTurn
    t = RAGTurn(user_q="x", rag_answer="y", sources=[])
    dumped = t.model_dump()
    restored = RAGTurn(**dumped)
    assert restored.user_q == "x"
    assert restored.rag_answer == "y"


def test_rag_summary_minimal_fields():
    from agents.rag.memory.schemas import RAGSummary
    s = RAGSummary(
        topics_covered=["Whittaker-Henderson", "TH 00-02"],
        user_focus="méthodes lissage",
        key_facts_stated=["paramètre h optimisé par CV"],
        citations_used=["D03.02", "D03.04"],
        n_turns_summarized=6,
    )
    assert len(s.topics_covered) == 2
    assert s.user_focus.startswith("méthodes")
    assert s.n_turns_summarized == 6


def test_rag_summary_defaults_empty_lists():
    from agents.rag.memory.schemas import RAGSummary
    s = RAGSummary()
    assert s.topics_covered == []
    assert s.user_focus == ""
    assert s.key_facts_stated == []
    assert s.citations_used == []
    assert s.n_turns_summarized == 0
    assert s.updated_at  # auto-rempli ISO datetime
