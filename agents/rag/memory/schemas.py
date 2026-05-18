"""
Schémas Pydantic de la mémoire conversationnelle RAG.

RAGTurn  : un échange (user_q, rag_answer, sources, timestamp)
RAGSummary : résumé structuré des anciens tours (topics, focus, facts, citations)
"""
from __future__ import annotations

import datetime
from typing import Any, Dict, List

from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.datetime.now().isoformat()


class RAGTurn(BaseModel):
    """Un échange Q/A RAG conservé en mémoire conversationnelle."""
    user_q:     str
    rag_answer: str
    sources:    List[Dict[str, Any]] = Field(default_factory=list)
    timestamp:  str                  = Field(default_factory=_now_iso)


class RAGSummary(BaseModel):
    """Résumé structuré incrémental des tours anciens (au-delà du buffer)."""
    topics_covered:    List[str] = Field(default_factory=list)
    user_focus:        str       = ""
    key_facts_stated:  List[str] = Field(default_factory=list)
    citations_used:    List[str] = Field(default_factory=list)
    n_turns_summarized: int      = 0
    updated_at:        str       = Field(default_factory=_now_iso)
