"""
agents.rag.pipeline.run_pipeline

Orchestrateur pur du pipeline RAG en 5 étapes (+ self-check optionnel) :

  RAG.1  Extract     — récupère le dernier HumanMessage utilisateur
  RAG.2  Normalize   — typo correction déterministe (Python pur)
  RAG.3  Rewrite     — reformulation LLM nano (skip si query courte+technique)
  RAG.4  Retrieve    — search_doctrine (FAISS+BM25+RRF)
  RAG.5  Generate    — answer_generator (LLM mini, citations groundées)
  RAG.6  Verify      — grounding_check (LLM mini, off par défaut)

Pure logique, pas de LangGraph : l'adapter `rag_node` se charge de l'intégrer
au state LangGraph et au stage_buffer du data_store.

Retour : {"answer": str, "sources": list[dict], "stage_events": list[(stage_id, label)]}
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage

from agents.rag.pipeline import (
    query_normalizer,
    query_rewriter,
    answer_generator,
    grounding_check,
)

log = logging.getLogger(__name__)


def _extract_user_query(messages: list) -> str:
    """Retourne le dernier HumanMessage non vide du state.

    On scanne de la fin vers le début — la convention LangChain place les
    messages plus récents en queue de liste.
    """
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            content = getattr(m, "content", "") or ""
            if content:
                return str(content)
        # Fallback dict (certains nodes émettent du brut)
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content") or ""
            if content:
                return str(content)
    return ""


def run(state: dict, verify: bool = False) -> dict[str, Any]:
    """Exécute le pipeline RAG et retourne {answer, sources, stage_events}.

    Args:
        state: dict LangGraph (clé `messages` au minimum).
        verify: si True, exécute RAG.6 (grounding_check).
    """
    messages = state.get("messages") or []
    stage_events: list[tuple[str, str]] = []

    # ── RAG.1 — Extract ──────────────────────────────────────────────────
    user_query = _extract_user_query(messages)
    if not user_query:
        stage_events.append(("RAG.1", "Aucune question utilisateur détectée"))
        return {
            "answer": "Je n'ai pas identifié de question dans le tour courant. "
                      "Reformulez votre demande.",
            "sources": [],
            "stage_events": stage_events,
        }
    stage_events.append(("RAG.1", f"Question extraite ({len(user_query)} chars)"))

    # ── RAG.2 — Normalize ────────────────────────────────────────────────
    normalized = query_normalizer.normalize(user_query)
    if normalized != user_query:
        stage_events.append(("RAG.2", f"Typos normalisés : '{normalized}'"))
    else:
        stage_events.append(("RAG.2", "Aucune typo détectée"))

    # ── RAG.3 — Rewrite (conditionnel) ───────────────────────────────────
    if query_rewriter.should_rewrite(normalized):
        rewritten = query_rewriter.rewrite(normalized)
        stage_events.append(("RAG.3", f"Reformulation LLM : '{rewritten}'"))
    else:
        rewritten = normalized

    # ── RAG.4 — Retrieve ─────────────────────────────────────────────────
    # Import paresseux : éviter de payer le coût d'import du retriever
    # quand le pipeline n'est pas appelé (cold start canvas).
    from tools.conversation import search_doctrine

    try:
        hits = search_doctrine.run(None, {"query": rewritten, "k": 5})
    except Exception as exc:
        log.error("[run_pipeline] retriever failure: %s", exc)
        stage_events.append(("RAG.4", f"Retrieval échoué : {exc}"))
        return {
            "answer": f"Impossible d'interroger la doctrine actuarielle : {exc}",
            "sources": [],
            "stage_events": stage_events,
        }

    if "erreur" in hits:
        stage_events.append(("RAG.4", f"Retrieval échoué : {hits['erreur']}"))
        return {
            "answer": f"Impossible d'interroger la doctrine actuarielle : {hits['erreur']}",
            "sources": [],
            "stage_events": stage_events,
        }

    chunks = hits.get("results") or []
    stage_events.append(("RAG.4", f"Retrieval hybride : {len(chunks)} chunks"))

    # ── RAG.5 — Generate ─────────────────────────────────────────────────
    # On passe la query ORIGINALE au générateur (pas la reformulée) — le LLM
    # doit répondre à ce que l'utilisateur a vraiment demandé, en s'appuyant
    # sur les chunks ramenés par la version reformulée.
    answer = answer_generator.generate(user_query, chunks)
    stage_events.append(("RAG.5", "Synthèse rédigée avec citations"))

    # ── RAG.6 — Verify (optionnel) ───────────────────────────────────────
    if verify and chunks:
        ok, reason = grounding_check.verify(answer, chunks)
        stage_events.append(("RAG.6", f"Self-check : {'OK' if ok else reason}"))

    return {
        "answer": answer,
        "sources": chunks,
        "stage_events": stage_events,
    }
