"""
agents.rag.pipeline.run_pipeline

Orchestrateur pur du pipeline RAG en 8 étapes :

  RAG.0     Hydrate mémoire conversationnelle (cache module-level)
  RAG.1     Extract user query (dernier HumanMessage)
  RAG.0bis  Pre-filter sécurité (sanitize + jailbreak + scope)
  RAG.2     Normalize typos
  RAG.3     Rewrite multi-turn (buffer + summary + vectorstore)
  RAG.4     Hybrid retrieval (FAISS+BM25+RRF)
  RAG.5     Generate (LLM mini, citations groundées)
  RAG.5sec  Citation check (rejet si missing alors chunks présents)
  RAG.6     Self-check optionnel (grounding_check, verify=True)
  RAG.7     Append turn en mémoire (sanitization + summary trigger)

Pure logique. L'adapter LangGraph `rag_node` se charge du wiring state.
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
from agents.rag.pipeline._safety import (
    sanitize_input,
    detect_jailbreak,
    is_in_scope,
    has_anaphora,
    REFUSAL_JAILBREAK,
    REFUSAL_OFF_TOPIC,
)
from agents.rag.memory.rag_memory_store import (
    RAGMemoryStore,
    BUFFER_SIZE,
    VECTORSTORE_TOP_K,
    VECTORSTORE_MIN_SCORE,
)

log = logging.getLogger(__name__)


def _extract_user_query(messages: list) -> str:
    """Dernier HumanMessage non vide (en remontant l'historique)."""
    for m in reversed(messages or []):
        if isinstance(m, HumanMessage):
            content = getattr(m, "content", "") or ""
            if content:
                return str(content)
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content") or ""
            if content:
                return str(content)
    return ""


def run(state: dict, verify: bool = False) -> dict[str, Any]:
    """Exécute le pipeline RAG multi-turn + sécurité.

    Args:
        state: dict LangGraph avec au moins `messages` et `session_id`.
        verify: si True, exécute RAG.6 (grounding_check).

    Returns:
        {"answer": str, "sources": list, "stage_events": list[(id, label)]}
    """
    messages   = state.get("messages") or []
    session_id = state.get("session_id") or "default"
    stage_events: list[tuple[str, str]] = []

    # ── RAG.0 — Hydrate memory ──────────────────────────────────────────
    # IMPORTANT : on passe history[:-1] pour exclure le tour courant
    # (encore non répondu), évitant double-indexing dans le vectorstore
    # quand append_turn (RAG.7) ajoutera le tour final.
    history_for_rebuild = messages[:-1] if messages else []
    memory = RAGMemoryStore.for_session(session_id, history=history_for_rebuild)
    has_summary = memory.get_summary() is not None
    stage_events.append((
        "RAG.0",
        f"Mémoire hydratée (buffer={len(memory.get_buffer())}, summary={'oui' if has_summary else 'non'})",
    ))

    # ── RAG.1 — Extract query ───────────────────────────────────────────
    user_query = _extract_user_query(messages)
    if not user_query:
        stage_events.append(("RAG.1", "Aucune question détectée"))
        return {
            "answer": "Je n'ai pas identifié de question. Reformulez votre demande.",
            "sources": [],
            "stage_events": stage_events,
        }
    stage_events.append(("RAG.1", f"Question extraite ({len(user_query)} chars)"))

    # ── RAG.0bis — Pre-filter sécurité ──────────────────────────────────
    user_query = sanitize_input(user_query)
    is_jb, pattern = detect_jailbreak(user_query)
    if is_jb:
        log.warning("[rag.safety] jailbreak bloqué : %s", pattern)
        stage_events.append(("RAG.0bis", f"Jailbreak bloqué : {pattern[:30]}"))
        return {"answer": REFUSAL_JAILBREAK, "sources": [], "stage_events": stage_events}

    # ── RAG.2 — Normalize ───────────────────────────────────────────────
    normalized = query_normalizer.normalize(user_query)
    stage_events.append(("RAG.2",
        f"Typos normalisés : '{normalized}'" if normalized != user_query
        else "Aucune typo détectée"))

    # Scope filter (utilise normalized + has_anaphora + buffer non vide)
    anaphora = has_anaphora(normalized)
    in_scope = is_in_scope(
        normalized,
        anaphora_present=(anaphora and len(memory.get_buffer()) > 0),
    )
    if not in_scope:
        log.info("[rag.safety] hors-scope refusé : %s", normalized[:60])
        stage_events.append(("RAG.0bis", "Hors-scope actuariel — refus"))
        return {"answer": REFUSAL_OFF_TOPIC, "sources": [], "stage_events": stage_events}
    stage_events.append(("RAG.0bis", "Pre-filter sécurité pass"))

    # ── RAG.3 — Rewrite multi-turn ──────────────────────────────────────
    buffer = memory.get_buffer(n=BUFFER_SIZE)
    if query_rewriter.should_rewrite(normalized, buffer_size=len(buffer)):
        vs_hits = memory.retrieve_similar(
            normalized, k=VECTORSTORE_TOP_K, min_score=VECTORSTORE_MIN_SCORE,
        )
        rewritten = query_rewriter.rewrite(
            normalized,
            buffer=buffer,
            summary=memory.get_summary(),
            vectorstore_hits=vs_hits,
        )
        srcs = []
        if buffer:               srcs.append("buffer")
        if memory.get_summary(): srcs.append("summary")
        if vs_hits:              srcs.append("vectorstore")
        stage_events.append((
            "RAG.3",
            f"Reformulation LLM (sources : {'+'.join(srcs) or 'aucune'}) : '{rewritten}'",
        ))
    else:
        rewritten = normalized
        stage_events.append(("RAG.3", "Skip — query déjà self-contained"))

    # ── RAG.4 — Retrieve ────────────────────────────────────────────────
    from tools.conversation import search_doctrine
    try:
        hits = search_doctrine.run(None, {"query": rewritten, "k": 5})
    except Exception as exc:
        log.error("[run_pipeline] retriever failure: %s", exc)
        stage_events.append(("RAG.4", f"Retrieval échoué : {exc}"))
        return {
            "answer": f"Impossible d'interroger la doctrine : {exc}",
            "sources": [], "stage_events": stage_events,
        }
    if "erreur" in hits:
        stage_events.append(("RAG.4", f"Retrieval échoué : {hits['erreur']}"))
        return {
            "answer": f"Impossible d'interroger la doctrine : {hits['erreur']}",
            "sources": [], "stage_events": stage_events,
        }
    chunks = hits.get("results") or []
    stage_events.append(("RAG.4", f"Retrieval hybride : {len(chunks)} chunks"))

    # ── RAG.5 — Generate ────────────────────────────────────────────────
    answer = answer_generator.generate(user_query, chunks)
    stage_events.append(("RAG.5", "Synthèse rédigée avec citations"))

    # ── RAG.5-safety — Citation obligatoire ─────────────────────────────
    if chunks and not answer_generator.answer_has_citation(answer):
        log.warning("[rag.safety] answer sans citation rejetée (injection probable)")
        answer = REFUSAL_JAILBREAK
        stage_events.append(("RAG.5-safety", "Réponse rejetée : citation manquante"))

    # ── RAG.6 — Verify (optionnel) ──────────────────────────────────────
    if verify and chunks:
        ok, reason = grounding_check.verify(answer, chunks)
        stage_events.append(("RAG.6", f"Self-check : {'OK' if ok else reason}"))

    # ── RAG.7 — Persist turn ────────────────────────────────────────────
    summary_before = memory.get_summary()
    memory.append_turn(user_q=user_query, rag_answer=answer, sources=chunks)
    summary_after = memory.get_summary()
    summary_changed = summary_after is not summary_before
    stage_events.append((
        "RAG.7",
        f"Mémoire mise à jour (summary regénéré : {'oui' if summary_changed else 'non'})",
    ))

    return {
        "answer":       answer,
        "sources":      chunks,
        "stage_events": stage_events,
    }
