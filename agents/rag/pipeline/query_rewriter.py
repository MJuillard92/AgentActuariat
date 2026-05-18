"""
agents.rag.pipeline.query_rewriter

Reformulation LLM nano de la question utilisateur en requête de recherche
self-contained, optimisée pour le retriever hybride (FAISS+BM25+RRF).

Multi-turn : le rewriter est la SEULE porte d'entrée pour le contexte
conversationnel. Il reçoit buffer (verbatim) + summary (compact) + vectorstore
top-k (Q/A passés sémantiquement liés) et résout les anaphores ("compare-les",
"et pour", "cette méthode") en produisant une requête autonome.

Skip rules (`should_rewrite`) :
- anaphore + buffer non vide → FORCE
- query déclarative → skip
- courte + terme corpus → skip
- longue/interrogative → rewrite

Graceful degradation : si l'appel LLM échoue, retombe sur la query d'entrée.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import openai

from agents.mortality.agents._utils import call_with_retry
from agents.mortality.agents.llm_config import get_llm_config
from agents.rag.pipeline._corpus_lexicon import get_lexicon

if TYPE_CHECKING:
    from agents.rag.memory.schemas import RAGTurn, RAGSummary

log = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "agent_instructions" / "query_rewriter_prompt.md"
)

SHORT_QUERY_THRESHOLD = 40

# Anaphores qui justifient une réécriture si buffer dispo
_ANAPHORA_PATTERNS = (
    " les ", " ça ", " ca ", "cette ", "celle", "celui",
    "et pour", "et avec", "et sur", "compare",
    "leur ", "leurs ", " son ", " sa ", " ses ",
)

# Marqueurs interrogatifs : distingue question (rewrite utile) de déclaration
# (rewrite inutile, déjà sous forme de recherche)
_INTERROGATIVE_MARKERS = (
    "?", "qu'est", "qu' est", "comment", "pourquoi", "explique",
    "donne-moi", "donne moi", "c'est quoi", "c' est quoi",
    "différence", "difference", "détaille", "aide",
)


def should_rewrite(query: str, buffer_size: int = 0) -> bool:
    """Décide si la query mérite un appel au LLM rewriter.

    Table de décision :
    - empty → False
    - anaphore + buffer > 0 → True (FORCE — c'est le cas multi-turn)
    - pas de marqueur interrogatif → False (déjà déclaratif)
    - courte (<40) + terme du corpus_lexicon présent → False (retriever géra)
    - default → True
    """
    if not query:
        return False
    padded = f" {query.lower()} "

    # PRIORITÉ : anaphore + contexte → toujours rewrite
    if buffer_size > 0 and any(p in padded for p in _ANAPHORA_PATTERNS):
        return True

    # Pas de marqueur interrogatif → déjà déclaratif, skip
    if not any(m in padded for m in _INTERROGATIVE_MARKERS):
        return False

    # Courte + terme corpus → skip
    if len(query) <= SHORT_QUERY_THRESHOLD:
        lexicon = get_lexicon()
        if lexicon and any(term in padded for term in lexicon):
            return False

    return True


def _load_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return ("Reformule la question en requête de recherche concise "
            "(max 15 mots), affirmation, sans ponctuation finale.")


def _format_buffer(buffer: "list[RAGTurn] | None") -> str:
    if not buffer:
        return ""
    lines = ["[Conversation récente]"]
    for i, t in enumerate(buffer):
        idx = len(buffer) - i
        lines.append(f"T-{idx} user: {t.user_q[:400]}")
        lines.append(f"T-{idx} assistant: {t.rag_answer[:400]}")
    return "\n".join(lines)


def _format_summary(summary: "RAGSummary | None") -> str:
    # Gate alignée sur le body : on émet le bloc UNIQUEMENT si au moins un
    # des 3 champs effectivement rendus (topics/user_focus/citations) est non
    # vide. Évite l'orphan header quand seul key_facts_stated est populé,
    # et évite de droper silencieusement les citations.
    if not summary or not (summary.topics_covered or summary.user_focus
                            or summary.citations_used):
        return ""
    lines = ["[Résumé contexte antérieur]"]
    if summary.topics_covered:
        lines.append(f"Topics couverts : {', '.join(summary.topics_covered)}")
    if summary.user_focus:
        lines.append(f"User focus : {summary.user_focus}")
    if summary.citations_used:
        lines.append(f"Citations utilisées : {', '.join(summary.citations_used)}")
    return "\n".join(lines)


def _format_vectorstore_hits(hits: "list[RAGTurn] | None") -> str:
    if not hits:
        return ""
    lines = ["[Échanges passés pertinents]"]
    for h in hits:
        lines.append(f"user: \"{h.user_q[:200]}\"")
        lines.append(f"assistant: \"{h.rag_answer[:300]}\"")
    return "\n".join(lines)


def _build_user_prompt(
    query: str,
    buffer: "list[RAGTurn] | None",
    summary: "RAGSummary | None",
    vectorstore_hits: "list[RAGTurn] | None",
) -> str:
    parts: list[str] = []
    if buf := _format_buffer(buffer):
        parts.append(buf)
    if sm := _format_summary(summary):
        parts.append(sm)
    if vs := _format_vectorstore_hits(vectorstore_hits):
        parts.append(vs)
    parts.append(f"[Nouvelle question]\n{query}")
    parts.append("Requête de recherche :")
    return "\n\n".join(parts)


def _strip_decoration(text: str) -> str:
    t = (text or "").strip()
    while t and t[0] in ('"', "'", "«", "\u201c", "\u201d"):
        t = t[1:].lstrip()
    while t and t[-1] in ('"', "'", "»", "\u201c", "\u201d", "."):
        t = t[:-1].rstrip()
    return t


def rewrite(
    query: str,
    buffer: "list[RAGTurn] | None" = None,
    summary: "RAGSummary | None" = None,
    vectorstore_hits: "list[RAGTurn] | None" = None,
) -> str:
    """Reformule la query en affirmation de recherche self-contained.

    Args:
        query: nouvelle question utilisateur
        buffer: derniers RAGTurn (verbatim)
        summary: RAGSummary courant (compact)
        vectorstore_hits: top-k RAGTurn passés sémantiquement similaires

    Returns:
        Requête reformulée ou query d'entrée (graceful fallback).
    """
    if not query:
        return ""

    system_prompt = _load_prompt()
    user_prompt   = _build_user_prompt(query, buffer, summary, vectorstore_hits)
    cfg = get_llm_config("rag.query_rewriter")

    try:
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 200),
        )
        raw = response.choices[0].message.content or ""
        rewritten = _strip_decoration(raw)
        return rewritten or query
    except Exception as exc:
        log.warning("[query_rewriter] LLM failure, falling back to input: %s", exc)
        return query
