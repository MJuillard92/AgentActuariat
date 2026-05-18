"""
agents.rag.pipeline.query_rewriter

Reformulation LLM nano de la question utilisateur en une affirmation de
recherche optimisée pour le retrieval hybride.

Skip si la query est déjà courte (<40 chars) et contient un terme technique
connu — l'appel LLM ne ferait que ralentir sans rien apporter.

En cas d'erreur LLM (timeout, 500, etc.), graceful degradation : on retombe
sur la query d'entrée, le pipeline continue.
"""
from __future__ import annotations

import logging
from pathlib import Path

import openai

from agents.mortality.agents._utils import call_with_retry
from agents.mortality.agents.llm_config import get_llm_config

log = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "agent_instructions" / "query_rewriter_prompt.md"
)

# Termes techniques qui justifient un skip du LLM si la query est déjà courte.
# Liste alignée sur les corrections du query_normalizer — après normalisation,
# ces formes canoniques apparaissent dans la query.
_TECHNICAL_TERMS = (
    "whittaker", "kaplan-meier", "nelson-aalen",
    "gompertz", "makeham",
    "lee-carter", "cairns-blake-dowd", "brouhns-denuit-vermunt",
    "denuit-goderniaux",
    "chi-2", "smr", "a132-18",
    "bcac", "th 00-02", "tf 00-02", "tgh 05", "tgf 05", "tprv 93",
)

_SHORT_QUERY_THRESHOLD = 40


def should_rewrite(query: str) -> bool:
    """Décide si la query mérite un appel au LLM rewriter.

    Skip (False) si la query est courte ET contient un terme technique :
    le retrieval hybride saura traiter une telle requête directement.
    """
    if not query:
        return False
    short = len(query) <= _SHORT_QUERY_THRESHOLD
    if not short:
        return True
    lower = query.lower()
    return not any(term in lower for term in _TECHNICAL_TERMS)


def _load_prompt() -> str:
    """Charge le prompt système depuis le fichier d'instructions."""
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "Reformule la question utilisateur en une requête de recherche "
        "concise (max 15 mots), affirmation de recherche, sans ponctuation finale."
    )


def _strip_decoration(text: str) -> str:
    """Retire guillemets externes et whitespace que le LLM ajoute parfois."""
    t = (text or "").strip()
    while t and t[0] in ('"', "'", "«", "“", "”"):
        t = t[1:].lstrip()
    while t and t[-1] in ('"', "'", "»", "“", "”", "."):
        t = t[:-1].rstrip()
    return t


def rewrite(query: str) -> str:
    """Reformule la query utilisateur en affirmation de recherche.

    Returns:
        La requête reformulée si l'appel LLM réussit, sinon la query d'entrée
        (graceful degradation).
    """
    if not query:
        return ""

    system_prompt = _load_prompt()
    cfg = get_llm_config("rag.query_rewriter")

    try:
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": query},
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
