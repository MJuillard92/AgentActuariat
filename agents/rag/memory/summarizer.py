"""
Summarizer LLM nano pour la compaction des tours anciens de la mémoire RAG.

Appelé synchronement par RAGMemoryStore.append_turn() quand le nombre total
de tours dépasse SUMMARY_TRIGGER. Pénalité latence ~1s sur ~1 tour sur 5.

Mode incrémental : reçoit le summary existant + nouveaux tours, produit
le summary mis à jour. Pas de regen from scratch (évite la dérive).

Graceful : si le LLM échoue (timeout, parse JSON), retombe sur le summary
existant (ou un summary vide si aucun n'existait). Le pipeline continue.
"""
from __future__ import annotations

import json
import logging

import openai

from agents.mortality.agents._utils import call_with_retry
from agents.mortality.agents.llm_config import get_llm_config
from agents.rag.memory.schemas import RAGSummary, RAGTurn

log = logging.getLogger(__name__)


_SYSTEM_PROMPT = (
    "Tu es archiviste conversationnel. Tu maintiens un résumé STRUCTURÉ JSON "
    "de la conversation RAG actuarielle. Tu réponds UNIQUEMENT en JSON valide. "
    "Tu suis strictement les règles du prompt user."
)


def _format_turns(turns: list[RAGTurn]) -> str:
    lines: list[str] = []
    for i, t in enumerate(turns):
        lines.append(f"T-{len(turns) - i} user: {t.user_q[:400]}")
        lines.append(f"T-{len(turns) - i} assistant: {t.rag_answer[:400]}")
    return "\n".join(lines)


def _build_user_prompt(old_turns: list[RAGTurn], existing: RAGSummary | None) -> str:
    existing_block = (
        existing.model_dump_json(indent=2) if existing
        else '"vide (première compaction)"'
    )
    return (
        f"Résumé existant (à enrichir, pas à écraser) :\n{existing_block}\n\n"
        f"Nouveaux tours à intégrer :\n{_format_turns(old_turns)}\n\n"
        "Produis le résumé mis à jour au format JSON strict :\n"
        '{"topics_covered": ["..."], "user_focus": "...", '
        '"key_facts_stated": ["..."], "citations_used": ["..."], '
        '"n_turns_summarized": <int>}\n\n'
        "Règles :\n"
        "- max 8 topics_covered (déduplique les variantes)\n"
        "- max 10 key_facts_stated\n"
        "- user_focus en 1 phrase, max 15 mots\n"
        "- citations_used : liste des [Dxx.yy] vus dans les tours\n"
        "- Conserver l'existant SAUF si contredit par les nouveaux tours\n"
        "- Ne pas inventer"
    )


def summarize_old_turns(
    old_turns: list[RAGTurn],
    existing: RAGSummary | None = None,
) -> RAGSummary:
    """Génère/met à jour le RAGSummary à partir des tours anciens.

    Returns:
        Le summary mis à jour. En cas d'erreur LLM, retombe sur `existing`
        (ou RAGSummary() vide si existing is None).
    """
    if not old_turns:
        return existing or RAGSummary()

    cfg = get_llm_config("rag.summarizer")
    try:
        client = openai.OpenAI(timeout=30.0)  # HOTFIX-pre-refacto-2026-05 (Bug 4)
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_prompt(old_turns, existing)},
            ],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 500),
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content or "{}"
        payload = json.loads(raw)
        return RAGSummary(**payload)
    except Exception as exc:
        log.warning("[rag.summarizer] LLM/parse failure, keeping existing: %s", exc)
        return existing or RAGSummary()
