"""
agents.rag.pipeline.grounding_check

Vérification optionnelle (off par défaut) : chaque affirmation citée dans
la réponse est-elle effectivement supportée par les chunks fournis ?

Activée via `verify=True` dans run_pipeline.run(). Coût : 1 appel LLM mini
supplémentaire (~$0.0003, +1s de latence).

Heuristique légère sans LLM : si la réponse cite un `[Dxx.yy]` qui n'apparaît
dans aucun chunk fourni, c'est un faux self-grounding évident. Cette
vérification rapide précède l'appel LLM et le shortcut si elle suffit
à invalider.
"""
from __future__ import annotations

import logging
import re

import openai

from agents.mortality.agents._utils import call_with_retry
from agents.mortality.agents.llm_config import get_llm_config

log = logging.getLogger(__name__)

# Capture les citations `[D03]`, `[D03.02]`, `[D03.2]` (1 ou 2 chiffres
# pour la sous-section — défensif au cas où la doctrine évoluerait vers
# un numérotage simple-chiffre).
_CITATION_RE = re.compile(r"\[([A-Z]\d{2}(?:\.\d{1,2})?)\]")


def _extract_citations(answer: str) -> set[str]:
    """Extrait tous les identifiants `[Dxx.yy]` cités dans la réponse."""
    return set(_CITATION_RE.findall(answer or ""))


def _available_sids(chunks: list[dict]) -> set[str]:
    """Retourne l'ensemble des section_id / doc_id disponibles dans les chunks."""
    sids: set[str] = set()
    for c in chunks:
        sid = c.get("section_id") or ""
        did = c.get("doc_id") or ""
        if sid:
            sids.add(sid)
        if did:
            sids.add(did)
    return sids


def verify(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Vérifie que les citations de `answer` sont supportées par `chunks`.

    Returns:
        (ok, reason) — `ok=True` si grounding OK, sinon `reason` explique
        ce qui cloche (ex: "citation [D99.99] absente des chunks").
    """
    citations = _extract_citations(answer)
    if not citations:
        return (False, "aucune citation dans la réponse")

    available = _available_sids(chunks)
    unknown = citations - available
    if unknown:
        return (False, f"citation(s) hors corpus : {sorted(unknown)}")

    # Heuristique légère OK — pour un check sémantique plus fin, déléguer au LLM
    cfg = get_llm_config("rag.grounding_check")
    try:
        client = openai.OpenAI(timeout=30.0)  # HOTFIX-pre-refacto-2026-05 (Bug 4)
        user_payload = (
            f"Réponse :\n{answer}\n\n"
            f"Chunks fournis ({len(chunks)} disponibles) :\n"
            + "\n\n".join(
                f"[{c.get('section_id') or c.get('doc_id')}] {c.get('text', '')}"
                for c in chunks
            )
        )
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {"role": "system",
                 "content": "Tu es un vérificateur de grounding. Réponds UNIQUEMENT par "
                            "'OK' ou 'KO: <raison brève>'. Vérifie que chaque affirmation "
                            "de la réponse est supportée par au moins un chunk."},
                {"role": "user", "content": user_payload},
            ],
            temperature=cfg.get("temperature", 0.0),
            max_tokens=cfg.get("max_tokens", 500),
        )
        verdict = (response.choices[0].message.content or "").strip()
        if verdict.upper().startswith("OK"):
            return (True, "grounding OK")
        return (False, verdict)
    except Exception as exc:
        log.warning("[grounding_check] LLM unavailable, falling back to syntactic check OK: %s", exc)
        return (True, "syntactic check only (LLM unavailable)")
