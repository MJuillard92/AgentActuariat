"""
agents.rag.pipeline.answer_generator

Synthèse rédigée LLM mini à partir des chunks doctrinaux retournés par
search_doctrine, avec citations `[Dxx.yy]` inline et section Sources finale.

Composant le plus critique du pipeline : un appel raté ici fait l'effet
d'une réponse pauvre côté utilisateur, même si normalize/rewrite/retrieve
ont parfaitement fonctionné.

Stratégie de robustesse :
- 0 chunk → réponse déterministe "corpus ne couvre pas" sans appel LLM
- LLM error → réponse dégradée affichant les sources brutes (graceful)
- doc_id dupliqué côté formatage chunks → blindage via _format_chunks_for_prompt
"""
from __future__ import annotations

import logging
import re as _re
from pathlib import Path

import openai

from agents.mortality.agents._utils import call_with_retry
from agents.mortality.agents.llm_config import get_llm_config

log = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent
    / "agent_instructions" / "answer_generator_prompt.md"
)

# Capture les citations [Dxx], [Dxx.y], [Dxx.yy]. Utilisé par
# answer_has_citation() pour le safety check post-RAG.5 dans run_pipeline :
# si chunks fournis mais answer sans citation → rejet (injection probable
# ou hallucination grossière).
_CITATION_RE = _re.compile(r"\[[A-Z]\d{2}(?:\.\d{1,2})?\]")


def answer_has_citation(answer: str) -> bool:
    """True si l'answer contient au moins une citation `[Dxx.yy]`."""
    if not answer:
        return False
    return bool(_CITATION_RE.search(answer))


# ─────────────────────────────────────────────────────────────────────────────
# Formatage des chunks pour injection dans le prompt
# ─────────────────────────────────────────────────────────────────────────────

def _clean_section_id(doc_id: str, section_id: str) -> str:
    """Retourne le section_id sans préfixe doc_id redondant.

    Le retriever émet souvent `section_id` qui commence déjà par `{doc_id}.`,
    donc `f'{doc_id}.{section_id}'` produit 'D03.D03.02'. Cette fonction garantit
    un identifiant propre côté affichage et côté prompt LLM.

    Note : la détection de préfixe utilise un séparateur strict (`{doc_id}.`)
    pour éviter les faux positifs (ex: `doc_id="D1"` et `section_id="D10.02"`
    où "D10" commencerait par "D1" sans être un préfixe sémantique).
    """
    if not section_id:
        return doc_id or ""
    if not doc_id:
        return section_id
    if section_id == doc_id:
        return doc_id
    if section_id.startswith(f"{doc_id}."):
        return section_id
    return f"{doc_id}.{section_id}"


def _format_chunks_for_prompt(chunks: list[dict]) -> str:
    """Formate les chunks pour injection dans le prompt user du LLM.

    Format :
        [D03.02 — Whittaker-Henderson 1D]
        <texte du chunk>

        [D03.04 — Sélection du paramètre h]
        <texte du chunk>
    """
    blocks = []
    for c in chunks:
        sid = _clean_section_id(c.get("doc_id", ""), c.get("section_id", ""))
        title = c.get("section_title", "")
        text = (c.get("text", "") or "").strip()
        header = f"[{sid} — {title}]" if title else f"[{sid}]"
        blocks.append(f"{header}\n{text}")
    return "\n\n".join(blocks)


def _format_sources_section(chunks: list[dict]) -> str:
    """Section 'Sources :' textuelle utilisée en fallback."""
    lines = ["Sources :"]
    for c in chunks:
        sid = _clean_section_id(c.get("doc_id", ""), c.get("section_id", ""))
        title = c.get("section_title", "")
        lines.append(f"- {sid} — {title}" if title else f"- {sid}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

def _load_prompt_template() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "Tu es actuaire expert. Réponds en t'appuyant exclusivement sur les "
        "extraits ci-dessous. Cite chaque affirmation par [Dxx.yy]. Si les "
        "extraits ne couvrent pas la question, dis 'Le corpus ne couvre pas ce "
        "point'. Convertis le LaTeX en Unicode (Σ, Δ, q̃).\n\n"
        "Question : {original_query}\n\nExtraits :\n{chunks}\n\nRéponse :"
    )


def _build_prompt(original_query: str, chunks: list[dict]) -> tuple[str, str]:
    """Retourne (system_prompt, user_prompt) à envoyer au LLM."""
    tmpl = _load_prompt_template()
    formatted_chunks = _format_chunks_for_prompt(chunks)
    # Le template contient déjà la consigne complète + les placeholders {original_query} et {chunks}
    # On l'utilise comme system prompt et on envoie un user prompt minimal pour
    # éviter de dupliquer la query (le template la contient déjà via {original_query}).
    try:
        user_prompt = tmpl.format(original_query=original_query, chunks=formatted_chunks)
    except (KeyError, IndexError):
        # Le template n'a pas les placeholders attendus — fallback ad hoc
        user_prompt = (
            f"{tmpl}\n\nQuestion : {original_query}\n\n"
            f"Extraits :\n{formatted_chunks}\n\nRéponse :"
        )
    system_prompt = (
        "Tu es l'agent RAG. Réponds en français professionnel, 3-6 phrases, "
        "avec citations [Dxx.yy] inline. Suis strictement les règles du prompt user."
    )
    return system_prompt, user_prompt


# ─────────────────────────────────────────────────────────────────────────────
# API publique
# ─────────────────────────────────────────────────────────────────────────────

def generate(original_query: str, chunks: list[dict]) -> str:
    """Génère la réponse rédigée avec citations à partir des chunks fournis.

    Args:
        original_query: la question utilisateur d'origine (non reformulée).
        chunks: liste de dicts comme retournés par search_doctrine
                (`doc_id`, `section_id`, `section_title`, `text`, ...).

    Returns:
        Réponse rédigée prête à afficher (sans signal `<RAG_DONE>` —
        c'est l'adapter LangGraph qui l'ajoute).
    """
    # Cas dégénéré : aucun chunk → réponse déterministe, pas d'appel LLM
    if not chunks:
        return (
            "Le corpus doctrinal ne couvre pas ce point. "
            "Reformulez la question ou consultez la documentation interne."
        )

    system_prompt, user_prompt = _build_prompt(original_query, chunks)
    cfg = get_llm_config("rag.answer_generator")

    try:
        client = openai.OpenAI()
        response = call_with_retry(
            client,
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=cfg.get("temperature", 0.3),
            max_tokens=cfg.get("max_tokens", 1500),
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            raise RuntimeError("LLM returned empty content")
        return answer
    except Exception as exc:
        log.warning("[answer_generator] LLM failure, using degraded fallback: %s", exc)
        # Fallback : 1 phrase d'excuse + sources brutes en clair pour ne pas
        # laisser l'utilisateur sans information utile.
        sources_block = _format_sources_section(chunks)
        return (
            "Une erreur est survenue lors de la rédaction de la réponse. "
            "Les extraits doctrinaux suivants ont été identifiés comme pertinents :\n\n"
            f"{sources_block}"
        )
