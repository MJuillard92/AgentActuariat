"""
agents/report/pipeline/03_completion_plan.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ÉTAPE 3 — RAG (parallèle)

Reçoit le ReportPlan validé (étape 02).
Pour chaque section prioritaire, interroge le corpus RAG (search_exemplars)
en parallèle (ThreadPoolExecutor, max 4 workers — I/O bound).

Le résultat est un ReportPlan enrichi : chaque SectionPlan reçoit un
bloc ## Exemples de rédaction injecté dans son prompt.

Si le corpus est vide ou sans résultat pertinent : section inchangée — non bloquant.

Interface publique :
    complete_plan(plan, data_store) -> ReportPlan   (plan enrichi)
"""
from __future__ import annotations

import concurrent.futures
import logging
from copy import deepcopy
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# Nombre de chunks RAG demandés par section
_N_RESULTS = 3

# Score de distance max acceptable (ChromaDB — plus petit = plus proche)
# Au-dessus de ce seuil, l'extrait est ignoré (pas assez pertinent)
_MAX_DISTANCE = 1.2

# Sections pour lesquelles le RAG apporte le plus de valeur
# (les sections purement chiffrées comme annex n'en ont pas besoin)
_RAG_PRIORITY_SECTIONS = {
    "preamble", "data_submission", "construction",
    "obs_vs_modeled", "regulatory_positioning", "conclusion",
}


# ── Query RAG par section ─────────────────────────────────────────────────────

def _query_for_section(section_id: str, label: str) -> str:
    """Retourne la query RAG depuis llm_directives.rag_query du YAML."""
    try:
        from knowledge_base.report_template.template_loader import load_section
        sec = load_section(section_id)
        q = (sec.llm_directives or {}).get("rag_query")
        if q:
            return q
    except Exception:
        pass
    return f"rédaction professionnelle de la section '{label}' d'un rapport actuariel"


# ── Appel search_exemplars ────────────────────────────────────────────────────

def _search_rag(query: str, n_results: int = _N_RESULTS) -> list[dict]:
    """
    Appelle search_exemplars et retourne les chunks pertinents (hors guide de style).
    Retourne [] si le corpus est vide ou si search_exemplars est indisponible.
    """
    try:
        from tools.build_pdf.search_exemplars import run as _search_run
        # On demande un peu plus que _N_RESULTS : le style_guide capture souvent
        # le top-1 sémantique, on le filtre pour le réinjecter à part.
        result = _search_run(data={}, params={
            "query": query,
            "n_results": n_results + 1,
        })

        if not result.get("chunks"):
            return []

        chunks = result["chunks"]

        # Exclure le style_guide : on l'ajoute séparément via _fetch_style_guide
        # pour garantir sa présence systématique.
        chunks = [c for c in chunks if _get_section_id(c) != "_style_guide"]
        return chunks[:n_results]

    except Exception as exc:
        log.debug("[03_completion_plan] search_exemplars indisponible : %s", exc)
        return []


def _get_section_id(chunk: dict) -> str:
    """section_id peut être dans `section` (search_exemplars) ou meta['section_id']."""
    meta = chunk.get("metadata") or {}
    return meta.get("section_id") or chunk.get("section") or ""


# ── Fetch du guide de style (mis en cache pour toute la durée du pipeline) ────

_STYLE_GUIDE_CACHE: dict = {"fetched": False, "chunk": None}


def _fetch_style_guide() -> dict | None:
    """
    Récupère le chunk `_style_guide` via search_exemplars (filtre par section).
    Mis en cache : ne fait qu'un seul appel par run de pipeline.
    Retourne None si le corpus n'a pas encore été peuplé.
    """
    if _STYLE_GUIDE_CACHE["fetched"]:
        return _STYLE_GUIDE_CACHE["chunk"]

    _STYLE_GUIDE_CACHE["fetched"] = True
    try:
        from tools.build_pdf.search_exemplars import run as _search_run
        result = _search_run(data={}, params={
            "query": "guide de style tournures conventions typographiques rapport actuariel",
            "n_results": 5,
            "filters": {"section_id": "_style_guide"},
        })
        for c in result.get("chunks", []):
            if _get_section_id(c) == "_style_guide":
                _STYLE_GUIDE_CACHE["chunk"] = c
                return c
    except Exception as exc:
        log.debug("[03_completion_plan] fetch style_guide indisponible : %s", exc)
    return None


# ── Formatage des extraits bruts ──────────────────────────────────────────────

def _chunk_text(chunk: dict) -> str:
    """
    Extrait le texte du chunk. `search_exemplars` remonte `contenu` (français) ;
    on accepte aussi `content` / `document` pour tolérance aux autres producteurs.
    """
    return (
        chunk.get("contenu")
        or chunk.get("content")
        or chunk.get("document")
        or ""
    ).strip()


def _chunk_source(chunk: dict, fallback: str = "rapport") -> str:
    meta = chunk.get("metadata") or {}
    return (
        meta.get("rapport_id")
        or meta.get("source")
        or chunk.get("rapport_id")
        or chunk.get("source")
        or fallback
    )


# Limite de troncature par extrait dans le prompt. 1800 chars = ~450 tokens,
# suffisant pour transmettre le style d'une section entière sans saturer.
_EXTRACT_MAX_CHARS = 1800


def _format_chunks(chunks: list[dict], style_guide: dict | None = None) -> str:
    """
    Formate les extraits RAG bruts pour injection dans le prompt de rédaction.
    L'agent de rédaction voit le texte réel des rapports de référence :
    wording, niveau de détail, longueur — sans transformation.

    Si `style_guide` est fourni, il est injecté AVANT les extraits (rôle :
    établir les conventions avant de donner les exemples).
    """
    if not chunks and not style_guide:
        return ""

    lines = [
        "## Exemples de rédaction issus du corpus de référence",
        "Les éléments ci-dessous proviennent de rapports actuariels réels.",
        "Ils te donnent le ton, le niveau de détail et la longueur attendus.",
        "Ne les copie pas — inspire-toi du style.",
        "",
    ]

    if style_guide:
        sg_text = _chunk_text(style_guide)
        if sg_text:
            lines += [
                "### Guide de style (tournures et conventions du rapport de référence)",
                sg_text[:_EXTRACT_MAX_CHARS],
                "",
            ]

    for i, chunk in enumerate(chunks):
        content = _chunk_text(chunk)
        if not content:
            continue
        source = _chunk_source(chunk, fallback=f"rapport_{i+1}")
        lines += [
            f"### Extrait {i+1} — {source}",
            content[:_EXTRACT_MAX_CHARS],
            "",
        ]

    return "\n".join(lines)


# ── Point d'entrée public ─────────────────────────────────────────────────────

def _fetch_rag_for_section(sec, style_guide: dict | None = None) -> tuple[str, str, int]:
    """
    Interroge le corpus RAG pour une section donnée.
    Thread-safe — pas d'état partagé (le style_guide est pré-fetché à l'entrée
    de complete_plan et passé par paramètre, pas via globale).
    Retourne (section_id, rag_block, n_chunks_injected).
    """
    query  = _query_for_section(sec.section_id, sec.label)
    chunks = _search_rag(query)

    # Filtrage par distance (pertinence). Les chunks avec distance > _MAX_DISTANCE
    # sont ignorés. NB : search_exemplars remonte `score = 1 - distance`.
    def _dist(c):
        score = c.get("score")
        if score is not None:
            try:
                return 1.0 - float(score)
            except (TypeError, ValueError):
                return 0.0
        d = c.get("distance")
        return float(d) if d is not None else 0.0

    chunks = [c for c in chunks if _dist(c) <= _MAX_DISTANCE]

    # Même sans chunks pertinents, on injecte le guide de style seul (c'est sa raison d'être).
    if not chunks and not style_guide:
        return sec.section_id, "", 0

    block = _format_chunks(chunks, style_guide=style_guide)
    n_items = len(chunks) + (1 if style_guide else 0)
    return sec.section_id, block, n_items


def complete_plan(plan, data_store: dict):
    """
    Enrichit chaque SectionPlan du ReportPlan avec des exemples RAG.

    Les recherches RAG sont I/O-bound (ChromaDB) → parallélisées via
    ThreadPoolExecutor (max 4 workers).

    Args:
        plan       : ReportPlan produit par 01_load_plan (validé par 02)
        data_store : non utilisé ici, conservé pour interface homogène

    Returns:
        ReportPlan enrichi (copie profonde — le plan original n'est pas modifié)
    """
    enriched_plan = deepcopy(plan)

    # Sections éligibles au RAG
    eligible = [
        sec for sec in enriched_plan.sections
        if sec.section_id in _RAG_PRIORITY_SECTIONS and sec.ready
    ]

    if not eligible:
        log.info("[03_completion_plan] aucune section éligible au RAG")
        return enriched_plan

    log.info("[03_completion_plan] %d sections à enrichir (parallèle)", len(eligible))

    # Pré-fetch du guide de style une seule fois (commun à toutes les sections).
    # Reset du cache pour cette exécution du pipeline, puis fetch.
    _STYLE_GUIDE_CACHE["fetched"] = False
    _STYLE_GUIDE_CACHE["chunk"]   = None
    style_guide = _fetch_style_guide()
    if style_guide:
        log.info("[03_completion_plan] guide de style récupéré (sera injecté dans chaque section)")

    # Recherches RAG en parallèle (I/O bound — pas de conflit)
    # Timeout par section : 20s max (ChromaDB + embedding model init peut être lent)
    _RAG_TIMEOUT = 20

    rag_results: dict[str, tuple[str, int]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_fetch_rag_for_section, sec, style_guide): sec.section_id
            for sec in eligible
        }
        for future in concurrent.futures.as_completed(futures, timeout=_RAG_TIMEOUT * len(eligible)):
            sid = futures[future]
            try:
                section_id, rag_block, n_chunks = future.result(timeout=_RAG_TIMEOUT)
                rag_results[section_id] = (rag_block, n_chunks)
            except concurrent.futures.TimeoutError:
                log.warning("[03_completion_plan] '%s' — RAG timeout (%ds), section ignorée", sid, _RAG_TIMEOUT)
                rag_results[sid] = ("", 0)
            except Exception as exc:
                log.debug("[03_completion_plan] '%s' — RAG échoué : %s", sid, exc)
                rag_results[sid] = ("", 0)

    # Injection dans le plan (séquentiel — ordre préservé)
    n_enriched = 0
    for sec in enriched_plan.sections:
        rag_block, n_chunks = rag_results.get(sec.section_id, ("", 0))
        if rag_block:
            sec.prompt += "\n\n" + rag_block
            n_enriched += 1
            log.info("[03_completion_plan] '%s' — %d extrait(s) injecté(s)",
                     sec.section_id, n_chunks)
        elif sec.section_id in _RAG_PRIORITY_SECTIONS and sec.ready:
            log.info("[03_completion_plan] '%s' — aucun extrait (corpus vide ou hors seuil)",
                     sec.section_id)

    log.info("[03_completion_plan] terminé — %d sections enrichies sur %d éligibles",
             n_enriched, len(eligible))
    return enriched_plan
