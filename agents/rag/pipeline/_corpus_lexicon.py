"""
Lexique technique actuariel auto-derivé depuis le meta.json du corpus FAISS.

Remplace une liste hardcodée non-évolutive : à chaque ré-ingest du corpus
(`python knowledge_base/rag_doctrine/ingest_doctrine.py`), le lexique se
met automatiquement à jour au prochain appel via check mtime.

Usage typique :
    from agents.rag.pipeline._corpus_lexicon import get_lexicon
    lexicon = get_lexicon()   # set[str] de termes lowercase
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_META_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "knowledge_base" / "rag_doctrine" / "index" / "meta.json"
)

# Mots à filtrer : trop courts (<3 chars) ou stop-words FR/EN évidents
_STOP_WORDS = {
    "de", "du", "la", "le", "les", "des", "et", "ou", "un", "une", "à", "au",
    "en", "sur", "pour", "par", "ce", "cette", "the", "of", "and", "or", "in",
    "on", "for", "by", "1d", "2d", "code", "art", "ans",
    # Mots français courts génériques — présents dans les titres de sections
    # de façon incidentelle, mais non-techniques (causeraient des faux positifs
    # dans is_in_scope sur des queries hors-actuariat).
    "loi", "lois",
    "non",
    "vie",
    "exp",
    "cas",
    "avec", "sans", "dans", "vers", "sous",
    "est", "sont", "ont", "peut", "plus", "moins", "deux", "trois", "plusieurs",
    "qui", "que", "quoi", "dont",
    "tout", "tous", "toute", "toutes",
    "tels", "telles",
    "mais", "donc",
    "très", "bien", "alors",
    # Mots anglais courants — non-techniques malgré leur présence éventuelle
    # dans des titres de sections en anglais.
    "is", "are", "was", "were", "will", "would", "can", "may", "has", "had",
    "it", "its", "this", "that", "these", "those",
    "with", "from", "into", "over", "under", "after", "before",
    "not", "no", "yes",
    "all", "any", "some", "each", "every",
}

# Sépare les section_titles en tokens significatifs.
# On garde les termes composés à tirets (Whittaker-Henderson) entiers.
_TOKEN_SPLIT_RE = re.compile(r"[^\w\-]+")


def _significant_tokens(text: str) -> set[str]:
    """Extrait les tokens techniques d'un titre de section."""
    tokens: set[str] = set()
    for tok in _TOKEN_SPLIT_RE.split(text or ""):
        tok = tok.strip().lower()
        if len(tok) < 3 or tok in _STOP_WORDS:
            continue
        tokens.add(tok)
    return tokens


def _load_meta() -> dict:
    """Charge le meta.json du corpus. Lève FileNotFoundError si absent."""
    if not _META_PATH.exists():
        raise FileNotFoundError(
            f"Corpus meta absent : {_META_PATH}. "
            "Lancer : python knowledge_base/rag_doctrine/ingest_doctrine.py"
        )
    with _META_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def build_lexicon_from_meta() -> set[str]:
    """Construit le lexique technique depuis le meta.json du corpus FAISS.

    Inclut : doc_ids, section_ids, tokens significatifs des section_titles, tags.
    Exclut : stop-words, tokens < 3 chars.
    """
    meta = _load_meta()
    terms: set[str] = set()
    for chunk in meta.get("chunks", []):
        if did := chunk.get("doc_id"):
            terms.add(did.lower())
        if sid := chunk.get("section_id"):
            terms.add(sid.lower())
        title = chunk.get("section_title", "")
        terms.update(_significant_tokens(title))
        for tag in chunk.get("tags", []) or []:
            terms.add(tag.lower())
    return terms


_LEXICON_CACHE: set[str] | None = None
_LEXICON_MTIME: float = 0.0


def get_lexicon() -> set[str]:
    """Cache module-level avec invalidation sur mtime de meta.json.

    Retourne un set vide si le meta.json est absent (graceful — le pipeline
    fonctionne quand même, juste sans optimisation lexicale).
    """
    global _LEXICON_CACHE, _LEXICON_MTIME
    try:
        mtime = _META_PATH.stat().st_mtime
    except FileNotFoundError:
        return _LEXICON_CACHE or set()
    if _LEXICON_CACHE is None or mtime > _LEXICON_MTIME:
        _LEXICON_CACHE = build_lexicon_from_meta()
        _LEXICON_MTIME = mtime
    return _LEXICON_CACHE
