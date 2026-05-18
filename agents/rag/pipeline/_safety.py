"""
Garde-fous sécurité Palier 1 du RAG — always-on, zéro coût LLM.

Composants :
- sanitize_input    : truncate + strip control chars (always)
- detect_jailbreak  : ~15 regex FR+EN pour bloquer prompts adversariaux
- is_in_scope       : filtre lexical via corpus_lexicon (refuse hors-actuariat)

Messages de refus polis exposés via REFUSAL_JAILBREAK / REFUSAL_OFF_TOPIC.

Le Palier 2 (LLM scope classifier, rate limit, moderation) est activé par
le flag config `rag.safety.public_mode` — non implémenté en v1.
"""
from __future__ import annotations

import re

from agents.rag.pipeline._corpus_lexicon import get_lexicon

# ── Limites d'input ──────────────────────────────────────────────────────────

MAX_INPUT_CHARS = 2000   # truncate avant tout traitement
SCOPE_MIN_LEN   = 20     # en dessous, on ne filtre pas scope (queries type "merci")


# ── Sanitization ─────────────────────────────────────────────────────────────

def sanitize_input(text: str | None) -> str:
    """Truncate + strip control chars dangereux (préserve \\n \\t)."""
    if not text:
        return ""
    text = text[:MAX_INPUT_CHARS]
    return "".join(c for c in text if c in ("\n", "\t", "\r") or ord(c) >= 32)


# ── Jailbreak detection ─────────────────────────────────────────────────────

_JAILBREAK_PATTERNS: list[re.Pattern[str]] = [
    # Anglais
    re.compile(r"\bignore\s+(all\s+)?(the\s+)?(previous|prior|above)\s+(instructions?|prompts?|rules?)", re.I),
    re.compile(r"\bdisregard\s+(your|the|all)\s+(rules?|instructions?|guidelines?|prompts?)", re.I),
    re.compile(r"\b(reveal|show|print|display|tell\s+me|give\s+me)\s+(your|the)\s+(system\s+prompt|instructions?|rules?|guidelines?)", re.I),
    re.compile(r"\bact\s+as\s+(if\s+)?(you|a)\s+(are|jailbroken|DAN|unrestricted)", re.I),
    re.compile(r"\b(you|tu)\s+(are|es)\s+(now|désormais|maintenant)\s+(DAN|jailbroken|unrestricted)", re.I),
    re.compile(r"\bsystem\s*[:>]\s*\S", re.I),
    re.compile(r"###\s*(instructions?|system|new\s+role)", re.I),
    # Français
    re.compile(r"\bignor[ea](z|s)?\s+(les|tes|vos|toutes\s+les)\s+(consignes?|instructions?|r[èe]gles?|prompts?)\s*(pr[ée]c[ée]dentes?|antérieur)?", re.I),
    re.compile(r"\boubli[ea](z|s)?\s+(tes|vos|les)\s+(r[èe]gles?|consignes?|instructions?|s[ée]curit[ée])", re.I),
    re.compile(r"\b(tu\s+es|vous\s+[êe]tes)\s+(d[ée]sormais|maintenant)\s+(un|une)\s+(assistant|llm|ai|ia)?\s*sans\s+limit", re.I),
    re.compile(r"\b(montre|donne|affiche|r[ée]v[èe]le)[\s\-]?(moi|nous)?\s+(ton|votre|le)\s+(system\s*prompt|prompt\s+system|consigne)", re.I),
    re.compile(r"\bquelles?\s+sont\s+(tes|vos)\s+(r[èe]gles?|consignes?|instructions?)", re.I),
    re.compile(r"\bfais\s+(comme\s+si|semblant)\s+(tu|que)", re.I),
    # Patterns techniques
    re.compile(r"<\s*/?\s*(system|user|assistant|s|u|a)\s*>", re.I),
    re.compile(r"\[\s*(INST|/INST|SYSTEM|/SYSTEM)\s*\]", re.I),
    re.compile(r"```\s*(system|prompt)\b", re.I),
]


def detect_jailbreak(query: str) -> tuple[bool, str | None]:
    """Retourne (is_jailbreak, pattern_matched_str). Pattern_matched_str = aperçu."""
    if not query:
        return (False, None)
    for pat in _JAILBREAK_PATTERNS:
        if pat.search(query):
            return (True, pat.pattern[:60])
    return (False, None)


# ── Scope filter ─────────────────────────────────────────────────────────────

# Marqueurs anaphoriques : si présents + buffer non vide, le rewriter pourra
# résoudre la référence. On laisse passer ces queries malgré l'absence de
# terme actuariel explicite.
_ANAPHORA_PATTERNS = (
    " les ", " ça ", " ca ", "cette ", "celle", "celui",
    "et pour", "et avec", "et sur", "compare",
    "leur ", "leurs ", " son ", " sa ", " ses ",
)


def has_anaphora(query: str) -> bool:
    """Détecte les signaux anaphoriques (les, ça, cette, et pour, compare...)."""
    if not query:
        return False
    padded = f" {query.lower()} "  # padding pour matcher " les ", " ça ", etc.
    return any(p in padded for p in _ANAPHORA_PATTERNS)


def is_in_scope(query: str, anaphora_present: bool = False) -> bool:
    """Vérifie qu'au moins UN terme du corpus actuariel est dans la query.

    Exceptions :
    - Query courte (< SCOPE_MIN_LEN) : on laisse passer (merci, plus, etc.)
    - Anaphore présente : le rewriter pourra résoudre via contexte
    """
    if not query or len(query) < SCOPE_MIN_LEN:
        return True
    if anaphora_present:
        return True
    lexicon = get_lexicon()
    if not lexicon:
        # Corpus indispo : on ne peut pas filtrer scope → on laisse passer
        return True
    query_lower = query.lower()
    return any(term in query_lower for term in lexicon)


# ── Messages de refus ──────────────────────────────────────────────────────

REFUSAL_JAILBREAK = (
    "Cette demande ne peut pas être traitée. L'agent RAG est dédié aux "
    "questions de doctrine actuarielle française. Reformulez votre question "
    "dans ce périmètre."
)

REFUSAL_OFF_TOPIC = (
    "Cette question semble hors du périmètre de la doctrine actuarielle "
    "(mortalité, lissage, validation, tables réglementaires, A132-18, "
    "Solvabilité 2). Reformulez votre demande dans ce périmètre."
)
